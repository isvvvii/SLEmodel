# utils.py

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import ot
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Loss
# =============================================================================
class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weights."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Propensity score classifier helpers
# =============================================================================
def train_classifier(model, features, labels, device, epochs: int = 100, batch_size: int = 32):
    """Fit a classifier for propensity-score estimation."""
    model.to(device)
    model.train()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(int(epochs)):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def get_propensity_scores(classifier, features: torch.Tensor, device: str) -> torch.Tensor:
    """Return softmax propensity scores."""
    classifier.eval()
    features = features.to(device)
    logits = classifier(features)
    ps = torch.softmax(logits, dim=1)
    return ps.cpu()


def calibrate_propensity_scores(
    ps_scores: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
) -> Tuple[torch.Tensor, float]:
    """
    Calibrate propensity scores by temperature scaling. Softmax probabilities
    are converted to log probabilities before scaling.
    """
    ps_scores = ps_scores.clone()
    ps_scores = torch.clamp(ps_scores, min=1e-7, max=1 - 1e-7)
    logits = torch.log(ps_scores)

    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)
    labels_long = labels.long()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        scaled_logits = logits / temperature
        loss = F.cross_entropy(scaled_logits, labels_long)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(torch.exp(log_temperature).item())
    temperature = max(temperature, 0.1)
    temperature = min(temperature, 10.0)

    calibrated_logits = logits / temperature
    calibrated_ps = F.softmax(calibrated_logits, dim=1)
    return calibrated_ps.detach(), temperature


def get_propensity_scores_calibrated(
    classifier,
    features: torch.Tensor,
    labels: Optional[torch.Tensor],
    device: str,
    calibrate: bool = True,
    max_iter: int = 100,
    lr: float = 0.01,
) -> Tuple[torch.Tensor, float]:
    """
    Return propensity scores and the fitted temperature. The temperature is
    one when calibration is disabled.
    """
    ps_scores = get_propensity_scores(classifier, features, device)
    if calibrate and labels is not None:
        calibrated_ps, temperature = calibrate_propensity_scores(ps_scores, labels, max_iter=max_iter, lr=lr)
        return calibrated_ps, temperature
    return ps_scores, 1.0


# =============================================================================
# OT helpers
# =============================================================================
def _compute_ps_cost(ps_t: np.ndarray, ps_s: np.ndarray, metric: str = "sqeuclidean") -> np.ndarray:
    """
    Compute pairwise cost between probability vectors on simplex.
    Supported:
      - "sqeuclidean": ||p-q||^2
      - "hellinger":  ||sqrt(p)-sqrt(q)||^2  (proportional to Hellinger^2)
    """
    m = (metric or "sqeuclidean").lower().strip()
    if m == "hellinger":
        ps_t2 = np.sqrt(np.clip(ps_t, 0.0, 1.0))
        ps_s2 = np.sqrt(np.clip(ps_s, 0.0, 1.0))
        return ot.dist(ps_t2, ps_s2, metric="sqeuclidean").astype(np.float64)
    return ot.dist(ps_t, ps_s, metric="sqeuclidean").astype(np.float64)


def _solve_sinkhorn_plan(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    reg: float,
    numItermax: int,
    stopThr: float,
    purpose: str = "eval",  # "train" or "eval"
) -> np.ndarray:
    """
    purpose="train": fast approximate OT (few iters, no multi-method retries).
    purpose="eval": use stabilized or log-domain solvers before standard Sinkhorn.

    Returns:
        G: nonnegative coupling matrix (NOT row-stochastic; row-normalize outside).
    """
    import warnings

    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    p = (purpose or "eval").lower().strip()

    def _run(method: str):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            try:
                G = ot.sinkhorn(
                    a, b, cost, reg,
                    method=method,
                    numItermax=numItermax,
                    stopThr=stopThr,
                    verbose=False,
                    warn=False,
                )
                did_not_converge = any("did not converge" in str(w.message).lower() for w in wlist)
                return np.asarray(G, dtype=np.float64), did_not_converge
            except TypeError:
                pass

            if method == "sinkhorn_log":
                G = ot.bregman.sinkhorn_log(a, b, cost, reg, numItermax=numItermax, stopThr=stopThr)
            elif method == "sinkhorn_stabilized":
                G = ot.bregman.sinkhorn_stabilized(a, b, cost, reg, numItermax=numItermax, stopThr=stopThr)
            else:
                G = ot.sinkhorn(a, b, cost, reg, numItermax=numItermax, stopThr=stopThr)

            did_not_converge = any("did not converge" in str(w.message).lower() for w in wlist)
            return np.asarray(G, dtype=np.float64), did_not_converge

    if p == "train":
        method = "sinkhorn_stabilized"
        try:
            G, did_not_converge = _run(method)
        except Exception:
            G, did_not_converge = _run("sinkhorn")

        if did_not_converge:
            logger.warning("Sinkhorn not converged (train, reg=%.2e, iters=%d). Using approximate plan.", reg, numItermax)

        G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        G[G < 0] = 0.0
        return G

    for method in ("sinkhorn_log", "sinkhorn_stabilized", "sinkhorn"):
        try:
            G, did_not_converge = _run(method)
            if did_not_converge:
                logger.warning("Sinkhorn not converged (eval, method=%s, reg=%.2e, iters=%d).", method, reg, numItermax)
            G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
            G[G < 0] = 0.0
            return G
        except Exception:
            continue

    raise RuntimeError(f"Sinkhorn failed for reg={reg} (purpose={purpose}).")


def _row_normalize_and_repair(
    G: np.ndarray,
    cost: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    repair: str = "nn_onehot",
    stats: Optional[Dict] = None,
) -> np.ndarray:
    """
    Convert coupling to row-stochastic for barycentric mapping, and repair degenerate rows.

    Degenerate rows: row sum <= eps OR non-finite row sum OR all zeros.
    Repair strategies:
      - "nn_onehot": set the row to one-hot at argmin(cost[i])
      - "uniform": set the row to uniform distribution
    """
    G = np.asarray(G, dtype=np.float64)

    if stats is not None:
        stats["n_rows"] = int(G.shape[0])
        stats["n_cols"] = int(G.shape[1])
        stats["repair_strategy"] = str(repair)

    if not np.isfinite(G).all():
        G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    G[G < 0] = 0.0

    row_sums = G.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums)) | (row_sums <= eps)

    G = np.divide(G, row_sums, out=np.zeros_like(G), where=~bad)

    bad_rows = np.where(bad.reshape(-1))[0]
    n_bad = int(bad_rows.size)
    if stats is not None:
        stats["n_repaired_rows"] = n_bad
        stats["frac_repaired_rows"] = float(n_bad / max(G.shape[0], 1))

    if n_bad > 0:
        if repair == "uniform":
            m = G.shape[1]
            G[bad_rows, :] = 1.0 / max(m, 1)
        else:
            if cost is None:
                m = G.shape[1]
                G[bad_rows, :] = 1.0 / max(m, 1)
            else:
                cost = np.asarray(cost, dtype=np.float64)
                for i in bad_rows:
                    row_cost = cost[i]
                    if not np.isfinite(row_cost).any():
                        G[i, :] = 1.0 / max(G.shape[1], 1)
                        continue
                    j = int(np.nanargmin(row_cost))
                    G[i, :] = 0.0
                    G[i, j] = 1.0

        logger.warning(
            "OT coupling had %d/%d degenerate rows; repaired with '%s'.",
            n_bad, int(G.shape[0]), repair
        )

    row_sums2 = G.sum(axis=1, keepdims=True)
    still_bad = (~np.isfinite(row_sums2)) | (np.abs(row_sums2 - 1.0) > 1e-6)
    if np.any(still_bad):
        row_sums2 = np.clip(row_sums2, eps, None)
        G = G / row_sums2

    return G


def get_ot_coupling_soft_stratified(
    ps_target_np: np.ndarray,
    ps_source_np: np.ndarray,
    cost_np: np.ndarray,
    reg: float,
    eps: float = 1e-12,
    sinkhorn_numItermax: int = 800,
    sinkhorn_stopThr: float = 1e-4,
    purpose: str = "eval",
    repair: str = "nn_onehot",
    stats: Optional[Dict] = None,
) -> np.ndarray:
    ps_t = np.asarray(ps_target_np, dtype=np.float64)
    ps_s = np.asarray(ps_source_np, dtype=np.float64)
    cost = np.asarray(cost_np, dtype=np.float64)

    n_t, n_s = cost.shape
    if stats is not None:
        stats["n_target"] = int(n_t)
        stats["n_source"] = int(n_s)
        stats["ot_method"] = "soft_stratified"
        stats["reg"] = float(reg)
        stats["purpose"] = str(purpose)
        stats["sinkhorn_numItermax"] = int(sinkhorn_numItermax)
        stats["sinkhorn_stopThr"] = float(sinkhorn_stopThr)

    if n_t == 0 or n_s == 0:
        if stats is not None:
            stats["n_repaired_rows"] = 0
            stats["frac_repaired_rows"] = 0.0
        return np.zeros((n_t, n_s), dtype=np.float64)

    n_classes = ps_t.shape[1]
    G = np.zeros((n_t, n_s), dtype=np.float64)

    pi = ps_t.mean(axis=0)
    pi_sum = float(pi.sum())
    if pi_sum <= eps:
        a = np.ones(n_t, dtype=np.float64) / n_t
        b = np.ones(n_s, dtype=np.float64) / n_s
        G0 = _solve_sinkhorn_plan(a, b, cost, reg, sinkhorn_numItermax, sinkhorn_stopThr, purpose=purpose)
        return _row_normalize_and_repair(G0, cost=cost, eps=eps, repair=repair, stats=stats)

    pi = pi / pi_sum

    for c in range(n_classes):
        a = ps_t[:, c].copy()
        b = ps_s[:, c].copy()
        sa, sb = float(a.sum()), float(b.sum())
        if sa <= eps or sb <= eps or pi[c] <= eps:
            continue
        a /= sa
        b /= sb

        try:
            Gc = _solve_sinkhorn_plan(a, b, cost, reg, sinkhorn_numItermax, sinkhorn_stopThr, purpose=purpose)
            G += pi[c] * Gc
        except Exception as e:
            logger.warning("soft_stratified: sinkhorn failed for class %d (skip). err=%s", c, e)

    return _row_normalize_and_repair(G, cost=cost, eps=eps, repair=repair, stats=stats)


def get_ot_coupling(
    ps_target: torch.Tensor,
    ps_source: torch.Tensor,
    reg: float = 1e-3,
    method: str = "standard",
    sinkhorn_numItermax: Optional[int] = None,
    sinkhorn_stopThr: Optional[float] = None,
    purpose: str = "eval",  # "train" or "eval"
    repair: str = "nn_onehot",
    return_stats: bool = False,
    cost_metric: str = "sqeuclidean",
):
    """
    Return a row-stochastic coupling for barycentric mapping.

    cost_metric:
      - "sqeuclidean" (default): Euclidean on probabilities
      - "hellinger": Euclidean on sqrt(probabilities)
    """
    ps_t = ps_target.detach().cpu().numpy()
    ps_s = ps_source.detach().cpu().numpy()

    if ps_t.ndim != 2 or ps_s.ndim != 2:
        raise ValueError(f"ps_target/ps_source must be 2D, got {ps_t.shape} and {ps_s.shape}")

    n_t, n_s = ps_t.shape[0], ps_s.shape[0]
    if n_t == 0 or n_s == 0:
        G0 = torch.zeros((n_t, n_s), dtype=torch.float32)
        if return_stats:
            stats = {
                "n_target": int(n_t),
                "n_source": int(n_s),
                "reg": float(reg),
                "ot_method": (method or "standard").lower().strip(),
                "purpose": str(purpose),
                "sinkhorn_numItermax": int(sinkhorn_numItermax or 0),
                "sinkhorn_stopThr": float(sinkhorn_stopThr or 0.0),
                "n_repaired_rows": 0,
                "frac_repaired_rows": 0.0,
                "repair_strategy": str(repair),
                "cost_metric": str(cost_metric),
            }
            return G0, stats
        return G0

    p = (purpose or "eval").lower().strip()
    if sinkhorn_numItermax is None:
        sinkhorn_numItermax = 600 if p == "train" else 3000
    if sinkhorn_stopThr is None:
        sinkhorn_stopThr = 1e-4 if p == "train" else 1e-6

    m = (method or "standard").lower().strip()
    stats: Dict = {
        "n_target": int(n_t),
        "n_source": int(n_s),
        "reg": float(reg),
        "ot_method": m,
        "purpose": str(purpose),
        "sinkhorn_numItermax": int(sinkhorn_numItermax),
        "sinkhorn_stopThr": float(sinkhorn_stopThr),
        "repair_strategy": str(repair),
        "cost_metric": str(cost_metric),
    }

    cost = _compute_ps_cost(ps_t, ps_s, metric=cost_metric)

    if m == "soft_stratified":
        G_np = get_ot_coupling_soft_stratified(
            ps_t, ps_s, cost, reg=reg,
            sinkhorn_numItermax=sinkhorn_numItermax,
            sinkhorn_stopThr=sinkhorn_stopThr,
            purpose=purpose,
            repair=repair,
            stats=stats,
        )
    else:
        a = np.ones(n_t, dtype=np.float64) / n_t
        b = np.ones(n_s, dtype=np.float64) / n_s
        G0 = _solve_sinkhorn_plan(a, b, cost, reg, sinkhorn_numItermax, sinkhorn_stopThr, purpose=purpose)
        G_np = _row_normalize_and_repair(G0, cost=cost, eps=1e-12, repair=repair, stats=stats)

    G = torch.from_numpy(G_np).float()
    if return_stats:
        stats.setdefault("n_repaired_rows", 0)
        stats.setdefault("frac_repaired_rows", 0.0)
        return G, stats
    return G


def get_ot_coupling_modality_specific(
    ps_target: torch.Tensor,
    ps_source: torch.Tensor,
    modality: str,
    reg_mass: float = 5e-4,
    reg_rna: float = 5e-3,
    reg_default: float = 1e-3,
    method_mass: str = "standard",
    method_rna: str = "standard",
    method_default: str = "standard",
    purpose: str = "eval",
    repair: str = "nn_onehot",
    return_stats: bool = False,
    cost_metric_mass: str = "sqeuclidean",
    cost_metric_rna: str = "sqeuclidean",
    cost_metric_default: str = "sqeuclidean",
):
    m = (modality or "").lower().strip()
    if m == "mass":
        reg = reg_mass
        method = method_mass
        cost_metric = cost_metric_mass
    elif m == "rna":
        reg = reg_rna
        method = method_rna
        cost_metric = cost_metric_rna
    else:
        reg = reg_default
        method = method_default
        cost_metric = cost_metric_default
        logger.warning("Unknown modality '%s', using default settings.", modality)

    return get_ot_coupling(
        ps_target, ps_source,
        reg=reg, method=method,
        purpose=purpose,
        repair=repair,
        return_stats=return_stats,
        cost_metric=cost_metric,
    )


# =============================================================================
# Pseudo-sample export
# =============================================================================
def save_pseudo_samples(
    gly_ids: np.ndarray,
    gly_labels_encoded: np.ndarray,
    gly_features: np.ndarray,
    pseudo_mass_features: np.ndarray,
    pseudo_rna_features: np.ndarray,
    reverse_label_map: Dict,
    file_path: Path,
):
    """Save anchor samples and their pseudo-paired donor features."""
    try:
        gly_labels_str = pd.Series(gly_labels_encoded).map(reverse_label_map).values

        df_ids_labels = pd.DataFrame({"ID": gly_ids, "Group": gly_labels_str})
        df_gly = pd.DataFrame(gly_features, columns=[f"gly_feat_{i}" for i in range(gly_features.shape[1])])
        df_mass = pd.DataFrame(pseudo_mass_features, columns=[f"pseudo_mass_feat_{i}" for i in range(pseudo_mass_features.shape[1])])
        df_rna = pd.DataFrame(pseudo_rna_features, columns=[f"pseudo_rna_feat_{i}" for i in range(pseudo_rna_features.shape[1])])

        final_df = pd.concat([df_ids_labels, df_gly, df_mass, df_rna], axis=1)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved pseudo-samples to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save pseudo-samples to {file_path}: {e}", exc_info=True)
