# interpretability.py

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import spearmanr

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

import ot

from slemodel import config as cfg
from slemodel.models import DecoupledModel, PropensityScoreClassifier
from slemodel.utils import (
    get_ot_coupling,
    train_classifier, get_propensity_scores
)


BASE_DIR = Path(cfg.BASE_EXPERIMENT_DIR) / cfg.EXPERIMENT_TAG
OUT_DIR = Path("interpretability_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_ORDER = ['gly', 'mass', 'rna']
MODALITY_DISPLAY = {'gly': 'Glycan', 'mass': 'Mass', 'rna': 'RNA'}

warnings.filterwarnings("ignore", category=FutureWarning)

def _discover_seeds() -> list[int]:
    """Discover seeds from experiment directories, with configured defaults as fallback."""
    if not BASE_DIR.exists():
        return [getattr(cfg, "SEED", 42)]
    seeds = []
    for p in BASE_DIR.glob("SLEmodel_Run_Seed_*"):
        m = re.search(r"SLEmodel_Run_Seed_(\d+)$", p.name)
        if m:
            seeds.append(int(m.group(1)))
    if seeds:
        return sorted(seeds)
    return [getattr(cfg, "SEED", 42), 100, 2025, 7, 123]

SEEDS = _discover_seeds()

def set_publication_style():
    plt.rcParams.update({
        'figure.figsize': (6.4, 4.8), 'figure.dpi': 150,
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'axes.grid': False,
        'savefig.bbox': 'tight'
    })
    sns.set_style("ticks")

set_publication_style()

def _savefig(fig, path_no_ext: Path):
    fig.savefig(str(path_no_ext.with_suffix(".png")))
    plt.close(fig)


def _torch_cos_sim_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1)

def _perm_pvalue_greater(obs: float, null_vals: np.ndarray) -> float:
    m = null_vals.size
    return float((np.sum(null_vals >= obs) + 1) / (m + 1))

def _pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        d = torch.cdist(x, y, p=2)
    return d.cpu().numpy()

def _get_ps_cost_metric(modality: str | None) -> str:
    m = (modality or "").lower().strip()
    if m == "mass":
        return getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean")
    if m == "rna":
        return getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean")
    return getattr(cfg, "OT_COST_METRIC_DEFAULT", "sqeuclidean")


def _ps_space_transform(ps: np.ndarray, metric: str) -> np.ndarray:
    """
    Map PS vectors to the feature space where we compute the OT cost.
    - sqeuclidean: identity
    - hellinger: sqrt(ps)
    """
    m = (metric or "sqeuclidean").lower().strip()
    if m == "hellinger":
        return np.sqrt(np.clip(ps, 0.0, 1.0))
    return ps

def _transport_cost_stats(G: np.ndarray, cost_matrix: np.ndarray) -> dict:
    """
    Compute OT transport-cost statistics for a row-stochastic coupling G used for barycentric mapping.

    Definitions (consistent with row-stochastic G):
      row_cost[i] = sum_j G[i, j] * C[i, j]    (expected cost for target i)
      mean_transport_cost   = mean_i row_cost[i]
      median_transport_cost = median_i row_cost[i]
      total_transport_cost  = sum_i row_cost[i]  (useful sanity check; scales with n_target)

    Returns:
      dict with keys: mean, median, total, row_cost (np.ndarray)
    """
    G = np.asarray(G, dtype=np.float64)
    C = np.asarray(cost_matrix, dtype=np.float64)

    if G.ndim != 2 or C.ndim != 2:
        raise ValueError(f"G and cost_matrix must be 2D, got {G.shape} and {C.shape}")
    if G.shape != C.shape:
        raise ValueError(f"Shape mismatch: G {G.shape} vs cost_matrix {C.shape}")

    # row-wise expected cost
    row_cost = np.sum(G * C, axis=1)

    # Reject non-finite values before distance calculation.
    row_cost = np.where(np.isfinite(row_cost), row_cost, np.nan)

    if row_cost.size == 0:
        mean_val = np.nan
        median_val = np.nan
        total_val = np.nan
    else:
        mean_val = float(np.nanmean(row_cost))
        median_val = float(np.nanmedian(row_cost))
        total_val = float(np.nansum(row_cost))

    return {
        "mean": mean_val,
        "median": median_val,
        "total": total_val,
        "row_cost": row_cost,
    }


def _ps_cost_matrix(ps_t: np.ndarray, ps_s: np.ndarray, metric: str) -> np.ndarray:
    """
    Cost matrix consistent with utils.get_ot_coupling(cost_metric=...).
    Squared Euclidean distance is used in the transformed space.
    """
    xt = _ps_space_transform(np.asarray(ps_t, dtype=np.float64), metric)
    xs = _ps_space_transform(np.asarray(ps_s, dtype=np.float64), metric)
    return ot.dist(xt, xs, metric="sqeuclidean").astype(np.float64)


def _row_entropy(G: np.ndarray) -> np.ndarray:
    return -np.sum(G * np.log(G + 1e-12), axis=1)

def _row_topk_mass(G: np.ndarray, k: int) -> np.ndarray:
    if k >= G.shape[1]:
        return np.ones(G.shape[0])
    part = np.partition(G, -k, axis=1)[:, -k:]
    return part.sum(axis=1)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _auc_ap(y_true: torch.Tensor, scores: torch.Tensor) -> Tuple[float, float]:
    y, s = y_true.cpu().numpy(), scores.cpu().numpy()
    try:
        auc = roc_auc_score(y, s, average='macro', multi_class='ovr')
    except Exception:
        auc = np.nan
    try:
        ap = np.mean([average_precision_score((y == k).astype(int), s[:, k]) for k in range(s.shape[1]) if np.any(y == k)])
    except Exception:
        ap = np.nan
    return float(auc), float(ap)

def _hsic_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    def _gram(a):
        pdist = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
        s = np.median(pdist[pdist > 0]) + 1e-12
        d2 = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * s ** 2))
    Kx, Ky = _gram(x), _gram(y)
    n = x.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Lc = H @ Kx @ H, H @ Ky @ H
    return float(np.trace(Kc @ Lc) / ((n - 1) ** 2))

def _load_csv(path: str):
    df = pd.read_csv(path)
    feats = df.iloc[:, 2:].values
    cols = df.columns[2:].tolist()
    return df.iloc[:, 1].values, feats, cols

def _get_global_fold_index(fold_name: str, n_splits: int) -> int:
    rep_idx = int(fold_name.split("Repeat_")[1].split("-")[0]) - 1
    fold_local = int(fold_name.split("Fold_")[1]) - 1
    return rep_idx * n_splits + fold_local

def _make_coupling(
    ps_val_gly: torch.Tensor,
    ps_pool_other: torch.Tensor,
    modality: str = None,
    reg: float = 1e-3,
    return_stats: bool = False,
    purpose: str = "eval",
):
    """Recompute a validation transport plan with the training configuration."""
    method_mass = getattr(cfg, "OT_METHOD_MASS", "standard")
    method_rna = getattr(cfg, "OT_METHOD_RNA", "standard")

    cost_metric_mass = getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean")
    cost_metric_rna = getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean")
    cost_metric_default = getattr(cfg, "OT_COST_METRIC_DEFAULT", "sqeuclidean")

    if modality is not None:
        m = modality.lower().strip()
        if m == "mass":
            reg = getattr(cfg, "OT_REG_MASS", reg)
            method = method_mass
            cost_metric = cost_metric_mass
        elif m == "rna":
            reg = getattr(cfg, "OT_REG_RNA", reg)
            method = method_rna
            cost_metric = cost_metric_rna
        else:
            method = "standard"
            cost_metric = cost_metric_default
    else:
        method = "standard"
        cost_metric = cost_metric_default

    return get_ot_coupling(
        ps_val_gly,
        ps_pool_other,
        reg=reg,
        method=method,
        purpose=purpose,
        return_stats=return_stats,
        cost_metric=cost_metric,
    )


def _knn_overlap(C: np.ndarray, D: np.ndarray, k: int) -> float:
    """Return k nearest indices using the zero-based argpartition boundary."""
    idx_c = np.argpartition(C, k-1, axis=1)[:, :k]
    idx_d = np.argpartition(D, k-1, axis=1)[:, :k]
    return float(np.mean([len(set(r1) & set(r2)) / k for r1, r2 in zip(idx_c, idx_d)]))

def compute_detailed_smd(ps_target: np.ndarray, ps_source: np.ndarray,
                         ps_matched: np.ndarray,
                         labels_target: np.ndarray,
                         n_classes: int = 3) -> dict:
    """
    Calculate covariate-level absolute SMD values before and after matching.
    - max |SMD|
    - mean |SMD|
    - % covariates < 0.1
    - % covariates < 0.05

    Args:
        ps_target: Propensity scores for the glycomics anchor cohort.
        ps_source: Propensity scores for the Mass or RNA donor cohort.
        ps_matched: Propensity scores after barycentric matching.
        labels_target: Class labels for the anchor cohort.
        n_classes: Number of classes.

    Returns:
        Dictionary containing class-specific SMD diagnostics.
    """
    label_names = cfg.VisualizationConfig.LABEL_NAMES

    def _smd_per_covariate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate absolute SMD for each propensity-score coordinate."""
        mu_a = a.mean(axis=0)
        mu_b = b.mean(axis=0)
        var_a = np.var(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mu_a)
        var_b = np.var(b, axis=0, ddof=1) if b.shape[0] > 1 else np.zeros_like(mu_b)
        sd_pool = np.sqrt((var_a + var_b) / 2)
        return np.abs((mu_a - mu_b) / (sd_pool + 1e-9))

    results = {
        'overall': {},
        'by_group': {},
        'per_covariate': {}
    }

    smd_pre_all = _smd_per_covariate(ps_target, ps_source)
    smd_post_all = _smd_per_covariate(ps_target, ps_matched)

    results['overall'] = {
        'pre_smd_mean': float(np.mean(smd_pre_all)),
        'pre_smd_max': float(np.max(smd_pre_all)),
        'pre_pct_below_0.1': float(np.mean(smd_pre_all < 0.1) * 100),
        'pre_pct_below_0.05': float(np.mean(smd_pre_all < 0.05) * 100),
        'post_smd_mean': float(np.mean(smd_post_all)),
        'post_smd_max': float(np.max(smd_post_all)),
        'post_pct_below_0.1': float(np.mean(smd_post_all < 0.1) * 100),
        'post_pct_below_0.05': float(np.mean(smd_post_all < 0.05) * 100),
        'delta_smd_mean': float(np.mean(smd_post_all) - np.mean(smd_pre_all)),
        'n_covariates': len(smd_pre_all),
    }

    results['per_covariate']['pre'] = smd_pre_all.tolist()
    results['per_covariate']['post'] = smd_post_all.tolist()

    source_mean = ps_source.mean(axis=0)
    source_var = np.var(ps_source, axis=0, ddof=1) if ps_source.shape[0] > 1 else np.zeros(ps_source.shape[1])

    for k in range(n_classes):
        mask = labels_target == k
        if mask.sum() == 0:
            continue

        group_name = label_names.get(k, f"Class_{k}")
        target_group = ps_target[mask]
        matched_group = ps_matched[mask]

        target_group_mean = target_group.mean(axis=0)
        target_group_var = np.var(target_group, axis=0, ddof=1) if target_group.shape[0] > 1 else np.zeros_like(target_group_mean)
        pooled_sd_pre = np.sqrt((target_group_var + source_var) / 2)
        smd_pre_group = np.abs((target_group_mean - source_mean) / (pooled_sd_pre + 1e-9))

        smd_post_group = _smd_per_covariate(target_group, matched_group)

        results['by_group'][group_name] = {
            'pre_smd_mean': float(np.mean(smd_pre_group)),
            'pre_smd_max': float(np.max(smd_pre_group)),
            'pre_pct_below_0.1': float(np.mean(smd_pre_group < 0.1) * 100),
            'pre_pct_below_0.05': float(np.mean(smd_pre_group < 0.05) * 100),
            'post_smd_mean': float(np.mean(smd_post_group)),
            'post_smd_max': float(np.max(smd_post_group)),
            'post_pct_below_0.1': float(np.mean(smd_post_group < 0.1) * 100),
            'post_pct_below_0.05': float(np.mean(smd_post_group < 0.05) * 100),
            'delta_smd_mean': float(np.mean(smd_post_group) - np.mean(smd_pre_group)),
            'n_samples': int(mask.sum()),
            'per_covariate_pre': smd_pre_group.tolist(),
            'per_covariate_post': smd_post_group.tolist(),
        }

    return results


def build_val_inputs_for_fold(seed: int, fold_name: str,
                              gly_feats: np.ndarray, gly_labels_enc: np.ndarray) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    exp_path = BASE_DIR / f"SLEmodel_Run_Seed_{seed}" / f"master_seed_{seed}"
    recon_file = exp_path / "reconstruction_data" / f"{fold_name}_validation.npz"
    if not recon_file.exists():
        return None
    rskf_gly = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)
    gly_splits = list(rskf_gly.split(gly_feats, gly_labels_enc))
    global_idx = _get_global_fold_index(fold_name, cfg.N_SPLITS_K_FOLD)
    _, gly_val_idx = gly_splits[global_idx]
    with np.load(recon_file) as data:
        for m in MODALITY_ORDER:
            if f'{m}_true' not in data:
                return None
        x_tensors = {m: torch.tensor(data[f'{m}_true'], dtype=torch.float32) for m in MODALITY_ORDER}
    y_val = torch.tensor(gly_labels_enc[gly_val_idx], dtype=torch.long)
    return x_tensors, y_val

def get_model_for_seed_fold(seed: int, fold_name: str, input_dims: Tuple[int, int, int]) -> DecoupledModel:
    ckpt_path = BASE_DIR / f"SLEmodel_Run_Seed_{seed}" / f"master_seed_{seed}" / fold_name / f"best_model_{fold_name}.pth"
    model = DecoupledModel(*input_dims).to(cfg.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
    model.eval()
    return model

def _load_saved_ps_ot_if_any(seed: int, fold_name: str):
    """Load saved validation propensity scores and couplings when available."""
    f = BASE_DIR / f"SLEmodel_Run_Seed_{seed}" / f"master_seed_{seed}" / fold_name / "val_ps_ot.npz"
    if f.exists():
        d = np.load(f)
        return {k: d[k] for k in d.files}
    return None

def ps_ot_qc_for_fold(seed: int, fold_name: str,
                      gly_feats: np.ndarray, gly_labels_enc: np.ndarray,
                      mass_feats: np.ndarray, mass_labels_enc: np.ndarray,
                      rna_feats: np.ndarray, rna_labels_enc: np.ndarray):
    """
    PS/OT quality control for one (seed, fold).

    For the PS-balance diagnostic:
      - SMD is computed in the *same feature space* as the OT cost metric.
        For the Hellinger cost, SMD is calculated in square-root propensity-score
        space; otherwise it is calculated in the original propensity-score space.

    This makes the Mass SMD (Hellinger OT) directly comparable to the notion
    of distance used in OT, and resolves the previous mismatch:
      OT cost in sqrt-space vs SMD in raw probability-space.
    """
    out_dir = OUT_DIR / f"seed_{seed}" / "ps_ot_qc" / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = _load_saved_ps_ot_if_any(seed, fold_name)

    rskf_gly = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)
    rskf_mass = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)
    rskf_rna  = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)

    global_idx = _get_global_fold_index(fold_name, cfg.N_SPLITS_K_FOLD)
    gly_train_idx, gly_val_idx = list(rskf_gly.split(gly_feats, gly_labels_enc))[global_idx]
    mass_train_idx, _ = list(rskf_mass.split(mass_feats, mass_labels_enc))[global_idx]
    rna_train_idx, _  = list(rskf_rna.split(rna_feats, rna_labels_enc))[global_idx]

    y_gly_val = gly_labels_enc[gly_val_idx]
    y_mass_tr = mass_labels_enc[mass_train_idx]
    y_rna_tr  = rna_labels_enc[rna_train_idx]

    used_saved_G = False
    max_abs_diff_mass = np.nan
    max_abs_diff_rna = np.nan

    if saved is not None:
        ps_g_va = saved['ps_val_gly']
        ps_m_tr = saved['ps_pool_mass']
        ps_r_tr = saved['ps_pool_rna']
        Gm = saved['G_mass']
        Gr = saved['G_rna']
        used_saved_G = True
    else:
        gly_scaler = StandardScaler().fit(gly_feats[gly_train_idx])
        mass_scaler = StandardScaler().fit(mass_feats[mass_train_idx])
        rna_scaler  = StandardScaler().fit(rna_feats[rna_train_idx])

        Xg_tr = torch.tensor(gly_scaler.transform(gly_feats[gly_train_idx]), dtype=torch.float32)
        Xg_va = torch.tensor(gly_scaler.transform(gly_feats[gly_val_idx]), dtype=torch.float32)
        Xm_tr = torch.tensor(mass_scaler.transform(mass_feats[mass_train_idx]), dtype=torch.float32)
        Xr_tr = torch.tensor(rna_scaler.transform(rna_feats[rna_train_idx]), dtype=torch.float32)

        yg_tr = torch.tensor(gly_labels_enc[gly_train_idx], dtype=torch.long)
        ym_tr = torch.tensor(mass_labels_enc[mass_train_idx], dtype=torch.long)
        yr_tr = torch.tensor(rna_labels_enc[rna_train_idx], dtype=torch.long)

        g_clf = train_classifier(PropensityScoreClassifier(Xg_tr.shape[1]), Xg_tr, yg_tr, cfg.DEVICE, epochs=cfg.EPOCHS_PS)
        m_clf = train_classifier(PropensityScoreClassifier(Xm_tr.shape[1]), Xm_tr, ym_tr, cfg.DEVICE, epochs=cfg.EPOCHS_PS)
        r_clf = train_classifier(PropensityScoreClassifier(Xr_tr.shape[1]), Xr_tr, yr_tr, cfg.DEVICE, epochs=cfg.EPOCHS_PS)

        ps_g_va = get_propensity_scores(g_clf, Xg_va, cfg.DEVICE).numpy()
        ps_m_tr = get_propensity_scores(m_clf, Xm_tr, cfg.DEVICE).numpy()
        ps_r_tr = get_propensity_scores(r_clf, Xr_tr, cfg.DEVICE).numpy()

        Gm_t = _make_coupling(torch.from_numpy(ps_g_va), torch.from_numpy(ps_m_tr), modality='mass', return_stats=False, purpose="eval")
        Gr_t = _make_coupling(torch.from_numpy(ps_g_va), torch.from_numpy(ps_r_tr), modality='rna',  return_stats=False, purpose="eval")
        Gm = Gm_t.numpy()
        Gr = Gr_t.numpy()

    # --- recompute once to get stats (and optionally diff to saved) ---
    Gm_re, stats_m = _make_coupling(
        torch.from_numpy(ps_g_va).float(),
        torch.from_numpy(ps_m_tr).float(),
        modality='mass',
        return_stats=True,
        purpose="eval",
    )
    Gr_re, stats_r = _make_coupling(
        torch.from_numpy(ps_g_va).float(),
        torch.from_numpy(ps_r_tr).float(),
        modality='rna',
        return_stats=True,
        purpose="eval",
    )

    if used_saved_G:
        try:
            max_abs_diff_mass = float(np.max(np.abs(Gm_re.numpy() - Gm)))
        except Exception:
            max_abs_diff_mass = np.nan
        try:
            max_abs_diff_rna = float(np.max(np.abs(Gr_re.numpy() - Gr)))
        except Exception:
            max_abs_diff_rna = np.nan

    repair_rows = []
    for modality_name, stats in [("Mass", stats_m), ("RNA", stats_r)]:
        repair_rows.append({
            "seed": seed,
            "fold": fold_name,
            "global_fold_index": global_idx,
            "modality": modality_name,
            "used_saved_G": bool(used_saved_G),
            "max_abs_diff_to_saved_G": max_abs_diff_mass if modality_name == "Mass" else max_abs_diff_rna,
            "n_target": int(stats.get("n_target", -1)),
            "n_source": int(stats.get("n_source", -1)),
            "reg": float(stats.get("reg", np.nan)),
            "ot_method": str(stats.get("ot_method", "")),
            "purpose": str(stats.get("purpose", "")),
            "sinkhorn_numItermax": int(stats.get("sinkhorn_numItermax", -1)),
            "sinkhorn_stopThr": float(stats.get("sinkhorn_stopThr", np.nan)),
            "repair_strategy": str(stats.get("repair_strategy", "")),
            "n_repaired_rows": int(stats.get("n_repaired_rows", 0)),
            "frac_repaired_rows": float(stats.get("frac_repaired_rows", 0.0)),
        })
    pd.DataFrame(repair_rows).to_csv(out_dir / "ot_repair_stats.csv", index=False)

    # -------------------- SMD (PS-aligned feature space) --------------------
    def _smd_signed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Signed SMD per covariate:
          (mean(a)-mean(b)) / pooled_sd
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
        var_a = np.var(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mu_a)
        var_b = np.var(b, axis=0, ddof=1) if b.shape[0] > 1 else np.zeros_like(mu_b)
        sd_pool = np.sqrt((var_a + var_b) / 2.0)
        return (mu_a - mu_b) / (sd_pool + 1e-9)

    # Use the same metric used by OT cost
    metric_mass = _get_ps_cost_metric("mass")  # e.g., "hellinger"
    metric_rna = _get_ps_cost_metric("rna")    # e.g., "sqeuclidean"

    # Transform PS into the cost-feature space (identity or sqrt)
    ps_g_va_mass = _ps_space_transform(ps_g_va, metric_mass)
    ps_m_tr_mass = _ps_space_transform(ps_m_tr, metric_mass)
    ps_r_tr_rna  = _ps_space_transform(ps_r_tr, metric_rna)
    ps_g_va_rna  = _ps_space_transform(ps_g_va, metric_rna)

    # Barycentric matched "PS-features" in the same space
    ps_m_matched_mass = Gm @ ps_m_tr_mass
    ps_r_matched_rna  = Gr @ ps_r_tr_rna

    smd_m_pre  = _smd_signed(ps_g_va_mass, ps_m_tr_mass)
    smd_m_post = _smd_signed(ps_g_va_mass, ps_m_matched_mass)

    smd_r_pre  = _smd_signed(ps_g_va_rna, ps_r_tr_rna)
    smd_r_post = _smd_signed(ps_g_va_rna, ps_r_matched_rna)

    records = []
    for i in range(len(smd_m_pre)):
        group_name = cfg.VisualizationConfig.LABEL_NAMES.get(i, f"PS_{i}")
        records.append({
            'modality': 'Mass',
            'group': group_name,
            'pre_match_smd': float(abs(smd_m_pre[i])),
            'post_match_smd': float(abs(smd_m_post[i])),
            'delta_smd': float(abs(smd_m_post[i]) - abs(smd_m_pre[i])),
            'smd_space': f"ps_feature_space({metric_mass})",
        })
    for i in range(len(smd_r_pre)):
        group_name = cfg.VisualizationConfig.LABEL_NAMES.get(i, f"PS_{i}")
        records.append({
            'modality': 'RNA',
            'group': group_name,
            'pre_match_smd': float(abs(smd_r_pre[i])),
            'post_match_smd': float(abs(smd_r_post[i])),
            'delta_smd': float(abs(smd_r_post[i]) - abs(smd_r_pre[i])),
            'smd_space': f"ps_feature_space({metric_rna})",
        })
    pd.DataFrame(records).to_csv(out_dir / "smd_summary.csv", index=False)

    # -------------------- Global distances (unchanged) --------------------
    def _compute_emd(a, b, metric: str):
        n, m = len(a), len(b)
        cost = _ps_cost_matrix(a, b, metric)
        return float(ot.emd2(np.ones(n)/n, np.ones(m)/m, cost))

    def _compute_mmd2(a, b, metric: str, gamma=1.0):
        xa = _ps_space_transform(np.asarray(a, dtype=np.float64), metric)
        xb = _ps_space_transform(np.asarray(b, dtype=np.float64), metric)

        def rbf_kernel(x, y, gamma):
            dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
            return np.exp(-gamma * dist)

        Kxx = rbf_kernel(xa, xa, gamma)
        Kyy = rbf_kernel(xb, xb, gamma)
        Kxy = rbf_kernel(xa, xb, gamma)
        return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())

    def _compute_sinkhorn_div(a, b, metric: str, reg=0.1):
        n, m = len(a), len(b)
        xa = _ps_space_transform(np.asarray(a, dtype=np.float64), metric)
        xb = _ps_space_transform(np.asarray(b, dtype=np.float64), metric)

        cost_ab = ot.dist(xa, xb, metric="sqeuclidean")
        cost_aa = ot.dist(xa, xa, metric="sqeuclidean")
        cost_bb = ot.dist(xb, xb, metric="sqeuclidean")

        sink_ab = ot.sinkhorn2(np.ones(n)/n, np.ones(m)/m, cost_ab, reg)
        sink_aa = ot.sinkhorn2(np.ones(n)/n, np.ones(n)/n, cost_aa, reg)
        sink_bb = ot.sinkhorn2(np.ones(m)/m, np.ones(m)/m, cost_bb, reg)
        return float(sink_ab - 0.5 * sink_aa - 0.5 * sink_bb)

    global_metrics = []
    for mod_name, ps_pool, ps_matched, G, metric in [
        ('Mass', ps_m_tr, (Gm @ ps_m_tr), Gm, metric_mass),
        ('RNA',  ps_r_tr, (Gr @ ps_r_tr), Gr, metric_rna),
    ]:
        emd_pre  = _compute_emd(ps_g_va, ps_pool, metric)
        mmd2_pre = _compute_mmd2(ps_g_va, ps_pool, metric)
        sink_pre = _compute_sinkhorn_div(ps_g_va, ps_pool, metric)

        emd_post  = _compute_emd(ps_g_va, ps_matched, metric)
        mmd2_post = _compute_mmd2(ps_g_va, ps_matched, metric)
        sink_post = _compute_sinkhorn_div(ps_g_va, ps_matched, metric)

        cost_matrix = _ps_cost_matrix(ps_g_va, ps_pool, metric)
        tc = _transport_cost_stats(G, cost_matrix)

        global_metrics.append({
            'modality': mod_name,
            'cost_metric': metric,
            'emd_pre': emd_pre, 'emd_post': emd_post,
            'mmd2_pre': mmd2_pre, 'mmd2_post': mmd2_post,
            'sink_pre': sink_pre, 'sink_post': sink_post,
            'mean_transport_cost': tc["mean"],
            'median_transport_cost': tc["median"],
            'total_transport_cost': tc["total"],
        })
    pd.DataFrame(global_metrics).to_csv(out_dir / "global_alignment_metrics.csv", index=False)

    # -------------------- Class Flow Matrix (unchanged) --------------------
    def _compute_class_flow_vectorized(G, y_target, y_source, n_classes):
        Yt = np.eye(n_classes, dtype=float)[y_target]
        Ys = np.eye(n_classes, dtype=float)[y_source]
        flow = Yt.T @ G @ Ys
        flow = flow / (flow.sum(axis=1, keepdims=True) + 1e-9)
        return flow

    n_classes = cfg.NUM_CLASSES
    label_names = [cfg.VisualizationConfig.LABEL_NAMES[i] for i in range(n_classes)]
    flow_mass = _compute_class_flow_vectorized(Gm, y_gly_val, y_mass_tr, n_classes)
    pd.DataFrame(flow_mass, index=label_names, columns=label_names).to_csv(out_dir / "class_flow_matrix_mass.csv")
    flow_rna = _compute_class_flow_vectorized(Gr, y_gly_val, y_rna_tr, n_classes)
    pd.DataFrame(flow_rna, index=label_names, columns=label_names).to_csv(out_dir / "class_flow_matrix_rna.csv")

    # Negative controls use SMD in the same propensity-score space.
    rng = np.random.default_rng(seed * 10_000 + global_idx)

    Gm_shuf = Gm[:, rng.permutation(Gm.shape[1])]
    Gr_shuf = Gr[:, rng.permutation(Gr.shape[1])]

    def _random_row_stochastic(n, m, rng_):
        return rng_.dirichlet(np.ones(m), size=n)

    Gm_rand = _random_row_stochastic(Gm.shape[0], Gm.shape[1], rng)
    Gr_rand = _random_row_stochastic(Gr.shape[0], Gr.shape[1], rng)

    # compute SMD in the same feature space used above
    smd_m_post_shuf = _smd_signed(ps_g_va_mass, Gm_shuf @ ps_m_tr_mass)
    smd_m_post_rand = _smd_signed(ps_g_va_mass, Gm_rand @ ps_m_tr_mass)

    smd_r_post_shuf = _smd_signed(ps_g_va_rna,  Gr_shuf @ ps_r_tr_rna)
    smd_r_post_rand = _smd_signed(ps_g_va_rna,  Gr_rand @ ps_r_tr_rna)

    neg_rows = []
    for i in range(len(smd_m_pre)):
        group_name = cfg.VisualizationConfig.LABEL_NAMES.get(i, f"PS_{i}")
        neg_rows.append({
            'modality': 'Mass', 'group': group_name, 'type': 'shuffled',
            'pre_match_smd': float(abs(smd_m_pre[i])),
            'post_match_smd': float(abs(smd_m_post_shuf[i])),
            'delta_smd': float(abs(smd_m_post_shuf[i]) - abs(smd_m_pre[i])),
            'smd_space': f"ps_feature_space({metric_mass})",
        })
        neg_rows.append({
            'modality': 'Mass', 'group': group_name, 'type': 'random',
            'pre_match_smd': float(abs(smd_m_pre[i])),
            'post_match_smd': float(abs(smd_m_post_rand[i])),
            'delta_smd': float(abs(smd_m_post_rand[i]) - abs(smd_m_pre[i])),
            'smd_space': f"ps_feature_space({metric_mass})",
        })
    for i in range(len(smd_r_pre)):
        group_name = cfg.VisualizationConfig.LABEL_NAMES.get(i, f"PS_{i}")
        neg_rows.append({
            'modality': 'RNA', 'group': group_name, 'type': 'shuffled',
            'pre_match_smd': float(abs(smd_r_pre[i])),
            'post_match_smd': float(abs(smd_r_post_shuf[i])),
            'delta_smd': float(abs(smd_r_post_shuf[i]) - abs(smd_r_pre[i])),
            'smd_space': f"ps_feature_space({metric_rna})",
        })
        neg_rows.append({
            'modality': 'RNA', 'group': group_name, 'type': 'random',
            'pre_match_smd': float(abs(smd_r_pre[i])),
            'post_match_smd': float(abs(smd_r_post_rand[i])),
            'delta_smd': float(abs(smd_r_post_rand[i]) - abs(smd_r_pre[i])),
            'smd_space': f"ps_feature_space({metric_rna})",
        })
    pd.DataFrame(neg_rows).to_csv(out_dir / "smd_summary_negcontrol.csv", index=False)

    # -------------------- Locality (unchanged) --------------------
    def _save_locality(G: np.ndarray, name: str):
        hhi = np.sum(G ** 2, axis=1)
        effective_donors = 1.0 / (hhi + 1e-12)
        pd.DataFrame({
            'row_entropy': _row_entropy(G),
            'hhi': hhi,
            'effective_donors': effective_donors,
            **{f'top{k}': _row_topk_mass(G, k) for k in (1, 3, 5, 10)}
        }).to_csv(out_dir / f"locality_{name}.csv", index=False)

    _save_locality(Gm, 'mass')
    _save_locality(Gr, 'rna')
    _save_locality(Gm_rand, 'mass_negcontrol_random')
    _save_locality(Gr_rand, 'rna_negcontrol_random')
    _save_locality(Gm_shuf, 'mass_negcontrol_shuffled')
    _save_locality(Gr_shuf, 'rna_negcontrol_shuffled')

    # Keep original PS distributions for KDE plots (raw PS space)
    np.savez(
        out_dir / "ps_distributions.npz",
        ps_gly_val=ps_g_va,
        ps_mass_pool=ps_m_tr,
        ps_rna_pool=ps_r_tr,
        ps_mass_matched=(Gm @ ps_m_tr),
        ps_rna_matched=(Gr @ ps_r_tr)
    )

    return (ps_g_va, ps_m_tr, ps_r_tr, Gm, Gr, Gm_rand, Gr_rand, Gm_shuf, Gr_shuf)


def within_pair_similarity_for_fold(seed: int, fold_name: str, model: DecoupledModel,
                                    x_t: Dict[str, torch.Tensor], out_root: Path, n_perm: int = 1000):
    out_dir = out_root / f"seed_{seed}" / "pair_consistency" / fold_name
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    device = cfg.DEVICE
    with torch.no_grad():
        zg_s, _ = model.gly_encoder(x_t['gly'].to(device))
        zm_s, _ = model.mass_encoder(x_t['mass'].to(device))
        zr_s, _ = model.rna_encoder(x_t['rna'].to(device))

    cos_m_obs_mean = float(_torch_cos_sim_rows(zg_s, zm_s).mean().cpu())
    cos_r_obs_mean = float(_torch_cos_sim_rows(zg_s, zr_s).mean().cpu())

    rng = np.random.default_rng(seed)
    N = zg_s.shape[0]
    null_m, null_r = np.zeros(n_perm), np.zeros(n_perm)
    for b in range(n_perm):
        idx = torch.as_tensor(rng.permutation(N), dtype=torch.long, device=device)
        null_m[b] = float(_torch_cos_sim_rows(zg_s, zm_s.index_select(0, idx)).mean().cpu())
        null_r[b] = float(_torch_cos_sim_rows(zg_s, zr_s.index_select(0, idx)).mean().cpu())

    rand_mean_m, rand_mean_r = float(np.mean(null_m)), float(np.mean(null_r))

    pd.DataFrame([{
        'seed': seed, 'fold': fold_name,
        'matched_mean_mass': cos_m_obs_mean, 'rand_mean_mass': rand_mean_m,
        'delta_mean_mass': cos_m_obs_mean - rand_mean_m,
        'perm_p_mass': _perm_pvalue_greater(cos_m_obs_mean, null_m),
        'matched_mean_rna': cos_r_obs_mean, 'rand_mean_rna': rand_mean_r,
        'delta_mean_rna': cos_r_obs_mean - rand_mean_r,
        'perm_p_rna': _perm_pvalue_greater(cos_r_obs_mean, null_r)
    }]).to_csv(out_dir / "tables" / "within_pair_summary.csv", index=False)

def coupling_consistency_for_fold(seed: int, fold_name: str, model: DecoupledModel,
                                  gly_feats: np.ndarray, gly_labels_enc: np.ndarray,
                                  mass_feats: np.ndarray, mass_labels_enc: np.ndarray,
                                  rna_feats: np.ndarray, rna_labels_enc: np.ndarray,
                                  saved_ps_ot: dict, out_root: Path, k_list=(1, 5, 10)):
    out_dir = out_root / f"seed_{seed}" / "coupling" / fold_name
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    rskf_gly = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)
    rskf_mass = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)
    rskf_rna  = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=seed)

    global_idx = _get_global_fold_index(fold_name, cfg.N_SPLITS_K_FOLD)
    gly_train_idx, gly_val_idx = list(rskf_gly.split(gly_feats, gly_labels_enc))[global_idx]
    mass_train_idx, _ = list(rskf_mass.split(mass_feats, mass_labels_enc))[global_idx]
    rna_train_idx, _  = list(rskf_rna.split(rna_feats, rna_labels_enc))[global_idx]

    gly_scaler = StandardScaler().fit(gly_feats[gly_train_idx])
    mass_scaler = StandardScaler().fit(mass_feats[mass_train_idx])
    rna_scaler  = StandardScaler().fit(rna_feats[rna_train_idx])

    Xg_va = torch.tensor(gly_scaler.transform(gly_feats[gly_val_idx]), dtype=torch.float32)
    Xm_tr = torch.tensor(mass_scaler.transform(mass_feats[mass_train_idx]), dtype=torch.float32)
    Xr_tr = torch.tensor(rna_scaler.transform(rna_feats[rna_train_idx]), dtype=torch.float32)

    with torch.no_grad():
        zs_g, _ = model.gly_encoder(Xg_va.to(cfg.DEVICE))
        zs_m, _ = model.mass_encoder(Xm_tr.to(cfg.DEVICE))
        zs_r, _ = model.rna_encoder(Xr_tr.to(cfg.DEVICE))

    Dm = _pairwise_euclidean(zs_g, zs_m)
    Dr = _pairwise_euclidean(zs_g, zs_r)

    ps_g_va = saved_ps_ot['ps_val_gly']
    ps_m_tr = saved_ps_ot['ps_pool_mass']
    ps_r_tr = saved_ps_ot['ps_pool_rna']
    metric_mass = _get_ps_cost_metric("mass")
    metric_rna = _get_ps_cost_metric("rna")

    Cm = _ps_cost_matrix(ps_g_va, ps_m_tr, metric_mass)
    Cr = _ps_cost_matrix(ps_g_va, ps_r_tr, metric_rna)

    # Spearman ρ（cost vs latent dist）
    rho_m = spearmanr(Cm.ravel(), Dm.ravel()).correlation
    rho_r = spearmanr(Cr.ravel(), Dr.ravel()).correlation
    pd.DataFrame([{
        'seed': seed, 'fold': fold_name,
        'spearman_cost_zs_mass': rho_m,
        'spearman_cost_zs_rna': rho_r
    }]).to_csv(out_dir / "tables" / "cost_latent_consistency.csv", index=False)

    rng = np.random.default_rng(seed * 10_000 + global_idx)
    Cm_shuf = Cm[:, rng.permutation(Cm.shape[1])]
    Cr_shuf = Cr[:, rng.permutation(Cr.shape[1])]
    rho_m_neg = spearmanr(Cm_shuf.ravel(), Dm.ravel()).correlation
    rho_r_neg = spearmanr(Cr_shuf.ravel(), Dr.ravel()).correlation
    pd.DataFrame([{
        'seed': seed, 'fold': fold_name,
        'spearman_cost_zs_mass': rho_m_neg,
        'spearman_cost_zs_rna': rho_r_neg,
        'type': 'shuffled_cost'
    }]).to_csv(out_dir / "tables" / "cost_latent_consistency_negcontrol.csv", index=False)

    # KNN overlap
    overlaps = [{'k': k, 'mass_overlap': _knn_overlap(Cm, Dm, k), 'rna_overlap': _knn_overlap(Cr, Dr, k)} for k in k_list]
    pd.DataFrame(overlaps).to_csv(out_dir / "tables" / "knn_overlap.csv", index=False)

    def _compute_hit_at_k(cost_matrix, latent_dist, k_list):
        n_queries = cost_matrix.shape[0]
        nearest_by_cost = np.argmin(cost_matrix, axis=1)
        ranks = []
        for i in range(n_queries):
            sorted_indices = np.argsort(latent_dist[i])
            rank = np.where(sorted_indices == nearest_by_cost[i])[0]
            ranks.append(int(rank[0] + 1) if len(rank) > 0 else int(latent_dist.shape[1]))
        out = {'mean_rank': float(np.mean(ranks))}
        for k in k_list:
            out[f'hit@{k}'] = float(np.mean(np.array(ranks) <= k))
        return out

    hit_results = []
    hit_mass = _compute_hit_at_k(Cm, Dm, k_list); hit_mass.update({'modality': 'Mass', 'seed': seed, 'fold': fold_name})
    hit_rna  = _compute_hit_at_k(Cr, Dr, k_list); hit_rna.update({'modality': 'RNA',  'seed': seed, 'fold': fold_name})
    hit_results.extend([hit_mass, hit_rna])
    pd.DataFrame(hit_results).to_csv(out_dir / "tables" / "hit_at_k.csv", index=False)

    def _linear_cka(X, Y):
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        Kx = X @ X.T
        Ky = Y @ Y.T
        hsic = np.trace(Kx @ Ky)
        norm_x = np.sqrt(np.trace(Kx @ Kx))
        norm_y = np.sqrt(np.trace(Ky @ Ky))
        if norm_x < 1e-10 or norm_y < 1e-10:
            return 0.0
        return float(hsic / (norm_x * norm_y))

    Gm = saved_ps_ot['G_mass']
    Gr = saved_ps_ot['G_rna']

    zs_g_np = zs_g.detach().cpu().numpy()
    zs_m_np = zs_m.detach().cpu().numpy()
    zs_r_np = zs_r.detach().cpu().numpy()

    # OT matched
    zs_m_ot = Gm @ zs_m_np
    zs_r_ot = Gr @ zs_r_np
    cka_m_ot = _linear_cka(zs_g_np, zs_m_ot)
    cka_r_ot = _linear_cka(zs_g_np, zs_r_ot)

    perm_m = rng.permutation(Gm.shape[1])
    perm_r = rng.permutation(Gr.shape[1])
    Gm_colperm = Gm[:, perm_m]
    Gr_colperm = Gr[:, perm_r]
    cka_m_colperm = _linear_cka(zs_g_np, Gm_colperm @ zs_m_np)
    cka_r_colperm = _linear_cka(zs_g_np, Gr_colperm @ zs_r_np)

    # Negative control: random row-stochastic coupling (Dirichlet)
    def _random_row_stochastic(n, m, rng_):
        return rng_.dirichlet(np.ones(m), size=n)

    Gm_rand = _random_row_stochastic(Gm.shape[0], Gm.shape[1], rng)
    Gr_rand = _random_row_stochastic(Gr.shape[0], Gr.shape[1], rng)
    cka_m_rand = _linear_cka(zs_g_np, Gm_rand @ zs_m_np)
    cka_r_rand = _linear_cka(zs_g_np, Gr_rand @ zs_r_np)

    cka_rows = [
        {
            'seed': seed, 'fold': fold_name, 'modality': 'Mass',
            'cka_ot': cka_m_ot,
            'cka_colperm': cka_m_colperm,
            'cka_random': cka_m_rand,
            'delta_vs_colperm': cka_m_ot - cka_m_colperm,
            'delta_vs_random': cka_m_ot - cka_m_rand,
        },
        {
            'seed': seed, 'fold': fold_name, 'modality': 'RNA',
            'cka_ot': cka_r_ot,
            'cka_colperm': cka_r_colperm,
            'cka_random': cka_r_rand,
            'delta_vs_colperm': cka_r_ot - cka_r_colperm,
            'delta_vs_random': cka_r_ot - cka_r_rand,
        }
    ]
    pd.DataFrame(cka_rows).to_csv(out_dir / "tables" / "cka_consistency.csv", index=False)


def model_internals_and_consistency(seed: int, fold_name: str, model: DecoupledModel,
                                    x_t: Dict[str, torch.Tensor], y_val: torch.Tensor,
                                    saved_ps_ot: dict, out_root: Path):
    out_dir = out_root / f"seed_{seed}" / "internals" / fold_name
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    device = cfg.DEVICE
    with torch.no_grad():
        logits, _, reps = model(x_t['gly'].to(device), x_t['mass'].to(device), x_t['rna'].to(device), return_attn=True)
        scores = F.softmax(logits, dim=1)
    auc0, ap0 = _auc_ap(y_val, scores)

    attn_col_mean = reps.get('attn', torch.zeros((len(y_val), 3, 3), device=device)).mean(dim=1).detach().cpu().numpy()  # [N, 3]
    attn_rows = []
    for i in range(len(y_val)):
        attn_rows.append({
            'label': int(y_val[i]),
            'attn_to_gly': float(attn_col_mean[i, 0]),
            'attn_to_mass': float(attn_col_mean[i, 1]),
            'attn_to_rna': float(attn_col_mean[i, 2])
        })
    pd.DataFrame(attn_rows).to_csv(out_dir / "tables" / "attention_per_sample.csv", index=False)

    lomo_rows = []
    auc_drops, ap_drops, attn_means = {}, {}, {}
    for j, m in enumerate(MODALITY_ORDER):
        with torch.no_grad():
            logits_m, _, _ = model(x_t['gly'].to(device), x_t['mass'].to(device), x_t['rna'].to(device),
                                   mask_shared={k: (k == m) for k in MODALITY_ORDER})
            s_m = F.softmax(logits_m, dim=1)
        auc_m, ap_m = _auc_ap(y_val, s_m)
        lomo_rows.append({'modality': m, 'auc_drop': auc0 - auc_m, 'ap_drop': ap0 - ap_m})
        auc_drops[m] = auc0 - auc_m
        ap_drops[m] = ap0 - ap_m
        attn_means[m] = float(attn_col_mean[:, j].mean())
    pd.DataFrame(lomo_rows).to_csv(out_dir / "tables" / "lomo_summary.csv", index=False)

    pd.DataFrame([{
        'seed': seed, 'fold': fold_name,
        **{f'attn_mean_{m}': attn_means[m] for m in MODALITY_ORDER},
        **{f'auc_drop_{m}': auc_drops[m] for m in MODALITY_ORDER},
        **{f'ap_drop_{m}': ap_drops[m] for m in MODALITY_ORDER}
    }]).to_csv(out_root / f"attn_lomo_consistency_seed_{seed}.csv", index=False,
               mode='a', header=not (out_root / f"attn_lomo_consistency_seed_{seed}.csv").exists())

    Gm = saved_ps_ot['G_mass']; Gr = saved_ps_ot['G_rna']
    ent_mass = _row_entropy(Gm); ent_rna = _row_entropy(Gr)
    conf = scores.max(dim=1).values.cpu().numpy()
    pred = scores.argmax(dim=1).cpu().numpy()
    correct = (pred == y_val.cpu().numpy()).astype(float)

    # --- ADD THIS BLOCK ---
    per_sample_df = pd.DataFrame({
        'label': y_val.cpu().numpy(),
        'confidence': conf,
        'is_correct': correct,
        'entropy_mass': ent_mass,
        'entropy_rna': ent_rna
    })
    per_sample_df.to_csv(out_dir / "tables" / "per_sample_performance.csv", index=False)
    # --- END ADD ---

    corr_rows = [{
        'seed': seed, 'fold': fold_name,
        'rho_entropy_mass_conf': float(spearmanr(ent_mass, conf).correlation),
        'rho_entropy_rna_conf': float(spearmanr(ent_rna, conf).correlation),
        'rho_entropy_mass_acc': float(spearmanr(ent_mass, correct).correlation),
        'rho_entropy_rna_acc': float(spearmanr(ent_rna, correct).correlation),
    }]
    pd.DataFrame(corr_rows).to_csv(out_root / f"entropy_confidence_corr_seed_{seed}.csv", index=False,
                                   mode='a', header=not (out_root / f"entropy_confidence_corr_seed_{seed}.csv").exists())

    if HAS_CAPTUM:
        class Wrapper(torch.nn.Module):
            def __init__(self, base):
                super().__init__(); self.base = base
            def forward(self, gly, mass, rna):
                return self.base(gly, mass, rna)[0]

        wrapper = Wrapper(model).eval().to(device)
        inputs = (x_t['gly'].to(device), x_t['mass'].to(device), x_t['rna'].to(device))
        targets = y_val.to(device)
        ig = IntegratedGradients(wrapper)
        reference_inputs = tuple(torch.zeros_like(i) for i in inputs)
        atts = ig.attribute(inputs, target=targets, baselines=reference_inputs, n_steps=50)
    else:
        gly_i = x_t['gly'].to(device).detach().requires_grad_(True)
        mass_i = x_t['mass'].to(device).detach().requires_grad_(True)
        rna_i = x_t['rna'].to(device).detach().requires_grad_(True)
        logits = model(gly_i, mass_i, rna_i)[0]
        logits.gather(1, y_val.to(device).view(-1, 1)).sum().backward()
        atts = (gly_i.grad * gly_i, mass_i.grad * mass_i, rna_i.grad * rna_i)

    for m, a in zip(MODALITY_ORDER, atts):
        mean_abs_attr = a.abs().mean(dim=0).detach().cpu().numpy()
        df = pd.DataFrame({'feature_index': np.arange(len(mean_abs_attr)), 'mean_abs_attr': mean_abs_attr})
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "tables" / f"ig_scores_{m}.csv", index=False)


def run_one_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gly_y_str, gly_x, gly_cols = _load_csv(cfg.GLY_PATH)
    mass_y_str, mass_x, _ = _load_csv(cfg.MASS_PATH)
    rna_y_str, rna_x, _ = _load_csv(cfg.RNA_PATH)

    label_map = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
    gly_y = pd.Series(gly_y_str).map(label_map).values
    mass_y = pd.Series(mass_y_str).map(label_map).values
    rna_y = pd.Series(rna_y_str).map(label_map).values
    input_dims = (gly_x.shape[1], mass_x.shape[1], rna_x.shape[1])

    exp_path = BASE_DIR / f"SLEmodel_Run_Seed_{seed}" / f"master_seed_{seed}"
    if not exp_path.exists():
        print(f"[Seed {seed}] Skip: path not found: {exp_path}")
        return
    folds = sorted([p.name for p in exp_path.glob("Repeat_*") if p.is_dir()])

    for fold_name in folds:
        print(f"[Seed {seed}] Processing {fold_name}...")
        built = build_val_inputs_for_fold(seed, fold_name, gly_x, gly_y)
        if built is None:
            print(f"  Skipping {fold_name}: validation reconstruction data not found.")
            continue
        x_t, y_val = built
        model = get_model_for_seed_fold(seed, fold_name, input_dims)

        res = ps_ot_qc_for_fold(seed, fold_name, gly_x, gly_y, mass_x, mass_y, rna_x, rna_y)
        ps_g_va, ps_m_tr, ps_r_tr, Gm, Gr, Gm_rand, Gr_rand, Gm_shuf, Gr_shuf = res

        # within-pair Δcosine
        within_pair_similarity_for_fold(seed, fold_name, model, x_t, OUT_DIR)

        saved_ps_ot = {'ps_val_gly': ps_g_va, 'ps_pool_mass': ps_m_tr, 'ps_pool_rna': ps_r_tr,
                       'G_mass': Gm, 'G_rna': Gr}
        coupling_consistency_for_fold(seed, fold_name, model, gly_x, gly_y, mass_x, mass_y, rna_x, rna_y,
                                      saved_ps_ot, OUT_DIR, k_list=(5, 10))

        saved_ps_ot_full = {'G_mass': Gm, 'G_rna': Gr}
        model_internals_and_consistency(seed, fold_name, model, x_t, y_val, saved_ps_ot_full, OUT_DIR)

def main():
    print("=== Running Interpretability Analysis (Data Generation) ===")
    print(f"Detected seeds: {SEEDS}")
    for seed in SEEDS:
        run_one_seed(seed)
    print("\nAll raw interpretability results saved to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
