# shap_analysis.py
# Standalone SHAP computation and result storage. No visualization code.

import os
import json
import argparse
import warnings
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import shap

from slemodel import config as cfg
from slemodel.dataset import StaticMatchedDataset
from slemodel.utils import (
    get_ot_coupling,
    get_ot_coupling_modality_specific,
    train_classifier,
    get_propensity_scores,
)
from slemodel.models import DecoupledModel, PropensityScoreClassifier
from .feature_annotations import build_feature_label_maps, is_exogenous_annotation

# ========== Logging ==========
logger = logging.getLogger(__name__)

# ========== Global Configuration ==========
GLOBAL_SEED = 202510
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

SEEDS = [42, 100, 2025, 7, 123]
EXP_TAG = cfg.EXPERIMENT_TAG
BASE_DIR = Path(cfg.BASE_EXPERIMENT_DIR)
OUT_DIR = Path("clinical_explain_results") / "shap_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Class ordering for consistent analysis
CLASS_ORDER = [1, 0, 2]  # Active, Stable, Control


# ========== Utility Functions ==========

def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def _modality_of(col: str) -> str:
    """Extract modality prefix from feature name."""
    return col.split("::", 1)[0] if "::" in col else "?"


def get_active_class_index() -> int:
    """Get the index of the 'Active' class from config."""
    for key, value in cfg.VisualizationConfig.LABEL_NAMES.items():
        if value == "Active":
            return key
    raise ValueError("'Active' class not found in label map configuration.")


# ========== Data Loading Functions ==========

def load_csv_as_arrays(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load CSV data and return IDs, features, labels, and feature names."""
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].values
    y_str = df.iloc[:, 1].values
    X = df.iloc[:, 2:].values
    featnames = df.columns[2:].tolist()
    return ids, X, y_str, featnames


def map_labels(y_str: np.ndarray) -> np.ndarray:
    """Map string labels to integer indices."""
    label_map = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
    return pd.Series(y_str).map(label_map).values


def build_splits(y_enc: np.ndarray, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build cross-validation splits."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.N_SPLITS_K_FOLD,
        n_repeats=cfg.N_REPEATS,
        random_state=seed
    )
    return list(rskf.split(np.zeros_like(y_enc), y_enc))


def get_feature_names(meta: Dict) -> List[str]:
    """Build full feature names with modality prefixes."""
    g = [f"gly::{n}" for n in meta["gly_featnames"]]
    m = [f"mass::{n}" for n in meta["mass_featnames"]]
    r = [f"rna::{n}" for n in meta["rna_featnames"]]
    return g + m + r


# ========== Model Loading Functions ==========

def fit_ps_models(
    gly_x_train: np.ndarray,
    gly_y_train: np.ndarray,
    mass_x_pool: np.ndarray,
    mass_y_pool: np.ndarray,
    rna_x_pool: np.ndarray,
    rna_y_pool: np.ndarray,
    device: str
) -> Tuple:
    """Fit propensity score classifiers for each modality."""
    g = train_classifier(
        PropensityScoreClassifier(gly_x_train.shape[1]),
        torch.tensor(gly_x_train, dtype=torch.float32),
        torch.tensor(gly_y_train, dtype=torch.long),
        device,
        epochs=cfg.EPOCHS_PS
    )
    m = train_classifier(
        PropensityScoreClassifier(mass_x_pool.shape[1]),
        torch.tensor(mass_x_pool, dtype=torch.float32),
        torch.tensor(mass_y_pool, dtype=torch.long),
        device,
        epochs=cfg.EPOCHS_PS
    )
    r = train_classifier(
        PropensityScoreClassifier(rna_x_pool.shape[1]),
        torch.tensor(rna_x_pool, dtype=torch.float32),
        torch.tensor(rna_y_pool, dtype=torch.long),
        device,
        epochs=cfg.EPOCHS_PS
    )
    return g, m, r


def get_val_loader_for_fold(
    seed: int,
    fold_idx: int,
    device: str
) -> Tuple[DataLoader, Dict]:
    """
    Get validation DataLoader for a specific fold.

    Prioritizes loading saved OT couplings from training for reproducibility.
    Falls back to recomputing if not available.
    """
    gid, gX, gyS, gnames = load_csv_as_arrays(cfg.GLY_PATH)
    mid, mX, myS, mnames = load_csv_as_arrays(cfg.MASS_PATH)
    rid, rX, ryS, rnames = load_csv_as_arrays(cfg.RNA_PATH)

    gy, my, ry = map_labels(gyS), map_labels(myS), map_labels(ryS)

    splits_g = build_splits(gy, seed)
    splits_m = build_splits(my, seed)
    splits_r = build_splits(ry, seed)

    g_tr, g_va = splits_g[fold_idx]
    m_tr, _ = splits_m[fold_idx]
    r_tr, _ = splits_r[fold_idx]

    # Fit scalers on training data
    g_scaler = StandardScaler().fit(gX[g_tr])
    m_scaler = StandardScaler().fit(mX[m_tr])
    r_scaler = StandardScaler().fit(rX[r_tr])

    gX_tr = g_scaler.transform(gX[g_tr])
    gX_va = g_scaler.transform(gX[g_va])
    mX_pool = m_scaler.transform(mX[m_tr])
    rX_pool = r_scaler.transform(rX[r_tr])

    # Convert to tensors
    t_gX = torch.tensor(gX_va, dtype=torch.float32)
    t_gY = torch.tensor(gy[g_va], dtype=torch.long)
    t_mX = torch.tensor(mX_pool, dtype=torch.float32)
    t_rX = torch.tensor(rX_pool, dtype=torch.float32)

    # Try to load saved OT couplings
    run_name = f"SLEmodel_Run_Seed_{seed}"
    fold_name = f"Repeat_1-Fold_{fold_idx+1}"
    fold_dir = BASE_DIR / EXP_TAG / run_name / f"master_seed_{seed}" / fold_name
    npz_path = fold_dir / "val_ps_ot.npz"

    P_g2m = None
    P_g2r = None

    if npz_path.exists():
        try:
            dat = np.load(npz_path, allow_pickle=True)
            Gm = dat["G_mass"]
            Gr = dat["G_rna"]

            if Gm.shape[0] == t_gX.shape[0] and Gm.shape[1] == t_mX.shape[0]:
                P_g2m = torch.tensor(Gm, dtype=torch.float32)
            else:
                warnings.warn(f"G_mass shape mismatch: {Gm.shape}")

            if Gr.shape[0] == t_gX.shape[0] and Gr.shape[1] == t_rX.shape[0]:
                P_g2r = torch.tensor(Gr, dtype=torch.float32)
            else:
                warnings.warn(f"G_rna shape mismatch: {Gr.shape}")

            if P_g2m is not None and P_g2r is not None:
                logger.info(f"Loaded OT couplings from: {npz_path}")
        except Exception as e:
            warnings.warn(f"Failed to load {npz_path}: {e}")
            P_g2m, P_g2r = None, None

    # Fallback: recompute PS + OT
    if P_g2m is None or P_g2r is None:
        with torch.enable_grad():
            gclf, mclf, rclf = fit_ps_models(
                gX_tr, gy[g_tr],
                mX_pool, my[m_tr],
                rX_pool, ry[r_tr],
                device
            )

        with torch.no_grad():
            ps_g_va = get_propensity_scores(gclf, torch.tensor(gX_va, dtype=torch.float32), device)
            ps_m_po = get_propensity_scores(mclf, torch.tensor(mX_pool, dtype=torch.float32), device)
            ps_r_po = get_propensity_scores(rclf, torch.tensor(rX_pool, dtype=torch.float32), device)

            P_g2m = get_ot_coupling_modality_specific(
                ps_g_va, ps_m_po,
                modality="mass",
                reg_mass=cfg.OT_REG_MASS,
                reg_rna=cfg.OT_REG_RNA,
                method_mass=cfg.OT_METHOD_MASS,
                method_rna=cfg.OT_METHOD_RNA,
                purpose="eval",
                cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
            )

            P_g2r = get_ot_coupling_modality_specific(
                ps_g_va, ps_r_po,
                modality="rna",
                reg_mass=cfg.OT_REG_MASS,
                reg_rna=cfg.OT_REG_RNA,
                method_mass=cfg.OT_METHOD_MASS,
                method_rna=cfg.OT_METHOD_RNA,
                purpose="eval",
                cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
            )

    ds = StaticMatchedDataset(t_gX, t_gY, t_mX, t_rX, P_g2m, P_g2r, gid[g_va])
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    meta = {
        "gly_ids": gid[g_va],
        "gly_featnames": gnames,
        "mass_featnames": mnames,
        "rna_featnames": rnames
    }
    return dl, meta


def load_best_model_for_fold(
    seed: int,
    fold_idx: int,
    device: str
) -> Tuple[DecoupledModel, str]:
    """Load the best model checkpoint for a specific fold."""
    run_name = f"SLEmodel_Run_Seed_{seed}"
    fold_name = f"Repeat_1-Fold_{fold_idx+1}"
    fold_dir = BASE_DIR / EXP_TAG / run_name / f"master_seed_{seed}" / fold_name
    ckpt = fold_dir / f"best_model_{fold_name}.pth"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    _, gX, _, _ = load_csv_as_arrays(cfg.GLY_PATH)
    _, mX, _, _ = load_csv_as_arrays(cfg.MASS_PATH)
    _, rX, _, _ = load_csv_as_arrays(cfg.RNA_PATH)

    model = DecoupledModel(gX.shape[1], mX.shape[1], rX.shape[1]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, fold_name


# ========== Model Selection ==========

def load_all_val_metrics() -> pd.DataFrame:
    """Load validation metrics from all seeds."""
    rows = []
    for seed in SEEDS:
        mp = (BASE_DIR / EXP_TAG / f"SLEmodel_Run_Seed_{seed}" /
              f"master_seed_{seed}" / "metrics" / "classification_metrics_per_fold.csv")
        if mp.exists():
            df = pd.read_csv(mp)
            df['seed'] = seed
            rows.append(df)

    if not rows:
        raise FileNotFoundError("No classification_metrics_per_fold.csv found.")

    df = pd.concat(rows, ignore_index=True)
    df = df[df['set'] == 'validation'].copy()
    return df


def pick_model_of_record() -> Tuple[int, int, str]:
    """
    Select the Model-of-Record (MoR) based on median performance.

    Returns:
        seed: The seed with median performance
        fold_idx: The fold index closest to mean AUC within that seed
        fold_name: The fold name string
    """
    df = load_all_val_metrics()
    seed_stats = df.groupby('seed')['macro_roc_auc'].mean().sort_values()
    med_rank = len(seed_stats) // 2
    seed_star = int(seed_stats.index.tolist()[med_rank])

    seed_df = df[df['seed'] == seed_star].copy()
    seed_mean = seed_df['macro_roc_auc'].mean()
    seed_df['dist'] = (seed_df['macro_roc_auc'] - seed_mean).abs()
    row = seed_df.sort_values('dist').iloc[0]
    fold_name = row['fold']

    try:
        fold_idx = int(str(fold_name).split("Fold_")[1]) - 1
    except Exception:
        fold_idx = 0

    return seed_star, fold_idx, fold_name


# ========== SHAP Computation ==========

class SingleInputWrapper(torch.nn.Module):
    """Wrap DecoupledModel to accept concatenated (gly|mass|rna) input."""

    def __init__(self, model: DecoupledModel, gly_dim: int, mass_dim: int, rna_dim: int):
        super().__init__()
        self.model = model.eval()
        self.split_sizes = [gly_dim, mass_dim, rna_dim]

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        g, m, r = torch.split(x_cat, self.split_sizes, dim=1)
        logits, _, _ = self.model(g, m, r)
        return logits


def compute_shap_for_fold(
    model: DecoupledModel,
    val_loader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    background_size: int = 75,
    nsamples: int = 200,
    random_state: int = 7
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute SHAP values for a model on validation data.

    Args:
        model: The trained DecoupledModel
        val_loader: DataLoader for validation data
        device: Computing device
        max_batches: Maximum batches to process (None for all)
        background_size: Number of samples for SHAP background
        nsamples: Number of samples for SHAP approximation
        random_state: Random seed for reproducibility

    Returns:
        shap_dict: Dictionary containing:
            - "macro": (N, F) mean absolute SHAP across classes
            - "per_class_stack": (N, F, C) SHAP values per class
            - "y_true": (N,) true labels
        X_total: (N, F) input feature matrix
    """
    torch.manual_seed(random_state)
    model.eval()

    # Infer dimensions
    gly_dim = model.gly_encoder.shared_encoder[0].in_features
    mass_dim = model.mass_encoder.shared_encoder[0].in_features
    rna_dim = model.rna_encoder.shared_encoder[0].in_features
    wrapper = SingleInputWrapper(model, gly_dim, mass_dim, rna_dim).to(device).eval()

    # Collect background samples
    bg_g, bg_m, bg_r = [], [], []
    collected = 0
    for b in val_loader:
        bg_g.append(b['gly_x'])
        bg_m.append(b['mass_x'])
        bg_r.append(b['rna_x'])
        collected += b['gly_x'].shape[0]
        if collected >= background_size:
            break

    if not bg_g:
        raise RuntimeError("Empty validation loader.")

    bg_g = torch.cat(bg_g, 0)[:background_size].to(device)
    bg_m = torch.cat(bg_m, 0)[:background_size].to(device)
    bg_r = torch.cat(bg_r, 0)[:background_size].to(device)
    background = torch.cat([bg_g, bg_m, bg_r], 1)

    # Collect foreground samples
    fg_g, fg_m, fg_r, y_list = [], [], [], []
    for i, b in enumerate(val_loader):
        fg_g.append(b['gly_x'].to(device))
        fg_m.append(b['mass_x'].to(device))
        fg_r.append(b['rna_x'].to(device))
        y_list.append(b['label'].cpu().numpy())
        if max_batches is not None and (i + 1) >= max_batches:
            break

    fg_g = torch.cat(fg_g, 0)
    fg_m = torch.cat(fg_m, 0)
    fg_r = torch.cat(fg_r, 0)
    foreground = torch.cat([fg_g, fg_m, fg_r], 1)
    y_true = np.concatenate(y_list) if y_list else np.array([])

    # Compute SHAP values
    explainer = shap.GradientExplainer(wrapper, background)
    shap_out = explainer.shap_values(foreground, nsamples=nsamples)

    # Process SHAP output into (N, F, C) format
    phi_per_class = {}
    if isinstance(shap_out, list):
        if len(shap_out) != cfg.NUM_CLASSES:
            raise RuntimeError(f"Unexpected shap outputs: {len(shap_out)}")
        for c in range(cfg.NUM_CLASSES):
            arr = _to_numpy(shap_out[c])
            if arr.ndim != 2:
                raise RuntimeError(f"Unexpected shape for class {c}: {arr.shape}")
            phi_per_class[c] = arr
    elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3 and shap_out.shape[2] == cfg.NUM_CLASSES:
        for c in range(cfg.NUM_CLASSES):
            phi_per_class[c] = shap_out[:, :, c]
    else:
        info = getattr(shap_out, "shape", f"len={len(shap_out)}")
        raise RuntimeError(f"Unexpected SHAP output format: {info}")

    n_samples = min(v.shape[0] for v in phi_per_class.values())
    n_features = next(iter(phi_per_class.values())).shape[1]

    phi_stack = np.empty((n_samples, n_features, cfg.NUM_CLASSES), dtype=np.float32)
    for c in range(cfg.NUM_CLASSES):
        phi_stack[:, :, c] = phi_per_class[c][:n_samples]

    phi_macro = np.abs(phi_stack).mean(axis=2)
    X_total = foreground[:n_samples].detach().cpu().numpy()
    y_true = y_true[:n_samples]

    return {
        "macro": phi_macro,
        "per_class_stack": phi_stack,
        "y_true": y_true
    }, X_total


# ========== Model Inference ==========

@torch.no_grad()
def infer_all(
    model: DecoupledModel,
    loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on all samples in loader.

    Returns:
        y_true: True labels
        y_prob: Predicted probabilities
        ids: Sample IDs
    """
    y_true, y_prob, ids = [], [], []
    for batch in loader:
        g = batch['gly_x'].to(device)
        m = batch['mass_x'].to(device)
        r = batch['rna_x'].to(device)
        logits, _, _ = model(g, m, r)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob.append(prob)
        y_true.append(batch['label'].cpu().numpy())
        if 'id' in batch:
            ids += list(batch['id'])

    return np.concatenate(y_true), np.concatenate(y_prob), np.array(ids)


@torch.no_grad()
def aggregate_attention_matrix(
    model: DecoupledModel,
    loader: DataLoader,
    device: str
) -> np.ndarray:
    """
    Compute average cross-modal attention matrix over validation set.

    Returns:
        (3, 3) numpy array of attention weights, or NaN array if unavailable.
    """
    mats = []
    model.eval()

    for batch in loader:
        g = batch['gly_x'].to(device)
        m = batch['mass_x'].to(device)
        r = batch['rna_x'].to(device)

        _, _, reps = model(g, m, r, return_attn=True)
        attn = reps.get("attn", None)

        if attn is None:
            attn = getattr(model, "cached_attention", None)

        if attn is None:
            continue

        a = attn
        if torch.is_tensor(a):
            a = a.detach()

        # [B, H, 3, 3] -> [B, 3, 3]
        if a.dim() == 4:
            a = a.mean(dim=1)

        if a.dim() != 3 or a.shape[-2:] != (3, 3):
            continue

        a_mean = a.mean(dim=0).cpu().numpy()
        mats.append(a_mean)

    if not mats:
        return np.full((3, 3), np.nan)

    return np.mean(np.stack(mats, axis=0), axis=0)


# ========== OOF Predictions ==========

def collect_oof_predictions() -> List[Dict]:
    """Collect out-of-fold predictions from all seeds."""
    records = []
    for seed in SEEDS:
        exp_path = (BASE_DIR / EXP_TAG / f"SLEmodel_Run_Seed_{seed}" /
                   f"master_seed_{seed}" / "oof_predictions")
        val_file = exp_path / "oof_val_predictions_repeat_1.npz"
        if val_file.exists():
            data = np.load(val_file)
            records.append({
                "seed": seed,
                "true": data["true_labels"],
                "pred_scores": data["pred_scores"]
            })
    return records


# ========== Feature Filtering ==========

def load_mass_quality_flags(
    path: Path = Path("processed_mass_spec") / "bin_quality_flags.csv"
) -> Tuple[Dict[str, bool], Optional[pd.DataFrame]]:
    """
    Load mass quality flags for filtering unreliable features.

    Returns:
        ok_map: bin_label -> True/False (detect_frac >= 0.30 and mad_ppm <= 60)
        meta: DataFrame for debugging (or None if file not found)
    """
    if not path.exists():
        return {}, None

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    lab = cols.get("bin_label", "bin_label")
    flag = cols.get("detect_ok_30pct", "detect_ok_30pct")
    mad = cols.get("mad_ppm", "mad_ppm")

    ok_map = {}
    for _, r in df.iterrows():
        try:
            ok = bool(r.get(flag, True))
            if mad in r and pd.notna(r[mad]):
                ok = ok and (float(r[mad]) <= 60.0)
            ok_map[str(r[lab])] = ok
        except Exception:
            continue

    return ok_map, df


def build_allowed_feature_mask(
    feature_names: List[str],
    annotated_flags: np.ndarray,
    display_names: Optional[List[str]] = None,
    detect_frac_threshold: float = 0.30
) -> np.ndarray:
    """
    Build mask for features allowed in analysis/visualization.

    Rules for mass features:
    - Must be annotated (is_annotated=True)
    - Must pass quality check (detect_frac >= threshold, mad_ppm <= 60)
    - Exogenous compounds are excluded

    Non-mass features are always allowed.
    """
    ok_map, _ = load_mass_quality_flags()
    allowed = np.ones(len(feature_names), dtype=bool)

    for i, fn in enumerate(feature_names):
        mod = _modality_of(fn)

        if mod != "mass":
            allowed[i] = True
            continue

        # Must be annotated
        if not bool(annotated_flags[i]):
            allowed[i] = False
            continue

        # Check if exogenous
        if display_names is not None:
            disp = display_names[i]
            if is_exogenous_annotation(disp):
                allowed[i] = False
                continue

        # Quality check
        try:
            bin_label = fn.split("::", 1)[1]
        except Exception:
            bin_label = ""

        if ok_map:
            allowed[i] = ok_map.get(bin_label, True)
        else:
            allowed[i] = True

    return allowed


def deduplicate_by_base_metabolite(
    display_feature_names: List[str],
    allowed_mask: np.ndarray,
    phi_macro: np.ndarray
) -> np.ndarray:
    """
    Deduplicate features by base metabolite name, keeping the one with highest contribution.

    Removes adduct forms and m/z annotations to identify the base metabolite,
    then keeps only the highest-contributing instance.
    """
    mean_abs = np.mean(np.abs(phi_macro), axis=0)

    def base_label(lbl: str) -> str:
        s = re.sub(r"\s*\[.*?\]", "", lbl)        # Remove [M+Na]+ etc.
        s = re.sub(r"\s*\(m/z.*?\)", "", s)       # Remove (m/z xxx)
        return s.strip().lower()

    new_mask = allowed_mask.copy()
    idxs = np.where(allowed_mask)[0]
    seen = {}

    # Process by importance (highest first)
    for i in idxs[np.argsort(mean_abs[idxs])[::-1]]:
        key = base_label(display_feature_names[i])
        if key in seen:
            new_mask[i] = False
        else:
            seen[key] = i

    return new_mask


# ========== SHAP Results Export ==========

def export_shap_top_feature_tables(
    phi_macro: np.ndarray,
    phi_per_class_stack: np.ndarray,
    feature_names: List[str],
    display_feature_names: List[str],
    annotated_flags: np.ndarray,
    allowed_mask: Optional[np.ndarray],
    out_dir: Path,
    topk_global: int = 200,
    topk_per_modality: int = 100,
    topk_per_class_pos: int = 80,
    topk_per_class_neg: int = 80,
    y_true: Optional[np.ndarray] = None,
):
    """
    Export SHAP top-feature tables for downstream analysis.

    Outputs (CSV):
        1) shap_top_features_global.csv
        2) shap_top_features_global_allowed.csv
        3) shap_top_features_by_modality.csv
        4) shap_top_features_by_class_directed.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    F = len(feature_names)
    if allowed_mask is None:
        allowed_mask = np.ones(F, dtype=bool)

    # Build base columns
    mods = []
    for fn in feature_names:
        if isinstance(fn, str) and "::" in fn:
            mods.append(fn.split("::", 1)[0])
        else:
            mods.append("?")

    mean_abs_global = np.mean(np.abs(phi_macro), axis=0).reshape(-1)

    df_base = pd.DataFrame({
        "feature_name": feature_names,
        "display_name": display_feature_names,
        "modality": mods,
        "is_annotated": annotated_flags.astype(bool),
        "is_allowed": np.array(allowed_mask, dtype=bool),
        "mean_abs_shap_global": mean_abs_global.astype(float),
    })

    # (1) Global top features
    df_global = df_base.sort_values("mean_abs_shap_global", ascending=False).reset_index(drop=True)
    df_global.insert(0, "rank_global", np.arange(1, len(df_global) + 1))
    df_global.head(topk_global).to_csv(out_dir / "shap_top_features_global.csv", index=False)

    # Global top (allowed only)
    df_global_allowed = df_global[df_global["is_allowed"]].reset_index(drop=True)
    df_global_allowed.insert(0, "rank_global_allowed", np.arange(1, len(df_global_allowed) + 1))
    df_global_allowed.head(topk_global).to_csv(out_dir / "shap_top_features_global_allowed.csv", index=False)

    # (2) Top features by modality
    rows = []
    for m in ["gly", "mass", "rna"]:
        sub = df_base[df_base["modality"] == m].copy()
        if sub.shape[0] == 0:
            continue

        sub = sub.sort_values("mean_abs_shap_global", ascending=False).reset_index(drop=True)
        sub.insert(0, "rank_within_modality", np.arange(1, len(sub) + 1))

        sub["rank_within_modality_allowed"] = pd.Series(pd.NA, index=sub.index, dtype="Int64")
        allowed_bool = sub["is_allowed"].to_numpy(dtype=bool)
        if allowed_bool.any():
            sub.loc[allowed_bool, "rank_within_modality_allowed"] = np.arange(1, allowed_bool.sum() + 1, dtype=int)

        rows.append(sub.head(topk_per_modality))

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_dir / "shap_top_features_by_modality.csv", index=False)

    # (3) Top features by class (directed)
    C = phi_per_class_stack.shape[2]
    class_rows = []

    y_true_arr = None
    if y_true is not None:
        y_true_arr = np.asarray(y_true).reshape(-1)
        if y_true_arr.shape[0] != phi_per_class_stack.shape[0]:
            y_true_arr = None

    for c in range(C):
        class_name = cfg.VisualizationConfig.LABEL_NAMES.get(c, str(c))

        if y_true_arr is not None:
            mask = (y_true_arr == c)
            n_used = int(mask.sum())
            if n_used > 0:
                slice_c = phi_per_class_stack[mask, :, c]
            else:
                slice_c = phi_per_class_stack[:, :, c]
                n_used = int(slice_c.shape[0])
            subset_note = "y_true==class" if mask.sum() > 0 else "all_samples_fallback"
        else:
            slice_c = phi_per_class_stack[:, :, c]
            n_used = int(slice_c.shape[0])
            subset_note = "all_samples"

        dir_mean = np.mean(slice_c, axis=0).reshape(-1)
        abs_mean = np.mean(np.abs(slice_c), axis=0).reshape(-1)

        dfc = pd.DataFrame({
            "class_index": int(c),
            "class_name": class_name,
            "subset": subset_note,
            "n_samples_used": n_used,
            "feature_name": feature_names,
            "display_name": display_feature_names,
            "modality": mods,
            "is_annotated": annotated_flags.astype(bool),
            "is_allowed": np.array(allowed_mask, dtype=bool),
            "directed_mean_shap": dir_mean.astype(float),
            "mean_abs_shap_in_class": abs_mean.astype(float),
        })

        # Positive contributors
        pos = dfc.sort_values("directed_mean_shap", ascending=False).reset_index(drop=True).head(topk_per_class_pos).copy()
        pos.insert(0, "rank_in_class", np.arange(1, len(pos) + 1))
        pos["direction"] = "pos"

        # Negative contributors
        neg = dfc.sort_values("directed_mean_shap", ascending=True).reset_index(drop=True).head(topk_per_class_neg).copy()
        neg.insert(0, "rank_in_class", np.arange(1, len(neg) + 1))
        neg["direction"] = "neg"

        class_rows.append(pos)
        class_rows.append(neg)

    pd.concat(class_rows, ignore_index=True).to_csv(out_dir / "shap_top_features_by_class_directed.csv", index=False)

    logger.info(f"Saved SHAP top-feature tables to: {out_dir}")


def save_shap_results(
    phi_macro: np.ndarray,
    phi_per_class_stack: np.ndarray,
    y_true: np.ndarray,
    X_total: np.ndarray,
    feature_names: List[str],
    display_feature_names: List[str],
    annotated_flags: np.ndarray,
    allowed_mask: np.ndarray,
    out_path: Path,
    additional_meta: Optional[Dict] = None
):
    """
    Save all SHAP computation results to a single NPZ file.

    This enables downstream visualization scripts to load precomputed results.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "phi_macro": phi_macro,
        "phi_stack": phi_per_class_stack,
        "y_true": y_true,
        "X_total": X_total,
        "feature_names": np.array(feature_names, dtype=object),
        "display_feature_names": np.array(display_feature_names, dtype=object),
        "annotated_flags": annotated_flags,
        "allowed_mask": allowed_mask,
    }

    if additional_meta:
        for k, v in additional_meta.items():
            save_dict[f"meta_{k}"] = v

    np.savez(out_path, **save_dict)
    logger.info(f"Saved SHAP results to: {out_path}")


def save_attention_matrix(
    attention_matrix: np.ndarray,
    out_path: Path,
    modality_labels: List[str] = None
):
    """Save attention matrix to NPZ file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if modality_labels is None:
        modality_labels = ["gly", "mass", "rna"]

    np.savez(
        out_path,
        attention_matrix=attention_matrix,
        modality_labels=np.array(modality_labels, dtype=object)
    )
    logger.info(f"Saved attention matrix to: {out_path}")


def save_inference_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ids: np.ndarray,
    out_path: Path
):
    """Save inference results to NPZ file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        y_true=y_true,
        y_prob=y_prob,
        ids=ids
    )
    logger.info(f"Saved inference results to: {out_path}")


def save_calibration_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    target_class: Optional[int] = None
):
    """
    Compute and save calibration statistics.

    If target_class is specified, compute calibration for that class only.
    Otherwise, compute for the 'Active' class.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if target_class is None:
        target_class = get_active_class_index()

    ybin = (y_true == target_class).astype(int)
    p_target = y_prob[:, target_class]

    # Compute Brier scores for different calibration methods
    X_dummy = p_target.reshape(-1, 1)

    try:
        base_lr = LogisticRegression(solver="lbfgs").fit(X_dummy, ybin)
        cal_platt = CalibratedClassifierCV(base_lr, method="sigmoid", cv=5).fit(X_dummy, ybin)
        cal_iso = CalibratedClassifierCV(base_lr, method="isotonic", cv=5).fit(X_dummy, ybin)

        p_raw = base_lr.predict_proba(X_dummy)[:, 1]
        p_platt = cal_platt.predict_proba(X_dummy)[:, 1]
        p_iso = cal_iso.predict_proba(X_dummy)[:, 1]

        b_raw = brier_score_loss(ybin, p_raw)
        b_platt = brier_score_loss(ybin, p_platt)
        b_iso = brier_score_loss(ybin, p_iso)

        if b_iso <= min(b_raw, b_platt):
            best_method = "isotonic"
            p_calibrated = p_iso
        elif b_platt <= min(b_raw, b_iso):
            best_method = "platt"
            p_calibrated = p_platt
        else:
            best_method = "none"
            p_calibrated = p_raw

        results = {
            "target_class": int(target_class),
            "target_class_name": cfg.VisualizationConfig.LABEL_NAMES.get(target_class, str(target_class)),
            "brier_raw": float(b_raw),
            "brier_platt": float(b_platt),
            "brier_isotonic": float(b_iso),
            "best_method": best_method,
        }

        np.savez(
            out_path,
            y_binary=ybin,
            p_raw=p_raw,
            p_platt=p_platt,
            p_isotonic=p_iso,
            p_calibrated=p_calibrated,
        )

        # Also save as JSON for easy reading
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved calibration results to: {out_path}")

    except Exception as e:
        logger.warning(f"Calibration failed: {e}")
        # Save basic results
        np.savez(
            out_path,
            y_binary=ybin,
            p_target=p_target,
        )


# ========== Main Execution ==========

def run_shap_analysis(
    force_recompute: bool = False,
    export_tables: bool = True,
    save_intermediate: bool = True
) -> Dict[str, any]:
    """
    Main function to run SHAP analysis and save results.

    Args:
        force_recompute: If True, recompute SHAP even if cached results exist
        export_tables: If True, export top feature tables
        save_intermediate: If True, save intermediate results (inference, attention, calibration)

    Returns:
        Dictionary containing all computed results and paths
    """
    # Setup
    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available() and cfg.DEVICE.startswith("cuda") else "cpu"
    )

    # Select Model-of-Record
    seed_star, fold_idx, fold_name = pick_model_of_record()
    logger.info(f"Using Model-of-Record: seed={seed_star}, fold={fold_name}")

    # Load model and data
    model, _ = load_best_model_for_fold(seed_star, fold_idx, device)
    loader, meta = get_val_loader_for_fold(seed_star, fold_idx, device)
    feature_names = get_feature_names(meta)

    # Check for cached SHAP results
    shap_cache_path = OUT_DIR / "mor_shap_outputs.npz"

    if shap_cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached SHAP values from {shap_cache_path}")
        dat = np.load(shap_cache_path, allow_pickle=True)

        phi_macro = dat["phi_macro"]
        phi_per_class_stack = dat["phi_stack"]
        y_true = dat["y_true"]
        X_total = dat["X_total"]

        # Verify feature names match
        saved_feature_names = dat["feature_names"].tolist()
        if saved_feature_names != feature_names:
            logger.warning("Feature names mismatch. Recomputing SHAP values.")
            force_recompute = True

    if force_recompute or not shap_cache_path.exists():
        logger.info("Computing SHAP values...")
        shap_pack, X_total = compute_shap_for_fold(
            model, loader, device,
            background_size=75,
            random_state=7
        )
        phi_macro = shap_pack["macro"]
        phi_per_class_stack = shap_pack["per_class_stack"]
        y_true = shap_pack["y_true"]

    # Feature annotation
    logger.info("Building feature annotations...")
    display_map, hover_map, annotated_map = build_feature_label_maps(feature_names)
    display_feature_names = [display_map.get(name, name) for name in feature_names]
    annotated_flags = np.array([annotated_map.get(name, True) for name in feature_names], dtype=bool)

    # Build allowed mask and deduplicate
    allowed_mask = build_allowed_feature_mask(feature_names, annotated_flags, display_feature_names)
    allowed_mask = deduplicate_by_base_metabolite(display_feature_names, allowed_mask, phi_macro)

    # Save main SHAP results
    logger.info("Saving SHAP results...")
    save_shap_results(
        phi_macro=phi_macro,
        phi_per_class_stack=phi_per_class_stack,
        y_true=y_true,
        X_total=X_total,
        feature_names=feature_names,
        display_feature_names=display_feature_names,
        annotated_flags=annotated_flags,
        allowed_mask=allowed_mask,
        out_path=shap_cache_path,
        additional_meta={
            "seed": seed_star,
            "fold_idx": fold_idx,
            "fold_name": fold_name,
        }
    )

    # Export top feature tables
    if export_tables:
        tables_dir = OUT_DIR / "tables"
        export_shap_top_feature_tables(
            phi_macro=phi_macro,
            phi_per_class_stack=phi_per_class_stack,
            feature_names=feature_names,
            display_feature_names=display_feature_names,
            annotated_flags=annotated_flags,
            allowed_mask=allowed_mask,
            out_dir=tables_dir,
            topk_global=200,
            topk_per_modality=120,
            topk_per_class_pos=100,
            topk_per_class_neg=100,
            y_true=y_true,
        )

    # Save intermediate results
    if save_intermediate:
        # Inference results
        logger.info("Running inference and saving results...")
        y_true_infer, y_prob_infer, ids_infer = infer_all(model, loader, device)
        save_inference_results(
            y_true_infer, y_prob_infer, ids_infer,
            OUT_DIR / "mor_inference_results.npz"
        )

        # Attention matrix
        logger.info("Computing attention matrix...")
        attention_matrix = aggregate_attention_matrix(model, loader, device)
        save_attention_matrix(
            attention_matrix,
            OUT_DIR / "mor_attention_matrix.npz"
        )

        # Calibration results
        logger.info("Computing calibration statistics...")
        save_calibration_results(
            y_true_infer, y_prob_infer,
            OUT_DIR / "mor_calibration_results.npz"
        )

        # OOF predictions summary
        oof_records = collect_oof_predictions()
        if oof_records:
            oof_summary = {
                "seeds": [r["seed"] for r in oof_records],
                "n_samples": [len(r["true"]) for r in oof_records],
            }
            with open(OUT_DIR / "oof_summary.json", "w") as f:
                json.dump(oof_summary, f, indent=2)

    # Return results dictionary
    results = {
        "shap_cache_path": str(shap_cache_path),
        "phi_macro": phi_macro,
        "phi_per_class_stack": phi_per_class_stack,
        "y_true": y_true,
        "X_total": X_total,
        "feature_names": feature_names,
        "display_feature_names": display_feature_names,
        "annotated_flags": annotated_flags,
        "allowed_mask": allowed_mask,
        "model_info": {
            "seed": seed_star,
            "fold_idx": fold_idx,
            "fold_name": fold_name,
        },
        "output_dir": str(OUT_DIR),
    }

    logger.info(f"SHAP analysis complete. Results saved to: {OUT_DIR}")
    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="SHAP analysis for SLEmodel. Computes and saves SHAP values."
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of SHAP values even if cached"
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip exporting top feature tables"
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Skip saving intermediate results (inference, attention, calibration)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Run analysis
    results = run_shap_analysis(
        force_recompute=args.force_recompute,
        export_tables=not args.no_tables,
        save_intermediate=not args.no_intermediate
    )

    print(f"\n[Done] SHAP analysis complete.")
    print(f"Results saved to: {results['output_dir']}")
    print(f"Main SHAP file: {results['shap_cache_path']}")


if __name__ == "__main__":
    main()
