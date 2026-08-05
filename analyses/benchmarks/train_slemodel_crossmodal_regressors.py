"""
Train cross-modal regressors on representations from a frozen SLEmodel.

For each seed × repeat × fold:
  1) Reproduce the cross-validation splits and PS-guided OT matching used for SLEmodel;
  2) Load the best SLEmodel for that (seed, fold), freeze all its parameters;
  3) Train lightweight regressors in latent space:

     - G2M : shared_gly -> pseudo_mass
     - G2R : shared_gly -> pseudo_rna
     - GM2R: [shared_gly, shared_mass] -> pseudo_rna

  4) Save:
     - regressor parameters to:
       crossmodal_regressor_<fold_name>.pth
     - validation pseudo data and gly features to:
       crossmodal_evaluation_data_<fold_name>.pt

  5) Compute final validation metrics with the best regressor and write all
     results to:
       experiments/<EXPERIMENT_TAG>/slemodel_crossmodal_regressor_metrics.csv
"""

import logging
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from slemodel import config as cfg
from slemodel.models import DecoupledModel, PropensityScoreClassifier
from slemodel.utils import (
    train_classifier,
    get_propensity_scores,
    get_propensity_scores_calibrated,
    get_ot_coupling_modality_specific,
)
SEEDS = [42, 100, 2025, 7, 123]
BASE_EXP_DIR = Path(cfg.BASE_EXPERIMENT_DIR) / cfg.EXPERIMENT_TAG
N_SPLITS = cfg.N_SPLITS_K_FOLD
N_REPEATS = cfg.N_REPEATS
EPOCHS_PS = cfg.EPOCHS_PS

REGRESSOR_MAX_EPOCHS = 80
REGRESSOR_EARLY_STOP_MIN_EPOCHS = 10
REGRESSOR_EARLY_STOP_PATIENCE = 15
REGRESSOR_HPARAMS_GRID = [
    {"lr": 1e-3,  "weight_decay": 1e-5, "batch_size": 64},
    {"lr": 5e-4,  "weight_decay": 1e-5, "batch_size": 64},
    {"lr": 1e-3,  "weight_decay": 1e-4, "batch_size": 64},
    {"lr": 5e-4,  "weight_decay": 1e-4, "batch_size": 128},
    {"lr": 2e-4,  "weight_decay": 1e-5, "batch_size": 64},
]

REGRESSOR_METRIC_CSV = BASE_EXP_DIR / "slemodel_crossmodal_regressor_metrics.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [crossmodal_regression] %(message)s",
)
logger = logging.getLogger("crossmodal_regression")


def load_data(path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values
    feats = df.iloc[:, 2:].values.astype(np.float32)
    return ids, feats, labels


def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Multi-output regression metrics:
      - R2 (uniform average)
      - RMSE
      - MAE
      - Pearson r on flattened vectors
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr

    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"

    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    try:
        pearson, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    except Exception:
        pearson = np.nan

    return {"r2": float(r2), "rmse": rmse, "mae": mae, "pearson": float(pearson)}


class CrossModalRegressor(nn.Module):
    """
    A slightly deeper MLP regressor on top of frozen encoders:

      - g2m : shared_gly -> pseudo_mass
      - g2r : shared_gly -> pseudo_rna
      - gm2r: concat(shared_gly, shared_mass) -> pseudo_rna
    """

    def __init__(self, dim_gly_shared: int, dim_mass_shared: int,
                 mass_dim: int, rna_dim: int,
                 hidden_dim_mass: int = 128,
                 hidden_dim_rna: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        self.g2m = nn.Sequential(
            nn.Linear(dim_gly_shared, hidden_dim_mass),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_mass, mass_dim),
        )

        self.g2r = nn.Sequential(
            nn.Linear(dim_gly_shared, hidden_dim_rna),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_rna, rna_dim),
        )

        # Gly + Mass -> RNA
        self.gm2r = nn.Sequential(
            nn.Linear(dim_gly_shared + dim_mass_shared, hidden_dim_rna),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_rna, rna_dim),
        )

    def forward_g2m(self, z_s_gly):
        return self.g2m(z_s_gly)

    def forward_g2r(self, z_s_gly):
        return self.g2r(z_s_gly)

    def forward_gm2r(self, z_s_gly, z_s_mass):
        z_cat = torch.cat([z_s_gly, z_s_mass], dim=1)
        return self.gm2r(z_cat)


def main():
    logger.info("=== Training Cross-Modal Regressors on Frozen SLEmodel Representations ===")
    logger.info(f"Using device: {cfg.DEVICE}")
    logger.info(f"Base experiment directory: {BASE_EXP_DIR}")

    BASE_EXP_DIR.mkdir(parents=True, exist_ok=True)

    if REGRESSOR_METRIC_CSV.exists():
        logger.warning(f"{REGRESSOR_METRIC_CSV} already exists and will be overwritten.")
        REGRESSOR_METRIC_CSV.unlink()

    all_records = []

    logger.info("--- Loading raw modality data ---")
    gly_ids, gly_x_raw, gly_y_str = load_data(cfg.GLY_PATH)
    mass_ids, mass_x_raw, mass_y_str = load_data(cfg.MASS_PATH)
    rna_ids, rna_x_raw, rna_y_str = load_data(cfg.RNA_PATH)

    label_map = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
    gly_y = pd.Series(gly_y_str).map(label_map).values
    mass_y = pd.Series(mass_y_str).map(label_map).values
    rna_y = pd.Series(rna_y_str).map(label_map).values

    for seed in SEEDS:
        logger.info("\n" + "=" * 70)
        logger.info(f"Processing SEED = {seed}")
        logger.info("=" * 70)

        run_name = f"SLEmodel_Run_Seed_{seed}"
        run_dir = BASE_EXP_DIR / run_name / f"master_seed_{seed}"
        if not run_dir.exists():
            logger.warning(f"Run directory not found: {run_dir}, skip this seed.")
            continue

        rskf_gly = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=seed,
        )
        rskf_mass = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=seed,
        )
        rskf_rna = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=seed,
        )

        gly_splits = list(rskf_gly.split(gly_x_raw, gly_y))
        mass_splits = list(rskf_mass.split(mass_x_raw, mass_y))
        rna_splits = list(rskf_rna.split(rna_x_raw, rna_y))

        for repeat in range(N_REPEATS):
            repeat_num = repeat + 1
            logger.info(f"  - Repeat {repeat_num}/{N_REPEATS}")

            for fold in range(N_SPLITS):
                fold_num = fold + 1
                global_fold_idx = repeat * N_SPLITS + fold
                fold_name = f"Repeat_{repeat_num}-Fold_{fold_num}"
                logger.info(f"\n  --- Training regressors for {fold_name} ---")

                fold_dir = run_dir / fold_name
                best_model_path = fold_dir / f"best_model_{fold_name}.pth"
                regressor_path = fold_dir / f"crossmodal_regressor_{fold_name}.pth"
                eval_data_path = fold_dir / f"crossmodal_evaluation_data_{fold_name}.pt"

                if not best_model_path.exists():
                    logger.warning(f"Best model not found: {best_model_path}, skip.")
                    continue

                if regressor_path.exists() and eval_data_path.exists():
                    logger.info(f"Regressor + eval data already exist for {fold_name}, skip training.")
                    continue

                gly_train_idx, gly_val_idx = gly_splits[global_fold_idx]
                mass_train_idx, _ = mass_splits[global_fold_idx]
                rna_train_idx, _ = rna_splits[global_fold_idx]

                gly_scaler = StandardScaler().fit(gly_x_raw[gly_train_idx])
                mass_scaler = StandardScaler().fit(mass_x_raw[mass_train_idx])
                rna_scaler = StandardScaler().fit(rna_x_raw[rna_train_idx])

                gly_x_train = gly_scaler.transform(gly_x_raw[gly_train_idx])
                gly_x_val = gly_scaler.transform(gly_x_raw[gly_val_idx])
                mass_x_train = mass_scaler.transform(mass_x_raw[mass_train_idx])
                rna_x_train = rna_scaler.transform(rna_x_raw[rna_train_idx])

                train_gly_x = torch.tensor(gly_x_train, dtype=torch.float32)
                val_gly_x = torch.tensor(gly_x_val, dtype=torch.float32)
                pool_mass_x = torch.tensor(mass_x_train, dtype=torch.float32)
                pool_rna_x = torch.tensor(rna_x_train, dtype=torch.float32)

                train_gly_y = torch.tensor(gly_y[gly_train_idx], dtype=torch.long)
                pool_mass_y = torch.tensor(mass_y[mass_train_idx], dtype=torch.long)
                pool_rna_y = torch.tensor(rna_y[rna_train_idx], dtype=torch.long)

                logger.info("    Training PS classifiers for this fold...")
                gly_clf = train_classifier(
                    PropensityScoreClassifier(train_gly_x.shape[1]),
                    train_gly_x,
                    train_gly_y,
                    cfg.DEVICE,
                    epochs=EPOCHS_PS,
                )
                mass_clf = train_classifier(
                    PropensityScoreClassifier(pool_mass_x.shape[1]),
                    pool_mass_x,
                    pool_mass_y,
                    cfg.DEVICE,
                    epochs=EPOCHS_PS,
                )
                rna_clf = train_classifier(
                    PropensityScoreClassifier(pool_rna_x.shape[1]),
                    pool_rna_x,
                    pool_rna_y,
                    cfg.DEVICE,
                    epochs=EPOCHS_PS,
                )

                logger.info("    Computing PS & OT couplings (train + val)...")

                if getattr(cfg, "USE_PS_TEMPERATURE_SCALING", False):
                    ps_train_gly, _ = get_propensity_scores_calibrated(
                        gly_clf, train_gly_x, train_gly_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                    ps_val_gly, _ = get_propensity_scores_calibrated(
                        gly_clf, val_gly_x, torch.tensor(gly_y[gly_val_idx], dtype=torch.long),
                        cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                    ps_pool_mass, _ = get_propensity_scores_calibrated(
                        mass_clf, pool_mass_x, pool_mass_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                    ps_pool_rna, _ = get_propensity_scores_calibrated(
                        rna_clf, pool_rna_x, pool_rna_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                else:
                    ps_train_gly = get_propensity_scores(gly_clf, train_gly_x, cfg.DEVICE)
                    ps_val_gly = get_propensity_scores(gly_clf, val_gly_x, cfg.DEVICE)
                    ps_pool_mass = get_propensity_scores(mass_clf, pool_mass_x, cfg.DEVICE)
                    ps_pool_rna = get_propensity_scores(rna_clf, pool_rna_x, cfg.DEVICE)

                train_mass_coupling = get_ot_coupling_modality_specific(
                    ps_train_gly, ps_pool_mass, modality="mass",
                    reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                    method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
                    cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                    cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
                    purpose="eval",
                )
                train_rna_coupling = get_ot_coupling_modality_specific(
                    ps_train_gly, ps_pool_rna, modality="rna",
                    reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                    method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
                    cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                    cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
                    purpose="eval",
                )
                val_mass_coupling = get_ot_coupling_modality_specific(
                    ps_val_gly, ps_pool_mass, modality="mass",
                    reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                    method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
                    cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                    cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
                    purpose="eval",
                )
                val_rna_coupling = get_ot_coupling_modality_specific(
                    ps_val_gly, ps_pool_rna, modality="rna",
                    reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                    method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
                    cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                    cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
                    purpose="eval",
                )

                pseudo_mass_train = train_mass_coupling @ pool_mass_x
                pseudo_rna_train = train_rna_coupling @ pool_rna_x
                pseudo_mass_val = val_mass_coupling @ pool_mass_x
                pseudo_rna_val = val_rna_coupling @ pool_rna_x

                logger.info(f"    Loading frozen SLEmodel from: {best_model_path}")
                model = DecoupledModel(
                    gly_x_raw.shape[1],
                    mass_x_raw.shape[1],
                    rna_x_raw.shape[1],
                ).to(cfg.DEVICE)

                state_dict = torch.load(best_model_path, map_location=cfg.DEVICE)
                model.load_state_dict(state_dict)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

                with torch.no_grad():
                    sample_gly = train_gly_x[: min(64, train_gly_x.shape[0])].to(cfg.DEVICE)
                    sample_pmass = pseudo_mass_train[: min(64, pseudo_mass_train.shape[0])].to(cfg.DEVICE)

                    _, z_s_gly, _ = model.reconstruct_single_modality(sample_gly, "gly")
                    _, z_s_mass, _ = model.reconstruct_single_modality(sample_pmass, "mass")

                    dim_gly_shared = z_s_gly.shape[1]
                    dim_mass_shared = z_s_mass.shape[1]

                mass_dim = pool_mass_x.shape[1]
                rna_dim = pool_rna_x.shape[1]

                logger.info(
                    f"    Shared dims: z_gly={dim_gly_shared}, z_mass={dim_mass_shared}; "
                    f"Targets: mass_dim={mass_dim}, rna_dim={rna_dim}"
                )

                train_dataset = TensorDataset(train_gly_x, pseudo_mass_train, pseudo_rna_train)
                val_dataset = TensorDataset(val_gly_x, pseudo_mass_val, pseudo_rna_val)

                logger.info("    Start hyper-parameter search for cross-modal regressors...")
                best_overall_score = -np.inf
                best_overall_state = None
                best_overall_hparams = None

                for cfg_idx, hparams in enumerate(REGRESSOR_HPARAMS_GRID, 1):
                    lr = hparams["lr"]
                    weight_decay = hparams["weight_decay"]
                    batch_size = hparams["batch_size"]

                    logger.info(
                        f"      > Config {cfg_idx}/{len(REGRESSOR_HPARAMS_GRID)}: "
                        f"lr={lr:.1e}, weight_decay={weight_decay:.1e}, batch_size={batch_size}"
                    )

                    regressor = CrossModalRegressor(
                        dim_gly_shared=dim_gly_shared,
                        dim_mass_shared=dim_mass_shared,
                        mass_dim=mass_dim,
                        rna_dim=rna_dim,
                    ).to(cfg.DEVICE)

                    optimizer = torch.optim.Adam(
                        regressor.parameters(),
                        lr=lr,
                        weight_decay=weight_decay,
                    )
                    criterion = nn.MSELoss()

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=max(batch_size, 128), shuffle=False)

                    best_val_score_cfg = -np.inf
                    best_state_cfg = None
                    patience_counter = 0

                    for epoch in range(REGRESSOR_MAX_EPOCHS):
                        regressor.train()
                        total_loss = 0.0

                        for gly_b, pmass_b, prna_b in train_loader:
                            gly_b = gly_b.to(cfg.DEVICE)
                            pmass_b = pmass_b.to(cfg.DEVICE)
                            prna_b = prna_b.to(cfg.DEVICE)

                            with torch.no_grad():
                                _, z_s_g, _ = model.reconstruct_single_modality(gly_b, "gly")
                                _, z_s_m, _ = model.reconstruct_single_modality(pmass_b, "mass")

                            pred_mass = regressor.forward_g2m(z_s_g)
                            pred_rna_g = regressor.forward_g2r(z_s_g)
                            pred_rna_gm = regressor.forward_gm2r(z_s_g, z_s_m)

                            loss = (
                                criterion(pred_mass, pmass_b)
                                + criterion(pred_rna_g, prna_b)
                                + criterion(pred_rna_gm, prna_b)
                            )

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()

                        avg_train_loss = total_loss / max(1, len(train_loader))

                        regressor.eval()
                        all_true_mass, all_pred_mass = [], []
                        all_true_rna_g, all_pred_rna_g = [], []
                        all_true_rna_gm, all_pred_rna_gm = [], []

                        with torch.no_grad():
                            for gly_b, pmass_b, prna_b in val_loader:
                                gly_b = gly_b.to(cfg.DEVICE)
                                pmass_b = pmass_b.to(cfg.DEVICE)
                                prna_b = prna_b.to(cfg.DEVICE)

                                _, z_s_g, _ = model.reconstruct_single_modality(gly_b, "gly")
                                _, z_s_m, _ = model.reconstruct_single_modality(pmass_b, "mass")

                                pred_mass = regressor.forward_g2m(z_s_g)
                                pred_rna_g = regressor.forward_g2r(z_s_g)
                                pred_rna_gm = regressor.forward_gm2r(z_s_g, z_s_m)

                                all_true_mass.append(pmass_b.cpu().numpy())
                                all_pred_mass.append(pred_mass.cpu().numpy())
                                all_true_rna_g.append(prna_b.cpu().numpy())
                                all_pred_rna_g.append(pred_rna_g.cpu().numpy())
                                all_true_rna_gm.append(prna_b.cpu().numpy())
                                all_pred_rna_gm.append(pred_rna_gm.cpu().numpy())

                        if len(all_true_mass) == 0:
                            logger.warning("    Empty validation loader, skip metrics.")
                            break

                        true_mass = np.concatenate(all_true_mass, axis=0)
                        pred_mass = np.concatenate(all_pred_mass, axis=0)
                        true_rna_g = np.concatenate(all_true_rna_g, axis=0)
                        pred_rna_g = np.concatenate(all_pred_rna_g, axis=0)
                        true_rna_gm = np.concatenate(all_true_rna_gm, axis=0)
                        pred_rna_gm = np.concatenate(all_pred_rna_gm, axis=0)

                        m_g2m = compute_reg_metrics(true_mass, pred_mass)
                        m_g2r = compute_reg_metrics(true_rna_g, pred_rna_g)
                        m_gm2r = compute_reg_metrics(true_rna_gm, pred_rna_gm)

                        val_score = float((m_g2m["r2"] + m_g2r["r2"] + m_gm2r["r2"]) / 3.0)

                        logger.info(
                            f"        [Cfg {cfg_idx}] Epoch {epoch+1:03d} | "
                            f"TrainLoss={avg_train_loss:.4f} | "
                            f"ValR2: G2M={m_g2m['r2']:.3f}, "
                            f"G2R={m_g2r['r2']:.3f}, GM2R={m_gm2r['r2']:.3f}, "
                            f"AvgR2={val_score:.3f}"
                        )

                        if epoch + 1 < REGRESSOR_EARLY_STOP_MIN_EPOCHS:
                            if val_score > best_val_score_cfg:
                                best_val_score_cfg = val_score
                                best_state_cfg = copy.deepcopy(regressor.state_dict())
                            continue

                        if val_score > best_val_score_cfg:
                            best_val_score_cfg = val_score
                            best_state_cfg = copy.deepcopy(regressor.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= REGRESSOR_EARLY_STOP_PATIENCE:
                            logger.info(
                                f"        [Cfg {cfg_idx}] Early stopping at epoch {epoch+1}, "
                                f"best AvgR2={best_val_score_cfg:.3f}"
                            )
                            break

                    if best_state_cfg is None:
                        best_state_cfg = copy.deepcopy(regressor.state_dict())

                    logger.info(f"      > Config {cfg_idx} finished. Best AvgR2={best_val_score_cfg:.3f}")

                    if best_val_score_cfg > best_overall_score:
                        best_overall_score = best_val_score_cfg
                        best_overall_state = best_state_cfg
                        best_overall_hparams = hparams

                if best_overall_state is None:
                    logger.warning("    No best state recorded for any config; saving last regressor parameters.")
                    best_overall_state = best_state_cfg
                    if best_overall_hparams is None and REGRESSOR_HPARAMS_GRID:
                        best_overall_hparams = REGRESSOR_HPARAMS_GRID[-1]

                logger.info(
                    f"    Best hyper-params for {fold_name}: "
                    f"lr={best_overall_hparams['lr']:.1e}, "
                    f"weight_decay={best_overall_hparams['weight_decay']:.1e}, "
                    f"batch_size={best_overall_hparams['batch_size']}"
                )

                regressor_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_overall_state, regressor_path)
                logger.info(f"    Saved best cross-modal regressor to: {regressor_path}")

                torch.save(
                    {
                        "val_gly_x": val_gly_x,
                        "pseudo_mass_val": pseudo_mass_val,
                        "pseudo_rna_val": pseudo_rna_val,
                    },
                    eval_data_path,
                )
                logger.info(f"    Saved validation pseudo data to: {eval_data_path}")

                regressor = CrossModalRegressor(
                    dim_gly_shared=dim_gly_shared,
                    dim_mass_shared=dim_mass_shared,
                    mass_dim=mass_dim,
                    rna_dim=rna_dim,
                ).to(cfg.DEVICE)
                regressor.load_state_dict(best_overall_state)
                regressor.eval()

                val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

                all_true_mass, all_pred_mass = [], []
                all_true_rna_g, all_pred_rna_g = [], []
                all_true_rna_gm, all_pred_rna_gm = [], []

                with torch.no_grad():
                    for gly_b, pmass_b, prna_b in val_loader:
                        gly_b = gly_b.to(cfg.DEVICE)
                        pmass_b = pmass_b.to(cfg.DEVICE)
                        prna_b = prna_b.to(cfg.DEVICE)

                        _, z_s_g, _ = model.reconstruct_single_modality(gly_b, "gly")
                        _, z_s_m, _ = model.reconstruct_single_modality(pmass_b, "mass")

                        pred_mass = regressor.forward_g2m(z_s_g)
                        pred_rna_g = regressor.forward_g2r(z_s_g)
                        pred_rna_gm = regressor.forward_gm2r(z_s_g, z_s_m)

                        all_true_mass.append(pmass_b.cpu().numpy())
                        all_pred_mass.append(pred_mass.cpu().numpy())
                        all_true_rna_g.append(prna_b.cpu().numpy())
                        all_pred_rna_g.append(pred_rna_g.cpu().numpy())
                        all_true_rna_gm.append(prna_b.cpu().numpy())
                        all_pred_rna_gm.append(pred_rna_gm.cpu().numpy())

                true_mass = np.concatenate(all_true_mass, axis=0)
                pred_mass = np.concatenate(all_pred_mass, axis=0)
                true_rna_g = np.concatenate(all_true_rna_g, axis=0)
                pred_rna_g = np.concatenate(all_pred_rna_g, axis=0)
                true_rna_gm = np.concatenate(all_true_rna_gm, axis=0)
                pred_rna_gm = np.concatenate(all_pred_rna_gm, axis=0)

                m_g2m = compute_reg_metrics(true_mass, pred_mass)
                m_g2r = compute_reg_metrics(true_rna_g, pred_rna_g)
                m_gm2r = compute_reg_metrics(true_rna_gm, pred_rna_gm)

                logger.info(
                    f"    [Final Val Metrics] {fold_name} | "
                    f"G2M_R2={m_g2m['r2']:.3f}, "
                    f"G2R_R2={m_g2r['r2']:.3f}, "
                    f"GM2R_R2={m_gm2r['r2']:.3f}"
                )

                for task_name, metrics in [("G2M", m_g2m), ("G2R", m_g2r), ("GM2R", m_gm2r)]:
                    rec = {
                        "model": "SLEmodel_Regressor",
                        "task": task_name,
                        "seed": seed,
                        "repeat": repeat_num,
                        "fold": fold_num,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "pearson": metrics["pearson"],
                        "n_val_samples": int(true_mass.shape[0]),
                        "mass_dim": int(mass_x_raw.shape[1]),
                        "rna_dim": int(rna_x_raw.shape[1]),
                        "regressor_lr": float(best_overall_hparams["lr"]),
                        "regressor_weight_decay": float(best_overall_hparams["weight_decay"]),
                        "regressor_batch_size": int(best_overall_hparams["batch_size"]),
                    }
                    all_records.append(rec)

    if not all_records:
        logger.warning("No cross-modal regression metrics were collected.")
    else:
        df = pd.DataFrame(all_records)
        df.to_csv(REGRESSOR_METRIC_CSV, index=False)
        logger.info("\n=== Cross-modal regressor validation metrics saved to ===")
        logger.info(f"  {REGRESSOR_METRIC_CSV.resolve()}")


if __name__ == "__main__":
    main()
