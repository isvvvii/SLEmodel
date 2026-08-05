"""
Evaluate cross-modal reconstruction of SLEmodel using the dedicated latent-space regressor,
on the same validation pseudo targets used to train the cross-modal regressors.

Assumes that for each (seed, repeat, fold), we have:
  - best_model_<fold>.pth
  - crossmodal_regressor_<fold>.pth
  - crossmodal_evaluation_data_<fold>.pt
    (containing val_gly_x, pseudo_mass_val, pseudo_rna_val)

Evaluated tasks:
  - G2M : gly-only -> pseudo_mass
  - G2R : gly-only -> pseudo_rna
  - GM2R: gly + pseudo_mass -> pseudo_rna

Results are saved to:
  experiments/<EXPERIMENT_TAG>/slemodel_crossmodal_reconstruction_metrics.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from slemodel import config as cfg
from slemodel.models import DecoupledModel

SEEDS = [42, 100, 2025, 7, 123]
BASE_EXP_DIR = Path(cfg.BASE_EXPERIMENT_DIR) / cfg.EXPERIMENT_TAG
OUTPUT_CSV = BASE_EXP_DIR / "slemodel_crossmodal_reconstruction_metrics.csv"
N_SPLITS = cfg.N_SPLITS_K_FOLD
N_REPEATS = cfg.N_REPEATS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [crossmodal_evaluation] %(message)s",
)
logger = logging.getLogger("crossmodal_evaluation")


def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
    logger.info("=== Evaluating SLEmodel Cross-Modal Reconstruction ===")
    logger.info(f"Using device: {cfg.DEVICE}")
    logger.info(f"Experiment directory: {BASE_EXP_DIR}")

    BASE_EXP_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists():
        logger.warning(f"Output CSV already exists; it will be overwritten: {OUTPUT_CSV}")
        OUTPUT_CSV.unlink()

    all_records = []

    mass_dim = pd.read_csv(cfg.MASS_PATH).shape[1] - 2
    rna_dim = pd.read_csv(cfg.RNA_PATH).shape[1] - 2
    gly_dim = pd.read_csv(cfg.GLY_PATH).shape[1] - 2

    for seed in SEEDS:
        logger.info("\n" + "=" * 70)
        logger.info(f"Evaluating SEED = {seed}")
        logger.info("=" * 70)

        run_name = f"SLEmodel_Run_Seed_{seed}"
        run_dir = BASE_EXP_DIR / run_name / f"master_seed_{seed}"
        if not run_dir.exists():
            logger.warning(f"Run directory not found: {run_dir}, skip this seed.")
            continue

        for repeat in range(N_REPEATS):
            repeat_num = repeat + 1
            logger.info(f"  - Repeat {repeat_num}/{N_REPEATS}")

            for fold_num in range(1, N_SPLITS + 1):
                fold_name = f"Repeat_{repeat_num}-Fold_{fold_num}"
                logger.info(f"\n  --- Seed {seed}, {fold_name} ---")

                fold_dir = run_dir / fold_name
                best_model_path = fold_dir / f"best_model_{fold_name}.pth"
                regressor_path = fold_dir / f"crossmodal_regressor_{fold_name}.pth"
                eval_data_path = fold_dir / f"crossmodal_evaluation_data_{fold_name}.pt"

                if not best_model_path.exists():
                    logger.warning(f"  Best model not found: {best_model_path}, skip.")
                    continue
                if not regressor_path.exists():
                    logger.warning(f"  Cross-modal regressor not found: {regressor_path}, skip.")
                    continue
                if not eval_data_path.exists():
                    logger.warning(f"  Eval data not found: {eval_data_path}, skip.")
                    continue

                eval_data = torch.load(eval_data_path, map_location="cpu")
                val_gly_x = eval_data["val_gly_x"].float()
                pseudo_mass_val = eval_data["pseudo_mass_val"].float()
                pseudo_rna_val = eval_data["pseudo_rna_val"].float()

                val_dataset = TensorDataset(val_gly_x, pseudo_mass_val, pseudo_rna_val)
                val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

                model = DecoupledModel(
                    gly_dim,
                    mass_dim,
                    rna_dim,
                ).to(cfg.DEVICE)
                state_dict = torch.load(best_model_path, map_location=cfg.DEVICE)
                model.load_state_dict(state_dict)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

                with torch.no_grad():
                    sample_gly = val_gly_x[: min(64, val_gly_x.shape[0])].to(cfg.DEVICE)
                    sample_pmass = pseudo_mass_val[: min(64, pseudo_mass_val.shape[0])].to(cfg.DEVICE)

                    _, z_s_gly, _ = model.reconstruct_single_modality(sample_gly, "gly")
                    _, z_s_mass, _ = model.reconstruct_single_modality(sample_pmass, "mass")

                    dim_gly_shared = z_s_gly.shape[1]
                    dim_mass_shared = z_s_mass.shape[1]

                regressor = CrossModalRegressor(
                    dim_gly_shared=dim_gly_shared,
                    dim_mass_shared=dim_mass_shared,
                    mass_dim=mass_dim,
                    rna_dim=rna_dim,
                ).to(cfg.DEVICE)
                regressor_state = torch.load(regressor_path, map_location=cfg.DEVICE)
                regressor.load_state_dict(regressor_state)
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
                    logger.warning("    Empty validation loader; skip metrics for this fold.")
                    continue

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
                    f"    G2M  | R2={m_g2m['r2']:.3f}, RMSE={m_g2m['rmse']:.3f}, "
                    f"MAE={m_g2m['mae']:.3f}, r={m_g2m['pearson']:.3f}"
                )
                logger.info(
                    f"    G2R  | R2={m_g2r['r2']:.3f}, RMSE={m_g2r['rmse']:.3f}, "
                    f"MAE={m_g2r['mae']:.3f}, r={m_g2r['pearson']:.3f}"
                )
                logger.info(
                    f"    GM2R | R2={m_gm2r['r2']:.3f}, RMSE={m_gm2r['rmse']:.3f}, "
                    f"MAE={m_gm2r['mae']:.3f}, r={m_gm2r['pearson']:.3f}"
                )

                for task_name, metrics in [("G2M", m_g2m), ("G2R", m_g2r), ("GM2R", m_gm2r)]:
                    rec = {
                        "model": "SLEmodel",
                        "task": task_name,
                        "seed": seed,
                        "repeat": repeat_num,
                        "fold": fold_num,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "pearson": metrics["pearson"],
                        "n_val_samples": int(true_mass.shape[0]),
                        "mass_dim": int(mass_dim),
                        "rna_dim": int(rna_dim),
                    }
                    all_records.append(rec)

    if not all_records:
        logger.warning("No cross-modal metrics were computed. Check regressors & eval data.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("\n=== Cross-modal reconstruction metrics saved to ===")
    logger.info(f"  {OUTPUT_CSV.resolve()}")

    metrics_cols = ["r2", "rmse", "mae", "pearson"]
    needed_cols = {"task", *metrics_cols}
    if not needed_cols.issubset(df.columns):
        logger.warning(
            "Cannot compute summary table: missing columns in cross-modal metrics "
            f"(expected at least: {needed_cols})."
        )
        logger.info("Done.")
        return

    grouped = df.groupby("task")[metrics_cols].agg(["mean", "std"])

    summary_rows = []
    for task_name, row in grouped.iterrows():

        def fmt(metric: str) -> str:
            m = row[(metric, "mean")]
            s = row[(metric, "std")]
            if pd.isna(m) or pd.isna(s):
                return ""
            return f"{m:.3f} ± {s:.3f}"

        summary_rows.append(
            {
                "Task": task_name,
                "Model": "SLEmodel_Regressor",
                "R2": fmt("r2"),
                "RMSE": fmt("rmse"),
                "MAE": fmt("mae"),
                "Pearson r": fmt("pearson"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    def parse_mean(s: str) -> float:
        try:
            return float(str(s).split("±")[0])
        except Exception:
            return -np.inf

    summary_df["__r2_mean"] = summary_df["R2"].apply(parse_mean)
    summary_df = summary_df.sort_values("__r2_mean", ascending=False).drop(columns="__r2_mean")

    summary_path = OUTPUT_CSV.with_name("slemodel_crossmodal_reconstruction_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    logger.info(
        "\n=== Cross-modal reconstruction summary "
        "(mean ± std over seeds / repeats / folds) ==="
    )
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info(f"\nSummary table saved to: {summary_path.resolve()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
