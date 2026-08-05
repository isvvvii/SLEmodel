# ml_crossmodal_reconstruction.py
"""Compare traditional regressors for cross-modal reconstruction.

Tasks are glycomics to pseudo-mass spectrometry, glycomics to pseudo-RNA, and
glycomics plus pseudo-mass spectrometry to pseudo-RNA.
"""

import time
import json
import warnings
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.stats as st
import torch

from sklearn import model_selection, pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer

# PyTorch for Propensity Score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from slemodel import config as cfg
from slemodel.models import PropensityScoreClassifier as PSNet
from slemodel.utils import (
    train_classifier,
    get_propensity_scores,
    get_propensity_scores_calibrated,
    get_ot_coupling_modality_specific,
)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)
SEEDS = [42, 100, 2025, 7, 123]
OUTER_CV_SPLITS = cfg.N_SPLITS_K_FOLD
INNER_CV_SPLITS = 2
N_ITER_SEARCH = 5
PS_EPOCHS = cfg.EPOCHS_PS

DATA_PATHS = {"gly": cfg.GLY_PATH, "mass": cfg.MASS_PATH, "rna": cfg.RNA_PATH}
LABEL_MAP = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
DEVICE = cfg.DEVICE

RESULT_DIR = pathlib.Path("ml_results_crossmodal_reconstruction")
RESULT_DIR.mkdir(exist_ok=True, parents=True)

CROSSMODAL_CSV = RESULT_DIR / "crossmodal_reconstruction_metrics_all.csv"
def load_modality(path):
   df = pd.read_csv(path)
   y_str = df.iloc[:, 1].values
   X = df.iloc[:, 2:].values.astype(np.float32)
   y_enc = pd.Series(y_str).map(LABEL_MAP).values
   return X, y_enc


def get_reg_metrics_full(y_true, y_pred):
   """
   Evaluate the complete feature vectors without masking:
   - y_true, y_pred: [N, D], multi-output regression
   """
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   mae = mean_absolute_error(y_true, y_pred)
   r2 = r2_score(y_true, y_pred)
   try:
       pearson = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
   except Exception:
       pearson = np.nan

   return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson}


def append_reg_record(record: dict) -> None:
   df = pd.DataFrame([record])
   file_exists = CROSSMODAL_CSV.exists()
   df.to_csv(CROSSMODAL_CSV, mode="a", header=not file_exists, index=False)


def clean_params_for_json(params: dict):
   cleaned = {}
   for k, v in params.items():
       if hasattr(v, "item"):
           cleaned[k] = v.item()
       elif isinstance(v, np.generic):
           cleaned[k] = v.item()
       else:
           cleaned[k] = v
   return cleaned


def _to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)

def _to_long(y: np.ndarray) -> torch.Tensor:
    return torch.tensor(y, dtype=torch.long)

def _get_ps(clf, x_t: torch.Tensor, y_t: torch.Tensor | None) -> torch.Tensor:
    if getattr(cfg, "USE_PS_TEMPERATURE_SCALING", False) and y_t is not None:
        ps, _ = get_propensity_scores_calibrated(
            clf, x_t, y_t, DEVICE,
            calibrate=True,
            max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
            lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
        )
        return ps
    return get_propensity_scores(clf, x_t, DEVICE)

def generate_pseudo_pairs(
    gly_x_train: np.ndarray,
    gly_x_valid: np.ndarray,
    mass_pool_x: np.ndarray,
    rna_pool_x: np.ndarray,
    gly_y_train: np.ndarray,
    gly_y_valid: np.ndarray,
    mass_y_pool: np.ndarray,
    rna_y_pool: np.ndarray,
):
    """
    Match the SLEmodel evaluation design: propensity-score models are fitted
    only on the training anchor and donor pools, and modality-specific OT
    settings are used for barycentric mapping.
    """
    gly_train_t = _to_torch(gly_x_train)
    gly_valid_t = _to_torch(gly_x_valid)
    mass_pool_t = _to_torch(mass_pool_x)
    rna_pool_t = _to_torch(rna_pool_x)

    gly_y_train_t = _to_long(gly_y_train)
    gly_y_valid_t = _to_long(gly_y_valid)
    mass_y_t = _to_long(mass_y_pool)
    rna_y_t = _to_long(rna_y_pool)

    gly_clf = train_classifier(PSNet(gly_train_t.shape[1]), gly_train_t, gly_y_train_t, DEVICE, epochs=PS_EPOCHS)
    mass_clf = train_classifier(PSNet(mass_pool_t.shape[1]), mass_pool_t, mass_y_t, DEVICE, epochs=PS_EPOCHS)
    rna_clf = train_classifier(PSNet(rna_pool_t.shape[1]), rna_pool_t, rna_y_t, DEVICE, epochs=PS_EPOCHS)

    ps_gly_train = _get_ps(gly_clf, gly_train_t, gly_y_train_t)
    ps_gly_valid = _get_ps(gly_clf, gly_valid_t, gly_y_valid_t)
    ps_mass_pool = _get_ps(mass_clf, mass_pool_t, mass_y_t)
    ps_rna_pool = _get_ps(rna_clf, rna_pool_t, rna_y_t)

    Gm_train = get_ot_coupling_modality_specific(
        ps_gly_train, ps_mass_pool, modality="mass",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )
    Gm_valid = get_ot_coupling_modality_specific(ps_gly_valid, ps_mass_pool, modality="mass",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )
    Gr_train = get_ot_coupling_modality_specific(
        ps_gly_train, ps_rna_pool, modality="rna",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )
    Gr_valid = get_ot_coupling_modality_specific(ps_gly_valid, ps_rna_pool, modality="rna",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )

    pseudo_mass_train = (Gm_train @ mass_pool_t).numpy().astype(np.float32)
    pseudo_mass_valid = (Gm_valid @ mass_pool_t).numpy().astype(np.float32)
    pseudo_rna_train = (Gr_train @ rna_pool_t).numpy().astype(np.float32)
    pseudo_rna_valid = (Gr_valid @ rna_pool_t).numpy().astype(np.float32)

    return pseudo_mass_train, pseudo_mass_valid, pseudo_rna_train, pseudo_rna_valid
def get_reg_models_and_params(seed):
    models = OrderedDict()

    plsr_pipe = pipeline.make_pipeline(
        SimpleImputer(strategy="mean"),
        PLSRegression(scale=True),
    )
    models["PLSR"] = (
        plsr_pipe,
        {"plsregression__n_components": st.randint(2, 15)},
    )

    rf_pipe = pipeline.make_pipeline(
        SimpleImputer(strategy="mean"),
        RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    models["RF_Regr"] = (
        rf_pipe,
        {
            "randomforestregressor__n_estimators": st.randint(100, 301),
            "randomforestregressor__max_depth": st.randint(5, 25),
            "randomforestregressor__min_samples_leaf": st.randint(1, 6),
        },
    )

    return models
def main():
   print(
       "--- Starting Traditional ML Cross-Modal Reconstruction Experiments "
       "(Gly→Mass / Gly→RNA / Gly+Mass→RNA) ---"
   )
   print(f"Using device: {DEVICE}")

   print("Step 1: Loading all modality data...")
   feats, labels = {}, {}
   for m, p in DATA_PATHS.items():
       feats[m], labels[m] = load_modality(p)

   all_reg_records = []
   done_reg = set()

   if CROSSMODAL_CSV.exists():
       print(
           f"[Info] {CROSSMODAL_CSV} already exists. "
           "This run will SKIP any (seed, task, model, fold) that are already in this file. "
           "Delete this cache before a complete rerun."
       )
       try:
           existing_reg_df = pd.read_csv(CROSSMODAL_CSV)
           needed = {"seed", "task", "model", "fold"}
           if not existing_reg_df.empty and needed.issubset(existing_reg_df.columns):
               for _, row in existing_reg_df[list(needed)].dropna().iterrows():
                   try:
                       done_reg.add(
                           (
                               int(row["seed"]),
                               str(row["task"]),
                               str(row["model"]),
                               int(row["fold"]),
                           )
                       )
                   except Exception:
                       continue
           print(f"[Info] Loaded {len(done_reg)} completed cross-modal keys.")
       except Exception as e:
           print(f"[Warning] Failed to read existing cross-modal CSV: {e}")
   # Loop over seeds
   for seed in SEEDS:
       print(
           f"\n{'=' * 20} Running CROSS-MODAL RECONSTRUCTION with SEED: {seed} {'=' * 20}"
       )

       X_gly, y_gly = feats["gly"], labels["gly"]
       X_mass, y_mass = feats["mass"], labels["mass"]
       X_rna, y_rna = feats["rna"], labels["rna"]

       outer_cv_gly = model_selection.StratifiedKFold(
           n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
       )
       outer_cv_mass = model_selection.StratifiedKFold(
           n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
       )
       outer_cv_rna = model_selection.StratifiedKFold(
           n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
       )
       mass_splits = list(outer_cv_mass.split(X_mass, y_mass))
       rna_splits = list(outer_cv_rna.split(X_rna, y_rna))
       for fold_idx, (train_idx, valid_idx) in enumerate(
           outer_cv_gly.split(X_gly, y_gly), 1
       ):
           print(f"\n  --- [Seed {seed}] Fold {fold_idx}/{OUTER_CV_SPLITS} ---")

           X_gly_train_raw, X_gly_valid_raw = X_gly[train_idx], X_gly[valid_idx]
           y_gly_train = y_gly[train_idx]

           mass_train_idx, _ = mass_splits[fold_idx - 1]
           rna_train_idx, _ = rna_splits[fold_idx - 1]

           X_mass_pool_raw, y_mass_pool = (
               X_mass[mass_train_idx],
               y_mass[mass_train_idx],
           )
           X_rna_pool_raw, y_rna_pool = (
               X_rna[rna_train_idx],
               y_rna[rna_train_idx],
           )
           gly_scaler = StandardScaler().fit(X_gly_train_raw)
           X_gly_train = gly_scaler.transform(X_gly_train_raw)
           X_gly_valid = gly_scaler.transform(X_gly_valid_raw)

           mass_scaler = StandardScaler().fit(X_mass_pool_raw)
           X_mass_pool = mass_scaler.transform(X_mass_pool_raw)

           rna_scaler = StandardScaler().fit(X_rna_pool_raw)
           X_rna_pool = rna_scaler.transform(X_rna_pool_raw)

           pseudo_mass_train, pseudo_mass_valid, pseudo_rna_train, pseudo_rna_valid = generate_pseudo_pairs(
                X_gly_train,
                X_gly_valid,
                X_mass_pool,
                X_rna_pool,
                y_gly_train,
                y_gly[valid_idx],
                y_mass_pool,
                y_rna_pool,
            )

           models_to_run = get_reg_models_and_params(seed)
           # Task 1: Gly → pseudo-Mass
           task_name = "Gly_to_Mass"
           X_train_task1 = X_gly_train
           Y_train_task1 = pseudo_mass_train
           X_valid_task1 = X_gly_valid
           Y_valid_task1 = pseudo_mass_valid

           for model_name, (model, params) in models_to_run.items():
               key = (seed, task_name, model_name, fold_idx)
               if key in done_reg:
                   print(
                       f"    - Skipping {task_name} / {model_name}, "
                       f"seed={seed}, fold={fold_idx} (already done)."
                   )
                   continue

               try:
                   start_time = time.time()
                   search = model_selection.RandomizedSearchCV(
                       model,
                       params,
                       n_iter=N_ITER_SEARCH,
                       cv=INNER_CV_SPLITS,
                       scoring="neg_mean_squared_error",
                       n_jobs=-1,
                       random_state=seed,
                   )
                   search.fit(X_train_task1, Y_train_task1)
                   Y_pred = search.predict(X_valid_task1)
                   runtime = time.time() - start_time

                   metrics = get_reg_metrics_full(Y_valid_task1, Y_pred)
                   metrics.update(
                       {
                           "task": task_name,
                           "model": model_name,
                           "seed": seed,
                           "fold": fold_idx,
                           "best_params": json.dumps(
                               clean_params_for_json(search.best_params_)
                           ),
                           "runtime": runtime,
                       }
                   )

                   all_reg_records.append(metrics)
                   append_reg_record(metrics)
                   done_reg.add(key)

                   print(
                       f"    - {task_name} / {model_name} done. "
                       f"R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}"
                   )

               except Exception as e:
                   warnings.warn(
                       f"[Error] Task {task_name}, Model '{model_name}' "
                       f"on seed={seed}, fold={fold_idx} failed: {e}"
                   )
                   continue
           # Task 2: Gly → pseudo-RNA
           task_name = "Gly_to_RNA"
           X_train_task2 = X_gly_train
           Y_train_task2 = pseudo_rna_train
           X_valid_task2 = X_gly_valid
           Y_valid_task2 = pseudo_rna_valid

           for model_name, (model, params) in models_to_run.items():
               key = (seed, task_name, model_name, fold_idx)
               if key in done_reg:
                   print(
                       f"    - Skipping {task_name} / {model_name}, "
                       f"seed={seed}, fold={fold_idx} (already done)."
                   )
                   continue

               try:
                   start_time = time.time()
                   search = model_selection.RandomizedSearchCV(
                       model,
                       params,
                       n_iter=N_ITER_SEARCH,
                       cv=INNER_CV_SPLITS,
                       scoring="neg_mean_squared_error",
                       n_jobs=-1,
                       random_state=seed,
                   )
                   search.fit(X_train_task2, Y_train_task2)
                   Y_pred = search.predict(X_valid_task2)
                   runtime = time.time() - start_time

                   metrics = get_reg_metrics_full(Y_valid_task2, Y_pred)
                   metrics.update(
                       {
                           "task": task_name,
                           "model": model_name,
                           "seed": seed,
                           "fold": fold_idx,
                           "best_params": json.dumps(
                               clean_params_for_json(search.best_params_)
                           ),
                           "runtime": runtime,
                       }
                   )

                   all_reg_records.append(metrics)
                   append_reg_record(metrics)
                   done_reg.add(key)

                   print(
                       f"    - {task_name} / {model_name} done. "
                       f"R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}"
                   )

               except Exception as e:
                   warnings.warn(
                       f"[Error] Task {task_name}, Model '{model_name}' "
                       f"on seed={seed}, fold={fold_idx} failed: {e}"
                   )
                   continue
           # Task 3: Gly + pseudo-Mass → pseudo-RNA
           task_name = "GlyMass_to_RNA"
           X_train_task3 = np.concatenate([X_gly_train, pseudo_mass_train], axis=1)
           X_valid_task3 = np.concatenate([X_gly_valid, pseudo_mass_valid], axis=1)
           Y_train_task3 = pseudo_rna_train
           Y_valid_task3 = pseudo_rna_valid

           for model_name, (model, params) in models_to_run.items():
               key = (seed, task_name, model_name, fold_idx)
               if key in done_reg:
                   print(
                       f"    - Skipping {task_name} / {model_name}, "
                       f"seed={seed}, fold={fold_idx} (already done)."
                   )
                   continue

               try:
                   start_time = time.time()
                   search = model_selection.RandomizedSearchCV(
                       model,
                       params,
                       n_iter=N_ITER_SEARCH,
                       cv=INNER_CV_SPLITS,
                       scoring="neg_mean_squared_error",
                       n_jobs=-1,
                       random_state=seed,
                   )
                   search.fit(X_train_task3, Y_train_task3)
                   Y_pred = search.predict(X_valid_task3)
                   runtime = time.time() - start_time

                   metrics = get_reg_metrics_full(Y_valid_task3, Y_pred)
                   metrics.update(
                       {
                           "task": task_name,
                           "model": model_name,
                           "seed": seed,
                           "fold": fold_idx,
                           "best_params": json.dumps(
                               clean_params_for_json(search.best_params_)
                           ),
                           "runtime": runtime,
                       }
                   )

                   all_reg_records.append(metrics)
                   append_reg_record(metrics)
                   done_reg.add(key)

                   print(
                       f"    - {task_name} / {model_name} done. "
                       f"R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}"
                   )

               except Exception as e:
                   warnings.warn(
                       f"[Error] Task {task_name}, Model '{model_name}' "
                       f"on seed={seed}, fold={fold_idx} failed: {e}"
                   )
                   continue
   print("\nStep 3: Aggregating cross-modal reconstruction results...")

   if CROSSMODAL_CSV.exists():
       df_reg = pd.read_csv(CROSSMODAL_CSV)
   else:
       df_reg = pd.DataFrame(all_reg_records)

   if not df_reg.empty:
       needed_cols_reg = {"task", "model", "r2", "rmse", "mae", "pearson"}
       if needed_cols_reg.issubset(df_reg.columns):
           reg_group = (
               df_reg.groupby(["task", "model"])[["r2", "rmse", "mae", "pearson"]]
               .agg(["mean", "std"])
           )

           rows = []
           for (task_name, model_name), row in reg_group.iterrows():

               def fmt(metric):
                   m = row[(metric, "mean")]
                   s = row[(metric, "std")]
                   if pd.isna(m) or pd.isna(s):
                       return ""
                   return f"{m:.3f} ± {s:.3f}"

               rows.append(
                   {
                       "Task": task_name,
                       "Model": model_name,
                       "R2": fmt("r2"),
                       "RMSE": fmt("rmse"),
                       "MAE": fmt("mae"),
                       "Pearson r": fmt("pearson"),
                   }
               )

           reg_summary_table = pd.DataFrame(rows)

           def parse_mean(s):
               try:
                   return float(str(s).split("±")[0])
               except Exception:
                   return -np.inf

           reg_summary_table["__r2_mean"] = reg_summary_table["R2"].apply(parse_mean)
           reg_summary_table = reg_summary_table.sort_values(
               ["Task", "__r2_mean"], ascending=[True, False]
           ).drop(columns="__r2_mean")

           out_path_reg = RESULT_DIR / "crossmodal_reconstruction_summary.csv"
           reg_summary_table.to_csv(out_path_reg, index=False)
           print(
               f"[Info] Saved cross-modal reconstruction summary table to: {out_path_reg}"
           )

       print("\n--- CROSS-MODAL RECONSTRUCTION SUMMARY (Mean +/- Std) ---")
       print(
           df_reg.groupby(["task", "model"])[["r2", "rmse", "mae", "pearson"]]
           .agg(["mean", "std"])
           .sort_values(("r2", "mean"), ascending=False)
           .to_string(float_format="%.4f")
       )
   else:
       print("No cross-modal reconstruction models were run or all failed.")

   print(f"\n>>> All cross-modal experiments finished. Results saved to '{RESULT_DIR}'")


if __name__ == "__main__":
   main()
