"""Compare traditional machine-learning classifiers."""

import os
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
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb

# --- PyTorch for Propensity Score ---
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)

from slemodel import config as cfg
from slemodel.models import PropensityScoreClassifier as PSNet
from slemodel.utils import (
    train_classifier,
    get_propensity_scores,
    get_propensity_scores_calibrated,
    get_ot_coupling_modality_specific,
)
SEEDS = [42, 100, 2025, 7, 123]
OUTER_CV_SPLITS = cfg.N_SPLITS_K_FOLD
INNER_CV_SPLITS = 2
N_ITER_SEARCH = 5
PS_EPOCHS = cfg.EPOCHS_PS

RESULT_DIR = pathlib.Path("ml_results_classification")
RESULT_DIR.mkdir(exist_ok=True, parents=True)
CLASSIFICATION_CSV = RESULT_DIR / "classification_metrics_all.csv"

DATA_PATHS = {"gly": cfg.GLY_PATH, "mass": cfg.MASS_PATH, "rna": cfg.RNA_PATH}
LABEL_MAP = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}  # "Stable"/"Active"/"Control" -> int
DEVICE = cfg.DEVICE

SLE_EXP_DIR = pathlib.Path(cfg.BASE_EXPERIMENT_DIR) / cfg.EXPERIMENT_TAG
def load_modality(path):
    df = pd.read_csv(path)
    y_str = df.iloc[:, 1].values
    X = df.iloc[:, 2:].values.astype(np.float32)
    y_enc = pd.Series(y_str).map(LABEL_MAP).values
    return X, y_enc


def spec_sens(y_true, y_pred, labels=(0, 1, 2)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sens_arr = np.diag(cm) / np.maximum(1, cm.sum(axis=1))
    spec_list = []
    for i in range(len(labels)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec_list.append(tn / np.maximum(1, tn + fp))
    spec_arr = np.array(spec_list)
    return sens_arr, spec_arr, sens_arr.mean(), spec_arr.mean()


def get_clf_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan
    sens_arr, spec_arr, sens_mean, spec_mean = spec_sens(y_true, y_pred)
    metrics_dict = {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "sensitivity": sens_mean,
        "specificity": spec_mean,
    }
    metrics_dict.update({f"sens_class{i}": v for i, v in enumerate(sens_arr)})
    metrics_dict.update({f"spec_class{i}": v for i, v in enumerate(spec_arr)})
    return metrics_dict


def append_clf_record(record: dict) -> None:
    df = pd.DataFrame([record])
    file_exists = CLASSIFICATION_CSV.exists()
    df.to_csv(CLASSIFICATION_CSV, mode="a", header=not file_exists, index=False)


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


def parse_fold_idx(fold_name):
    """
    Parse a fold number from names such as ``Repeat_1-Fold_3`` or ``Fold_2``.
    """
    try:
        s = str(fold_name)
        parts = s.split("-")
        for p in parts:
            if p.lower().startswith("fold_"):
                return int(p.split("_")[-1])
    except Exception:
        pass
    return 0


def build_seed_gradient_palette(seed_order, start_color="#F3D7D6", end_color="#2F294E"):
    start = np.array(to_rgb(start_color))
    end = np.array(to_rgb(end_color))
    n = len(seed_order)

    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        c = start * (1 - t) + end * t
        colors.append(tuple(c))
    return {seed_order[i]: colors[i] for i in range(n)}


def order_seeds_for_plot(df):
    preferred = [7, 42, 100, 123, 2025]
    existing = sorted(df["seed"].dropna().unique().tolist())
    return [s for s in preferred if s in existing] or existing
# Propensity-score-guided OT alignment used by SLEmodel
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

def generate_fused_features_aligned(
    gly_train_raw: np.ndarray,
    gly_valid_raw: np.ndarray,
    gly_y_train: np.ndarray,
    gly_y_valid: np.ndarray,
    mass_pool_raw: np.ndarray,
    mass_pool_y: np.ndarray,
    rna_pool_raw: np.ndarray,
    rna_pool_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_train_fused = [gly_scaled, pseudo_mass_scaled, pseudo_rna_scaled]
      X_valid_fused = [gly_scaled, pseudo_mass_scaled, pseudo_rna_scaled]
    Pseudo-features are generated by OT barycentric mapping in standardized
    feature space, consistent with SLEmodel.
    """
    gly_scaler = StandardScaler().fit(gly_train_raw)
    mass_scaler = StandardScaler().fit(mass_pool_raw)
    rna_scaler = StandardScaler().fit(rna_pool_raw)

    gly_train = gly_scaler.transform(gly_train_raw).astype(np.float32)
    gly_valid = gly_scaler.transform(gly_valid_raw).astype(np.float32)
    mass_pool = mass_scaler.transform(mass_pool_raw).astype(np.float32)
    rna_pool = rna_scaler.transform(rna_pool_raw).astype(np.float32)

    gly_x_t = _to_torch(gly_train)
    gly_y_t = _to_long(gly_y_train)
    gly_val_t = _to_torch(gly_valid)
    gly_val_y_t = _to_long(gly_y_valid)

    mass_x_t = _to_torch(mass_pool)
    mass_y_t = _to_long(mass_pool_y)

    rna_x_t = _to_torch(rna_pool)
    rna_y_t = _to_long(rna_pool_y)

    gly_clf = train_classifier(PSNet(gly_x_t.shape[1]), gly_x_t, gly_y_t, DEVICE, epochs=PS_EPOCHS)
    mass_clf = train_classifier(PSNet(mass_x_t.shape[1]), mass_x_t, mass_y_t, DEVICE, epochs=PS_EPOCHS)
    rna_clf = train_classifier(PSNet(rna_x_t.shape[1]), rna_x_t, rna_y_t, DEVICE, epochs=PS_EPOCHS)

    ps_gly_train = _get_ps(gly_clf, gly_x_t, gly_y_t)
    ps_gly_valid = _get_ps(gly_clf, gly_val_t, gly_val_y_t)
    ps_mass_pool = _get_ps(mass_clf, mass_x_t, mass_y_t)
    ps_rna_pool = _get_ps(rna_clf, rna_x_t, rna_y_t)

    G_mass_train = get_ot_coupling_modality_specific(
        ps_gly_train, ps_mass_pool, modality="mass",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )
    G_rna_train = get_ot_coupling_modality_specific(
        ps_gly_train, ps_rna_pool, modality="rna",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )
    G_mass_valid = get_ot_coupling_modality_specific(ps_gly_valid, ps_mass_pool, modality="mass",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )
    G_rna_valid = get_ot_coupling_modality_specific(ps_gly_valid, ps_rna_pool, modality="rna",
        reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
        method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
        cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
        purpose="eval",
    )

    mass_pool_t = _to_torch(mass_pool)
    rna_pool_t = _to_torch(rna_pool)

    pseudo_mass_train = (G_mass_train @ mass_pool_t).numpy()
    pseudo_rna_train = (G_rna_train @ rna_pool_t).numpy()
    pseudo_mass_valid = (G_mass_valid @ mass_pool_t).numpy()
    pseudo_rna_valid = (G_rna_valid @ rna_pool_t).numpy()

    X_train_fused = np.hstack([gly_train, pseudo_mass_train, pseudo_rna_train]).astype(np.float32)
    X_valid_fused = np.hstack([gly_valid, pseudo_mass_valid, pseudo_rna_valid]).astype(np.float32)
    return X_train_fused, X_valid_fused
def get_clf_models_and_params(seed):
    models = OrderedDict()

    lda_params = [
        {
            "lineardiscriminantanalysis__solver": ["svd"],
            "lineardiscriminantanalysis__shrinkage": [None],
        },
        {
            "lineardiscriminantanalysis__solver": ["lsqr", "eigen"],
            "lineardiscriminantanalysis__shrinkage": ["auto"]
            + list(np.linspace(0.1, 0.9, 5)),
        },
    ]
    models["LDA"] = (
        pipeline.make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        lda_params,
    )

    models["LR"] = (
        pipeline.make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="saga", random_state=seed),
        ),
        {
            "logisticregression__C": st.loguniform(1e-3, 1e2),
            "logisticregression__penalty": ["l1", "l2"],
        },
    )

    models["LinSVM"] = (
        pipeline.make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=seed, kernel="linear"),
        ),
        {"svc__C": st.loguniform(1e-3, 1e2)},
    )

    models["DT"] = (
        pipeline.make_pipeline(
            StandardScaler(), DecisionTreeClassifier(random_state=seed)
        ),
        {
            "decisiontreeclassifier__max_depth": st.randint(3, 20),
            "decisiontreeclassifier__min_samples_leaf": st.randint(1, 10),
        },
    )

    models["kNN"] = (
        pipeline.make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": st.randint(3, 21)},
    )

    models["RF"] = (
        pipeline.make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        ),
        {
            "randomforestclassifier__max_depth": st.randint(5, 25),
            "randomforestclassifier__max_features": ["sqrt", "log2", 0.5],
        },
    )

    models["LogReg_EN"] = (
        pipeline.make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=1000,
                random_state=seed,
            ),
        ),
        {
            "logisticregression__C": st.loguniform(1e-3, 1e2),
            "logisticregression__l1_ratio": st.uniform(0.01, 0.98),
        },
    )

    models["XGBoost"] = (
        pipeline.make_pipeline(
            StandardScaler(),
            xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
                device="cuda" if DEVICE == "cuda" else "cpu",
            ),
        ),
        {
            "xgbclassifier__n_estimators": st.randint(100, 800),
            "xgbclassifier__learning_rate": st.loguniform(0.01, 0.3),
            "xgbclassifier__max_depth": st.randint(3, 10),
        },
    )

    return models
def load_slemodel_classification_results():
    records = []
    if not SLE_EXP_DIR.exists():
        return pd.DataFrame()

    for seed in SEEDS:
        run_dir = SLE_EXP_DIR / f"SLEmodel_Run_Seed_{seed}" / f"master_seed_{seed}"
        oof_path = run_dir / "oof_predictions" / "oof_val_predictions_repeat_1.npz"
        if not oof_path.exists():
            continue

        data = np.load(oof_path)
        y_true = data["true_labels"]
        y_score = data["pred_scores"]
        y_pred = np.argmax(y_score, axis=1)

        m = get_clf_metrics(y_true, y_pred, y_score)
        m.update(
            {
                "model": "SLEmodel",
                "seed": seed,
                "fold": 0,
                "best_params": "{}",
                "runtime": np.nan,
            }
        )
        records.append(m)

    return pd.DataFrame(records) if records else pd.DataFrame()
def main():
    print("--- Starting Traditional ML Classification Experiments ---")
    print(f"Using device: {DEVICE}")

    print("Step 1: Loading all modality data...")
    feats, labels = {}, {}
    for m, p in DATA_PATHS.items():
        feats[m], labels[m] = load_modality(p)

    all_clf_records = []
    done_clf = set()

    if CLASSIFICATION_CSV.exists():
        print(
            f"[Info] {CLASSIFICATION_CSV} already exists. "
            "This run will SKIP any (seed, model, fold) that are already in this file. "
            "Delete this cache before a complete rerun."
        )
        try:
            existing_clf_df = pd.read_csv(CLASSIFICATION_CSV)
            if (
                not existing_clf_df.empty
                and {"seed", "model", "fold"}.issubset(existing_clf_df.columns)
            ):
                for _, row in existing_clf_df[["seed", "model", "fold"]].dropna().iterrows():
                    try:
                        done_clf.add((int(row["seed"]), str(row["model"]), int(row["fold"])))
                    except Exception:
                        continue
            print(f"[Info] Loaded {len(done_clf)} completed classification keys.")
        except Exception as e:
            print(f"[Warning] Failed to read existing classification CSV: {e}")
    # Loop over seeds
    for seed in SEEDS:
        print(f"\n{'=' * 20} Running CLASSIFICATION with SEED: {seed} {'=' * 20}")

        print(f"\n--- [SEED {seed}] Single-Modality CLASSIFICATION tasks ---")
        for mod_name in ["gly", "mass", "rna"]:
            print(f"  - Processing Single Modality: {mod_name.upper()}")
            X, y = feats[mod_name], labels[mod_name]

            outer_cv = model_selection.StratifiedKFold(
                n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
            )
            models_to_run = get_clf_models_and_params(seed)

            for model_name, (model, params) in models_to_run.items():
                fold_runtimes = []
                for fold_idx, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), 1):
                    model_tag = f"{mod_name.upper()}_{model_name}"
                    key = (seed, model_tag, fold_idx)

                    if key in done_clf:
                        print(
                            f"    - Skipping {model_tag}, seed={seed}, fold={fold_idx} (already done)."
                        )
                        continue

                    try:
                        fold_start_time = time.time()
                        X_train, X_valid = X[train_idx], X[valid_idx]
                        y_train, y_valid = y[train_idx], y[valid_idx]

                        search = model_selection.RandomizedSearchCV(
                            model,
                            params,
                            n_iter=N_ITER_SEARCH,
                            cv=INNER_CV_SPLITS,
                            scoring="f1_macro",
                            n_jobs=-1,
                            random_state=seed,
                        )
                        search.fit(X_train, y_train)

                        y_pred = search.predict(X_valid)
                        y_proba = search.predict_proba(X_valid)

                        runtime = time.time() - fold_start_time
                        fold_runtimes.append(runtime)

                        metrics_dict = get_clf_metrics(y_valid, y_pred, y_proba)
                        metrics_dict.update(
                            {
                                "model": model_tag,
                                "seed": seed,
                                "fold": fold_idx,
                                "best_params": json.dumps(
                                    clean_params_for_json(search.best_params_)
                                ),
                                "runtime": runtime,
                            }
                        )
                        all_clf_records.append(metrics_dict)
                        append_clf_record(metrics_dict)
                        done_clf.add(key)

                    except Exception as e:
                        warnings.warn(
                            f"Model '{model_name}' on modality '{mod_name}' fold {fold_idx} failed: {e}"
                        )
                        continue

                print(
                    f"    - {model_name} done. Avg fold time: "
                    f"{np.mean(fold_runtimes) if fold_runtimes else -1:.2f}s"
                )

        print(f"\n--- [SEED {seed}] FUSION CLASSIFICATION tasks (Early & Late) ---")

        gly_cv = model_selection.StratifiedKFold(
            n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
        )
        mass_cv = model_selection.StratifiedKFold(
            n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
        )
        rna_cv = model_selection.StratifiedKFold(
            n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=seed
        )
        gly_splits = list(gly_cv.split(feats["gly"], labels["gly"]))
        mass_splits = list(mass_cv.split(feats["mass"], labels["mass"]))
        rna_splits = list(rna_cv.split(feats["rna"], labels["rna"]))

        for fold_idx in range(OUTER_CV_SPLITS):
            print(f"    - Fusion Fold {fold_idx + 1}/{OUTER_CV_SPLITS}...")

            gly_train_idx, gly_valid_idx = gly_splits[fold_idx]
            mass_train_idx, _ = mass_splits[fold_idx]
            rna_train_idx, _ = rna_splits[fold_idx]

            gly_x_train, gly_x_valid = (
                feats["gly"][gly_train_idx],
                feats["gly"][gly_valid_idx],
            )
            gly_y_train, gly_y_valid = (
                labels["gly"][gly_train_idx],
                labels["gly"][gly_valid_idx],
            )

            mass_pool_x, mass_pool_y = (
                feats["mass"][mass_train_idx],
                labels["mass"][mass_train_idx],
            )
            rna_pool_x, rna_pool_y = (
                feats["rna"][rna_train_idx],
                labels["rna"][rna_train_idx],
            )

            print("      - Generating PS+OT fused features (aligned with SLEmodel)...")
            X_train_fused, X_valid_fused = generate_fused_features_aligned(
                gly_train_raw=gly_x_train,
                gly_valid_raw=gly_x_valid,
                gly_y_train=gly_y_train,
                gly_y_valid=gly_y_valid,
                mass_pool_raw=mass_pool_x,
                mass_pool_y=mass_pool_y,
                rna_pool_raw=rna_pool_x,
                rna_pool_y=rna_pool_y,
            )
            y_train, y_valid = gly_y_train, gly_y_valid

            models_early = {
                k: v
                for k, v in get_clf_models_and_params(seed).items()
                if k in ["LogReg_EN", "XGBoost"]
            }

            for model_name, (model, params) in models_early.items():
                model_tag = f"EarlyFusion_{model_name}"
                key = (seed, model_tag, fold_idx + 1)

                if key in done_clf:
                    print(
                        f"        - Skipping {model_tag}, seed={seed}, fold={fold_idx + 1} (already done)."
                    )
                    continue

                try:
                    start_time = time.time()
                    search = model_selection.RandomizedSearchCV(
                        model,
                        params,
                        n_iter=N_ITER_SEARCH,
                        cv=INNER_CV_SPLITS,
                        scoring="f1_macro",
                        n_jobs=-1,
                        random_state=seed,
                    )
                    search.fit(X_train_fused, y_train)

                    y_pred = search.predict(X_valid_fused)
                    y_proba = search.predict_proba(X_valid_fused)
                    runtime = time.time() - start_time

                    metrics_dict = get_clf_metrics(y_valid, y_pred, y_proba)
                    metrics_dict.update(
                        {
                            "model": model_tag,
                            "seed": seed,
                            "fold": fold_idx + 1,
                            "best_params": json.dumps(
                                clean_params_for_json(search.best_params_)
                            ),
                            "runtime": runtime,
                        }
                    )
                    all_clf_records.append(metrics_dict)
                    append_clf_record(metrics_dict)
                    done_clf.add(key)
                    print(f"        - EarlyFusion_{model_name} done.")

                except Exception as e:
                    warnings.warn(
                        f"Model 'EarlyFusion_{model_name}' on fold {fold_idx + 1} failed: {e}"
                    )
                    continue

            # Late Fusion (Stacking)
            model_tag = "LateFusion_Stacking"
            key = (seed, model_tag, fold_idx + 1)

            if key in done_clf:
                print(
                    f"        - Skipping {model_tag}, seed={seed}, fold={fold_idx + 1} (already done)."
                )
            else:
                try:
                    start_time = time.time()

                    n_gly = feats["gly"].shape[1]
                    n_mass = feats["mass"].shape[1]
                    n_rna = feats["rna"].shape[1]

                    assert (
                        X_train_fused.shape[1] == n_gly + n_mass + n_rna
                    ), "Fused feature dimension mismatch!"

                    def select_gly(x):
                        return x[:, :n_gly]

                    def select_mass(x):
                        return x[:, n_gly : n_gly + n_mass]

                    def select_rna(x):
                        return x[:, n_gly + n_mass :]

                    estimators = [
                        (
                            "gly_rf",
                            pipeline.make_pipeline(
                                FunctionTransformer(select_gly),
                                StandardScaler(),
                                RandomForestClassifier(
                                    n_estimators=100,
                                    random_state=seed,
                                    n_jobs=-1,
                                ),
                            ),
                        ),
                        (
                            "mass_rf",
                            pipeline.make_pipeline(
                                FunctionTransformer(select_mass),
                                StandardScaler(),
                                RandomForestClassifier(
                                    n_estimators=100,
                                    random_state=seed,
                                    n_jobs=-1,
                                ),
                            ),
                        ),
                        (
                            "rna_rf",
                            pipeline.make_pipeline(
                                FunctionTransformer(select_rna),
                                StandardScaler(),
                                RandomForestClassifier(
                                    n_estimators=100,
                                    random_state=seed,
                                    n_jobs=-1,
                                ),
                            ),
                        ),
                    ]

                    final_estimator_pipeline = pipeline.make_pipeline(
                        StandardScaler(), LogisticRegression(max_iter=1000)
                    )

                    stacking_clf = StackingClassifier(
                        estimators=estimators,
                        final_estimator=final_estimator_pipeline,
                        cv=INNER_CV_SPLITS,
                        n_jobs=-1,
                        passthrough=True,
                    )

                    stacking_clf.fit(X_train_fused, y_train)

                    y_pred = stacking_clf.predict(X_valid_fused)
                    y_proba = stacking_clf.predict_proba(X_valid_fused)

                    runtime = time.time() - start_time

                    metrics_dict = get_clf_metrics(y_valid, y_pred, y_proba)
                    metrics_dict.update(
                        {
                            "model": model_tag,
                            "seed": seed,
                            "fold": fold_idx + 1,
                            "best_params": "{}",
                            "runtime": runtime,
                        }
                    )
                    all_clf_records.append(metrics_dict)
                    append_clf_record(metrics_dict)
                    done_clf.add(key)
                    print(f"        - {model_tag} done.")

                except Exception as e:
                    warnings.warn(f"Model '{model_tag}' on fold {fold_idx + 1} failed: {e}")
                    continue
    print("\nStep 3: Aggregating classification results and generating visualizations...")

    if CLASSIFICATION_CSV.exists():
        df_clf = pd.read_csv(CLASSIFICATION_CSV)
    else:
        df_clf = pd.DataFrame(all_clf_records)

    sle_clf_df = load_slemodel_classification_results()
    if not sle_clf_df.empty:
        print(f"[Info] Loaded SLEmodel classification results: {len(sle_clf_df)} rows.")
        df_clf = pd.concat([df_clf, sle_clf_df], ignore_index=True)

    if not df_clf.empty:
        needed_cols = {"model", "auc", "f1", "sensitivity", "acc"}
        if needed_cols.issubset(df_clf.columns):
            clf_group = df_clf.groupby("model")[["auc", "f1", "sensitivity", "acc"]].agg(
                ["mean", "std"]
            )

            rows = []
            for model_name, row in clf_group.iterrows():

                def fmt(metric):
                    m = row[(metric, "mean")]
                    s = row[(metric, "std")]
                    if pd.isna(m) or pd.isna(s):
                        return ""
                    return f"{m:.3f} ± {s:.3f}"

                rows.append(
                    {
                        "Model": model_name,
                        "Macro AUC": fmt("auc"),
                        "F1 (Macro)": fmt("f1"),
                        "Sensitivity (Avg)": fmt("sensitivity"),
                        "Accuracy": fmt("acc"),
                    }
                )

            clf_summary_table = pd.DataFrame(rows)

            def parse_mean(s):
                try:
                    return float(str(s).split("±")[0])
                except Exception:
                    return -np.inf

            clf_summary_table["__auc_mean"] = clf_summary_table["Macro AUC"].apply(parse_mean)
            clf_summary_table = (
                clf_summary_table.sort_values("__auc_mean", ascending=False)
                .drop(columns="__auc_mean")
            )

            out_path_cls = RESULT_DIR / "classification_summary.csv"
            clf_summary_table.to_csv(out_path_cls, index=False)
            print(f"[Info] Saved classification summary table to: {out_path_cls}")

    plt.style.use("seaborn-v0_8-whitegrid")

    if not df_clf.empty:
        fig, axes = plt.subplots(2, 1, figsize=(20, 16))

        seed_order = order_seeds_for_plot(df_clf)
        seed_palette = build_seed_gradient_palette(
            seed_order=seed_order,
            start_color="#F3D7D6",
            end_color="#2F294E",
        )

        sns.boxplot(
            data=df_clf,
            x="model",
            y="f1",
            hue="seed",
            hue_order=seed_order,
            palette=seed_palette,
            ax=axes[0],
            dodge=True,
        )
        axes[0].set_title("Macro F1-Score Comparison (Aggregated over Folds)", fontsize=16)
        axes[0].tick_params(axis="x", rotation=45)
        for label in axes[0].get_xticklabels():
            label.set_ha("right")

        sns.boxplot(
            data=df_clf,
            x="model",
            y="auc",
            hue="seed",
            hue_order=seed_order,
            palette=seed_palette,
            ax=axes[1],
            dodge=True,
        )
        axes[1].set_title("Macro ROC-AUC Comparison", fontsize=16)
        axes[1].tick_params(axis="x", rotation=45)
        for label in axes[1].get_xticklabels():
            label.set_ha("right")

        plt.tight_layout()
        plt.savefig(RESULT_DIR / "boxplot_classification_metrics.png", dpi=300)
        plt.close()
    else:
        print("[Warning] No classification results found. Skipping classification plots.")

    if not df_clf.empty:
        mean_clf_df = (
            df_clf.groupby("model")[["acc", "f1", "auc", "sensitivity", "specificity"]]
            .mean()
            .fillna(0)
        )

        benchmark_rank = mean_clf_df.drop(index=["SLEmodel"], errors="ignore")
        top_benchmark_models = benchmark_rank.sort_values("f1", ascending=False).head(3).index.tolist()

        labels_radar = mean_clf_df.columns.tolist()
        num_vars = len(labels_radar)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        sle_color = "#BF616A"
        top_colors = ["#4C72B0", "#55A868", "#8172B2"]
        other_color = "#B0B0B0"

        for model_name in mean_clf_df.index:
            values = mean_clf_df.loc[model_name].tolist()
            values += values[:1]

            if model_name == "SLEmodel":
                continue
            if model_name in top_benchmark_models:
                continue

            ax.plot(
                angles,
                values,
                color=other_color,
                linewidth=1.0,
                alpha=0.25,
                label=None,
            )
            ax.fill(angles, values, color=other_color, alpha=0.03)

        for i, model_name in enumerate(top_benchmark_models):
            values = mean_clf_df.loc[model_name].tolist()
            values += values[:1]

            c = top_colors[i % len(top_colors)]
            ax.plot(
                angles,
                values,
                color=c,
                linewidth=2.0,
                alpha=0.65,
                label=model_name,
            )
            ax.fill(angles, values, color=c, alpha=0.08)

        if "SLEmodel" in mean_clf_df.index:
            values = mean_clf_df.loc["SLEmodel"].tolist()
            values += values[:1]

            ax.plot(
                angles,
                values,
                color=sle_color,
                linewidth=3.5,
                alpha=0.95,
                label="SLEmodel",
            )
            ax.fill(angles, values, color=sle_color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar)
        ax.set_title("Mean Classification Metrics Radar Plot", size=20, y=1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

        plt.savefig(
            RESULT_DIR / "radar_classification_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"\n>>> All classification experiments finished. Results and plots saved to '{RESULT_DIR}'")

    print("\n--- CLASSIFICATION SUMMARY (Mean +/- Std) ---")
    if not df_clf.empty:
        print(
            df_clf.groupby("model")[["f1", "auc", "acc", "sensitivity", "specificity"]]
            .agg(["mean", "std"])
            .sort_values(("f1", "mean"), ascending=False)
            .to_string(float_format="%.4f")
        )
    else:
        print("No classification models were run or all failed.")


if __name__ == "__main__":
    main()
