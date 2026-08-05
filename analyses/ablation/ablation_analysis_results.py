# ablation_analysis_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
import logging
import re
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, classification_report,
                             confusion_matrix, r2_score, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr
import shutil
from scipy.stats import ttest_ind
from sklearn.metrics import brier_score_loss, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from statsmodels.stats.multitest import fdrcorrection


try:
    from . import config as cfg
    from .config import AblationConfig
    from .visualization import (
        plot_confusion_matrix,
        plot_reconstruction_scatter,
        plot_decision_curve_active_agg
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure this script is in the same directory as config.py and visualization.py")
    exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

SLEMODEL_COLOR = '#D08770'
COMPARISON_PALETTE = ['#5E81AC', '#88C0D0', '#B48EAD']

TARGET_NAMES = [cfg.VisualizationConfig.LABEL_NAMES[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())]
MODALITY_NAMES = ["gly", "mass", "rna"]
MAX_SCATTER_POINTS = 3000


def safe_pearsonr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    return pearsonr(x, y)

def multiclass_brier_macro(y_true: np.ndarray, y_scores: np.ndarray):
    C = y_scores.shape[1]
    y_bin = label_binarize(y_true, classes=range(C))
    briers = [brier_score_loss(y_bin[:, c], y_scores[:, c]) for c in range(C)]
    return float(np.mean(briers))

def temp_scale_fit(y_true: np.ndarray, p: np.ndarray, T_grid=None):
    if T_grid is None: T_grid = np.linspace(0.5, 3.0, 51)
    y, eps = y_true.astype(int), 1e-12
    best_T, best_nll = 1.0, np.inf
    for T in T_grid:
        q = p**(1.0/T); q = q / np.clip(q.sum(axis=1, keepdims=True), eps, None)
        nll = -np.mean(np.log(np.clip(q[np.arange(len(y)), y], eps, 1.0)))
        if nll < best_nll: best_nll, best_T = nll, T
    q_final = p**(1.0/best_T); q_final = q_final / np.clip(q_final.sum(axis=1, keepdims=True), eps, None)
    return best_T, q_final

def get_significance_stars(p_value):
    if p_value < 0.001: return '***'
    elif p_value < 0.01: return '**'
    elif p_value < 0.05: return '*'
    else: return ''


# --------------------------
# --------------------------
def collect_experiment_results(exp_id, exp_name):
    base_dir = Path(cfg.BASE_EXPERIMENT_DIR)
    results = {'classification': [], 'reconstruction': {mod: [] for mod in MODALITY_NAMES}, 'oof_data': {'training': [], 'validation': []}}
    for seed in AblationConfig.EXPERIMENT_SEEDS:
        exp_path = base_dir / f"Ablation_Study_{exp_id}" / f"{exp_id}_{exp_name}_seed_{seed}" / f"master_seed_{seed}"
        metrics_dir = exp_path / "metrics"
        if not metrics_dir.exists():
            logging.debug(f"Path not found for {exp_id} seed {seed}: {metrics_dir}")
            continue

        cls_metrics_file = metrics_dir / "classification_metrics_per_fold.csv"
        if cls_metrics_file.exists():
            df = pd.read_csv(cls_metrics_file)
            df['seed'] = seed
            df['exp_id'] = exp_id
            results['classification'].append(df)

        for mod in MODALITY_NAMES:
            recon_metrics_file = metrics_dir / f"reconstruction_metrics_{mod}_per_fold.csv"
            if recon_metrics_file.exists():
                df = pd.read_csv(recon_metrics_file)
                df['seed'] = seed
                df['exp_id'] = exp_id
                results['reconstruction'][mod].append(df)

        oof_dir = exp_path / "oof_predictions"
        if oof_dir.exists():
            for set_name, file_prefix in [('validation', 'val'), ('training', 'train')]:
                oof_files = list(oof_dir.glob(f"oof_{file_prefix}_predictions_repeat_*.npz"))
                if oof_files:
                    all_true = np.concatenate([np.load(f)['true_labels'] for f in oof_files])
                    all_scores = np.concatenate([np.load(f)['pred_scores'] for f in oof_files])
                    results['oof_data'][set_name].append({'seed': seed, 'exp_id': exp_id, 'true': all_true, 'pred_scores': all_scores})

    results['classification'] = pd.concat(results['classification'], ignore_index=True) if results['classification'] else pd.DataFrame()
    for mod in MODALITY_NAMES:
        results['reconstruction'][mod] = pd.concat(results['reconstruction'][mod], ignore_index=True) if results['reconstruction'][mod] else pd.DataFrame()
    return results


# ============================================================
# ============================================================
def create_comparison_summary_table(all_results, save_path: Path):
    """Create the classification summary table for all ablations."""
    summary_data = []
    slemodel_metrics = {}

    if 'A0' in all_results and not all_results['A0']['classification'].empty:
        slemodel_df = all_results['A0']['classification'][all_results['A0']['classification']['set'] == 'validation']
        if not slemodel_df.empty:
            slemodel_metrics = {
                'AUC_Mean': slemodel_df['macro_roc_auc'].mean(),
                'Accuracy_Mean': slemodel_df['accuracy'].mean(),
                'Recall_Active_Mean': slemodel_df.get('recall_active', pd.Series(dtype=float)).mean(),
                'Recall_Stable_Mean': slemodel_df.get('recall_stable', pd.Series(dtype=float)).mean(),
                'F1_Active_Mean': slemodel_df.get('f1_active', pd.Series(dtype=float)).mean(),
                'F1_Stable_Mean': slemodel_df.get('f1_stable', pd.Series(dtype=float)).mean(),
            }

    for exp_id, exp_name in AblationConfig.EXPERIMENTS.items():
        if exp_id not in all_results or all_results[exp_id]['classification'].empty:
            continue
        val_df = all_results[exp_id]['classification'][all_results[exp_id]['classification']['set'] == 'validation'].copy()
        if val_df.empty:
            continue

        current_metrics = {
            'Accuracy_Mean': val_df['accuracy'].mean(),
            'Accuracy_Std': val_df['accuracy'].std(),
            'AUC_Mean': val_df['macro_roc_auc'].mean(),
            'AUC_Std': val_df['macro_roc_auc'].std(),
        }

        for m in ['recall_active', 'recall_stable', 'f1_active', 'f1_stable']:
            if m in val_df.columns:
                current_metrics[f'{m}_Mean'] = val_df[m].mean()
                current_metrics[f'{m}_Std']  = val_df[m].std()
            else:
                current_metrics[f'{m}_Mean'] = np.nan
                current_metrics[f'{m}_Std']  = np.nan

        row_data = {
            'Experiment_ID': exp_id,
            'Experiment_Name': exp_name,
            **current_metrics,
            'N_Runs': len(val_df)
        }

        # AUC change relative to SLEmodel
        if slemodel_metrics and exp_id != 'A0':
            auc_change = (current_metrics['AUC_Mean'] - slemodel_metrics['AUC_Mean']) / slemodel_metrics['AUC_Mean']
            row_data['AUC_Change_vs_SLEmodel'] = f"{auc_change:+.2%}"
        else:
            row_data['AUC_Change_vs_SLEmodel'] = "---"

        summary_data.append(row_data)

    summary_df = pd.DataFrame(summary_data)

    summary_df['Rank_AUC'] = summary_df['AUC_Mean'].rank(ascending=False, method='min').astype(int)
    summary_df['Rank_Accuracy'] = summary_df['Accuracy_Mean'].rank(ascending=False, method='min').astype(int)

    summary_df = summary_df.sort_values('AUC_Mean', ascending=False)
    summary_df.to_csv(save_path, index=False)
    logging.info(f"Saved enhanced classification summary with ranks to {save_path}")
    return summary_df


def create_reconstruction_summary_table(all_results, save_path: Path):
    recon_summary = []
    for exp_id, exp_name in AblationConfig.EXPERIMENTS.items():
        if exp_id not in all_results:
            continue

        row_data = {'Experiment_ID': exp_id, 'Experiment_Name': exp_name}
        has_data = False
        for mod in MODALITY_NAMES:
            if not all_results[exp_id]['reconstruction'][mod].empty:
                val_df = all_results[exp_id]['reconstruction'][mod][
                    all_results[exp_id]['reconstruction'][mod]['set'] == 'validation'
                ]
                if not val_df.empty:
                    has_data = True
                    for metric in ['r2', 'mae', 'rmse']:
                        mean_val = val_df[metric].mean()
                        std_val = val_df[metric].std()
                        row_data[f'{mod}_{metric}_mean'] = mean_val
                        row_data[f'{mod}_{metric}_std'] = std_val
        if has_data:
            recon_summary.append(row_data)

    if not recon_summary:
        logging.warning("No reconstruction data found to create a summary table.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(recon_summary)
    summary_df.to_csv(save_path, index=False)
    logging.info(f"Saved reconstruction summary table to {save_path}")
    return summary_df


def perform_significance_analysis(all_results, save_path: Path):
    if 'A0' not in all_results:
        logging.error("SLEmodel experiment 'A0' not found. Cannot perform significance analysis.")
        return pd.DataFrame()

    slemodel_cls_df = all_results['A0']['classification']
    slemodel_cls_val = slemodel_cls_df[slemodel_cls_df['set'] == 'validation']

    slemodel_recon_dfs = {
        mod: all_results['A0']['reconstruction'][mod][
            all_results['A0']['reconstruction'][mod]['set'] == 'validation'
        ] for mod in MODALITY_NAMES if not all_results['A0']['reconstruction'][mod].empty
    }

    test_results = []

    cls_metrics = ['accuracy', 'macro_roc_auc', 'recall_active', 'recall_stable', 'f1_active', 'f1_stable']
    recon_metrics = ['r2', 'mae', 'rmse']

    for exp_id, exp_name in AblationConfig.EXPERIMENTS.items():
        if exp_id == 'A0' or exp_id not in all_results:
            continue

        exp_cls_df = all_results[exp_id]['classification']
        if not exp_cls_df.empty:
            exp_cls_val = exp_cls_df[exp_cls_df['set'] == 'validation']
            for metric in cls_metrics:
                if metric in exp_cls_val.columns and not exp_cls_val[metric].isnull().all():
                    _, p_value = ttest_ind(
                        exp_cls_val[metric], slemodel_cls_val[metric],
                        equal_var=False, nan_policy='omit'
                    )
                    test_results.append({'Experiment_ID': exp_id, 'Metric': metric, 'P_Value': p_value})

        for mod in MODALITY_NAMES:
            exp_recon_df = all_results[exp_id]['reconstruction'][mod]
            if not exp_recon_df.empty and mod in slemodel_recon_dfs:
                exp_recon_val = exp_recon_df[exp_recon_df['set'] == 'validation']
                slemodel_recon_val = slemodel_recon_dfs[mod]
                for metric in recon_metrics:
                    if not exp_recon_val[metric].isnull().all():
                        _, p_value = ttest_ind(
                            exp_recon_val[metric], slemodel_recon_val[metric],
                            equal_var=False, nan_policy='omit'
                        )
                        test_results.append({'Experiment_ID': exp_id, 'Metric': f'{mod}_{metric}', 'P_Value': p_value})

    if not test_results:
        logging.warning("No valid test results to perform FDR correction.")
        return pd.DataFrame()

    results_df = pd.DataFrame(test_results)
    reject, q_values = fdrcorrection(results_df['P_Value'], alpha=0.05, method='indep')
    results_df['Q_Value_FDR'] = q_values
    results_df['Reject_H0_at_0.05'] = reject

    results_df = results_df.sort_values('P_Value')
    results_df.to_csv(save_path, index=False)
    logging.info(f"Saved significance analysis results to {save_path}")
    return results_df


# --------------------------
# --------------------------
def main():
    logging.info("=== Ablation result aggregation ===")
    results_dir = Path("ablation_analysis_results")
    if results_dir.exists():
        logging.warning(f"Removing existing results directory: {results_dir}")
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True)

    all_results = {}
    for exp_id, exp_name in AblationConfig.EXPERIMENTS.items():
        logging.info(f"\n---> Collecting results for {exp_id}: {exp_name}")
        exp_results = collect_experiment_results(exp_id, exp_name)

        if not exp_results['classification'].empty or any(not df.empty for df in exp_results['reconstruction'].values()):
            all_results[exp_id] = exp_results
            exp_save_dir = results_dir / f"{exp_id}_{exp_name}"
            exp_save_dir.mkdir(exist_ok=True)
        else:
            logging.warning(f"No valid results found for {exp_id}")

    if not all_results:
        logging.error("No valid results found for any experiment! Exiting.")
        return

    logging.info(f"\n---> Found valid results for {len(all_results)} experiments. Creating comparison summaries...")

    summary_df = create_comparison_summary_table(
        all_results, results_dir / "ablation_comparison_summary_with_ranks.csv"
    )
    create_reconstruction_summary_table(all_results, results_dir / "ablation_reconstruction_summary.csv")
    perform_significance_analysis(all_results, results_dir / "ablation_significance_tests.csv")

    logging.info("\n" + "="*60 + "\nABLATION ANALYSIS COMPLETED!\n" + f"All results saved in: {results_dir.absolute()}\n" + "="*60)

if __name__ == "__main__":
    main()
