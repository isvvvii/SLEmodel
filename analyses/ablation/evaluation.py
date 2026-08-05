# evaluation.py

import torch
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_auc_score)
from scipy.stats import pearsonr

from . import visualization as viz
from .config import VisualizationConfig as Cfg, NUM_CLASSES
from .utils import FocalLoss

logger = logging.getLogger(__name__)

def evaluate_and_visualize(
    history: dict,
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    pred_logits: np.ndarray,
    recon_data: dict,
    run_dir: Path,
    fold_name: str,
    set_name: str,
    save_plots: bool = True,
    print_report: bool = True,
    class_weights: torch.Tensor = None,
    single_modality_mode: str = None
):
    """Calculate fold-level classification and reconstruction metrics."""
    if save_plots:
        viz_dir = run_dir / "visualizations" / fold_name / set_name.lower()
        viz_dir.mkdir(parents=True, exist_ok=True)
        if print_report:
            logger.info(f"--- Evaluating for {fold_name} ({set_name} Set) ---")
            logger.info(f"Visualizations will be saved to: {viz_dir}")

    if save_plots and history and 'train_class_loss' in history:
        logger.info(f"Plotting learning curve for {fold_name}...")
        final_viz_dir = run_dir / "visualizations" / fold_name
        viz.plot_learning_curve(history, fold_name, final_viz_dir / "learning_curve.png")

    true_labels = np.asarray(true_labels)
    pred_scores = np.asarray(pred_scores)
    pred_logits = np.asarray(pred_logits)
    n_samples = len(true_labels)
    expected_shape = (n_samples, NUM_CLASSES)
    if n_samples == 0:
        raise ValueError(f"No predictions were produced for {fold_name} ({set_name}).")
    if pred_scores.shape != expected_shape:
        raise ValueError(f"Expected prediction scores with shape {expected_shape}, got {pred_scores.shape}.")
    if pred_logits.shape != expected_shape:
        raise ValueError(f"Expected prediction logits with shape {expected_shape}, got {pred_logits.shape}.")
    if not np.isfinite(pred_scores).all() or not np.isfinite(pred_logits).all():
        raise FloatingPointError(f"Non-finite predictions were produced for {fold_name} ({set_name}).")

    y_pred = np.argmax(pred_scores, axis=1)
    accuracy = accuracy_score(true_labels, y_pred)

    unique_labels = np.unique(true_labels)
    if len(unique_labels) > 1:
        macro_auc = roc_auc_score(
            true_labels, pred_scores, average='macro', multi_class='ovr'
        )
    else:
        logger.warning("Macro-AUROC is undefined when only one class is present.")
        macro_auc = np.nan

    report_dict = classification_report(
        true_labels, y_pred,
        labels=list(Cfg.LABEL_NAMES.keys()),
        target_names=list(Cfg.LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0
    )

    loss_fct = FocalLoss(alpha=class_weights)
    loss = loss_fct(
        torch.tensor(pred_logits, dtype=torch.float32),
        torch.tensor(true_labels, dtype=torch.long)
    ).item()

    cls_metrics = {'accuracy': accuracy, 'macro_roc_auc': macro_auc, 'loss': loss}
    for name in Cfg.LABEL_NAMES.values():
        cls_metrics[f"f1_{name.lower()}"] = report_dict.get(name, {}).get('f1-score', 0.0)
        cls_metrics[f"recall_{name.lower()}"] = report_dict.get(name, {}).get('recall', 0.0)

    if print_report:
        report_table_str = pd.DataFrame(report_dict).T.to_string(float_format="%.4f")
        logger.info(f"Classification Report for {fold_name} ({set_name} Set):\n{report_table_str}")

    recon_metrics = {}
    if recon_data and any(d.get('true', np.array([])).size > 0 for d in recon_data.values()):
        for mod_name, data in recon_data.items():

            if single_modality_mode and single_modality_mode != mod_name:
                continue

            true_vals, pred_vals = data.get('true', np.array([])), data.get('pred', np.array([]))
            if true_vals.size == 0 or pred_vals.size == 0:
                continue

            if np.allclose(pred_vals, 0) and not np.allclose(true_vals, 0):
                logger.warning(f"Skipping {mod_name} recon metrics: pred all-zero.")
                continue

            var = np.var(true_vals, axis=0)
            mask = var > 1e-5
            r2 = np.nan if mask.sum() == 0 else r2_score(true_vals[:, mask], pred_vals[:, mask])

            mse = mean_squared_error(true_vals, pred_vals)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(true_vals, pred_vals)

            if np.std(true_vals.flatten()) > 1e-6 and np.std(pred_vals.flatten()) > 1e-6:
                pearson, _ = pearsonr(true_vals.flatten(), pred_vals.flatten())
            else:
                pearson = np.nan

            recon_metrics[mod_name] = {
                'r2': r2, 'mse': mse, 'rmse': rmse,
                'mae': mae, 'pearson_corr': pearson
            }

            if save_plots:
                viz.plot_reconstruction_scatter(
                    true_vals, pred_vals, recon_metrics[mod_name],
                    mod_name, fold_name, viz_dir / f"recon_scatter_{mod_name}.png"
                )
                residuals = true_vals - pred_vals
                viz.plot_reconstruction_residuals(
                    pred_vals, residuals, mod_name, fold_name, viz_dir / f"recon_residuals_{mod_name}.png"
                )

        if recon_metrics and print_report:
            log_msg = f"Reconstruction Metrics for {fold_name} ({set_name} Set)"
            if single_modality_mode:
                log_msg += f" - {single_modality_mode.upper()}-only mode"
            log_msg += ":\n" + pd.DataFrame(recon_metrics).T.to_string(float_format="%.4f")
            logger.info(log_msg)

        if recon_metrics and save_plots:
            viz.plot_reconstruction_metrics_bars(recon_metrics, fold_name, viz_dir / "recon_metrics_bars.png")

    if save_plots:
        if len(unique_labels) > 1:
            viz.plot_roc_curves(true_labels, pred_scores, NUM_CLASSES, Cfg.LABEL_NAMES, fold_name, viz_dir / "roc_curve.png")
            viz.plot_pr_curves(true_labels, pred_scores, NUM_CLASSES, Cfg.LABEL_NAMES, fold_name, viz_dir / "pr_curve.png")

        cm = confusion_matrix(true_labels, y_pred, labels=list(Cfg.LABEL_NAMES.keys()))
        viz.plot_confusion_matrix(cm, list(Cfg.LABEL_NAMES.values()), fold_name, viz_dir / "confusion_matrix.png")
        viz.plot_confusion_matrix(cm, list(Cfg.LABEL_NAMES.values()), fold_name, viz_dir / "confusion_matrix_norm.png", normalize=True)

        if report_dict:
            viz.plot_classification_report_heatmap(report_dict, fold_name, viz_dir / "classification_report.png")

    return cls_metrics, recon_metrics, report_dict
