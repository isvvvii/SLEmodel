# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging
from pathlib import Path

from .config import VisualizationConfig as Cfg

logger = logging.getLogger(__name__)

def plot_learning_curve(history: dict, fold_name: str, save_path: Path):
    """Plot training and validation learning curves."""
    Cfg.apply_style()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{fold_name}: Model Training History", fontsize=Cfg.TITLE_FONTSIZE, y=0.98)

    epochs = range(1, len(history['train_class_loss']) + 1)

    train_color = Cfg.LEARNING_CURVE_COLORS['train']
    val_color = Cfg.LEARNING_CURVE_COLORS['validation']

    axes[0].plot(epochs, history['train_class_loss'], '-o', color=train_color,
                 label='Train Classification Loss', markersize=4, alpha=0.8)

    if 'val_class_loss' in history and history['val_class_loss']:
        val_epochs = range(1, len(history['val_class_loss']) + 1)
        axes[0].plot(val_epochs, history['val_class_loss'], '-s', color=val_color,
                     label='Validation Classification Loss', markersize=4, alpha=0.8)

    axes[0].set_ylabel("Classification Loss (Focal)")
    axes[0].set_title("Classification Loss Progression")
    axes[0].legend()

    axes[1].plot(epochs, history['train_loss'], 'k-d', label='Total Train Loss', markersize=4, alpha=0.9)
    if 'train_recon_loss' in history:
        axes[1].plot(epochs, history['train_recon_loss'], 'p-', color='rosybrown', label='Paired Recon Loss', markersize=4, alpha=0.7)
    if 'train_semi_loss' in history:
        axes[1].plot(epochs, history['train_semi_loss'], '^-', color='darkseagreen', label='Semi-supervised Recon Loss', markersize=4, alpha=0.7)

    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Training Loss Components")
    axes[1].set_title("Composition of Total Training Loss")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=Cfg.DPI, bbox_inches='tight')
    plt.close(fig)

def plot_roc_curves(y_true, y_score, n_classes, label_names, title_prefix, save_path):
    Cfg.apply_style()
    if len(np.unique(y_true)) < 2: logger.warning(f"Skipping ROC plot for {title_prefix}: Only one class present."); return
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        plt.figure(figsize=(8, 8))
        for i in range(n_classes):
            if i not in label_names or np.sum(y_true_bin[:, i]) == 0: continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            name = label_names.get(i, str(i))
            plt.plot(fpr, tpr, color=Cfg.LABEL_COLORS.get(i), lw=2.5, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        plt.xlim([-0.05, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic ({title_prefix})")
        plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    except Exception as e: logger.error(f"Error plotting ROC curve for {title_prefix}: {e}", exc_info=True)

def plot_pr_curves(y_true, y_score, n_classes, label_names, title_prefix, save_path):
    Cfg.apply_style()
    if len(np.unique(y_true)) < 2: logger.warning(f"Skipping PR plot for {title_prefix}: Only one class present."); return
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        plt.figure(figsize=(8, 8))
        for i in range(n_classes):
            if i not in label_names or np.sum(y_true_bin[:, i]) == 0: continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            pr_auc = auc(recall, precision)
            name = label_names.get(i, str(i))
            plt.plot(recall, precision, color=Cfg.LABEL_COLORS.get(i), lw=2.5, label=f"{name} (AUC = {pr_auc:.3f})")
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({title_prefix})"); plt.legend(loc="best"); plt.grid(True)
        plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()
    except Exception as e: logger.error(f"Error plotting PR curve for {title_prefix}: {e}", exc_info=True)

def plot_confusion_matrix(cm, classes, title_prefix, save_path, normalize=False):
    Cfg.apply_style()
    plot_title = f"Confusion Matrix ({title_prefix})"
    fmt = 'd'
    if normalize:
        plot_title += " - Normalized"
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
        fmt = '.2f'
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=Cfg.HEATMAP_CMAP, xticklabels=classes, yticklabels=classes, annot_kws={"size": Cfg.ANNOTATION_FONTSIZE + 2})
    plt.title(plot_title); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()

def plot_classification_report_heatmap(report_dict, title_prefix, save_path):
    """Plot a heatmap of class-specific performance metrics."""
    Cfg.apply_style()
    class_keys = [key for key in report_dict.keys() if key in Cfg.LABEL_NAMES.values()]
    if not class_keys:
        logger.warning(f"No valid classes found in report_dict for {title_prefix}. Skipping report heatmap.")
        return
    report_df = pd.DataFrame(report_dict).loc[['precision', 'recall', 'f1-score'], class_keys].T

    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df, annot=True, cmap="Greens", fmt='.3f', linewidths=.5, annot_kws={"size": Cfg.ANNOTATION_FONTSIZE + 2})
    plt.title(f"Classification Report ({title_prefix})"); plt.yticks(rotation=0); plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()

def plot_reconstruction_scatter(
    y_true,
    y_pred,
    metrics,
    modality_name,
    title_prefix,
    save_path,
    max_points=2000,
    metrics_std=None
):
    """
    Scatter plot with mean and standard deviation for reconstruction metrics.
    ``metrics_std`` may contain r2, mae, rmse and pearson_corr.
    """
    Cfg.apply_style()
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning(f"Skipping scatter plot for {modality_name} in {title_prefix}: No data.")
        return

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    if len(y_true_flat) > max_points:
        idx = np.random.choice(len(y_true_flat), max_points, replace=False)
        y_true_flat = y_true_flat[idx]
        y_pred_flat = y_pred_flat[idx]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.4, color=Cfg.RECON_SCATTER_COLOR, edgecolors='w', linewidths=0.3)
    min_val = np.min([y_true_flat.min(), y_pred_flat.min()])
    max_val = np.max([y_true_flat.max(), y_pred_flat.max()])
    plt.plot([min_val, max_val], [min_val, max_val], color=Cfg.RECON_IDEAL_LINE_COLOR, linestyle='--', linewidth=2, label='Ideal (y=x)')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{Cfg.MODALITY_NAMES.get(modality_name, modality_name.capitalize())} Reconstruction ({title_prefix})")
    plt.legend()

    def fmt_with_std(key):
        mean_val = metrics.get(key, np.nan)
        if metrics_std and key in metrics_std and metrics_std[key] is not None:
            return f"{mean_val:.3f} ± {metrics_std[key]:.3f}"
        return f"{mean_val:.3f}"

    metrics_text = (
        f"R² = {fmt_with_std('r2')}\n"
        f"MAE = {fmt_with_std('mae')}\n"
        f"RMSE = {fmt_with_std('rmse')}\n"
        f"Pearson's r = {fmt_with_std('pearson_corr')}"
    )
    plt.text(
        0.05, 0.95, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=Cfg.ANNOTATION_FONTSIZE,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=Cfg.DPI)
    plt.close()

def plot_reconstruction_residuals(y_pred, residuals, modality_name, title_prefix, save_path, max_points=3000):
    Cfg.apply_style()
    if y_pred.size == 0 or residuals.size == 0: logger.warning(f"Skipping residual plot for {modality_name} in {title_prefix}: No data."); return
    y_pred_flat, residuals_flat = y_pred.flatten(), residuals.flatten()
    if len(y_pred_flat) > max_points:
        indices = np.random.choice(len(y_pred_flat), max_points, replace=False)
        y_pred_flat, residuals_flat = y_pred_flat[indices], residuals_flat[indices]
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred_flat, y=residuals_flat, lowess=True, scatter_kws={'alpha': 0.3}, line_kws={'color': Cfg.RECON_IDEAL_LINE_COLOR, 'lw': 2})
    plt.xlabel("Predicted Values"); plt.ylabel("Residuals (True - Predicted)")
    plt.title(f"{Cfg.MODALITY_NAMES.get(modality_name, modality_name.capitalize())} Residual Plot ({title_prefix})")
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()

def plot_reconstruction_metrics_bars(metrics_data: dict, title_prefix: str, save_path: Path):
    Cfg.apply_style()
    plot_metrics = ['r2', 'rmse', 'mae']
    records = []
    for mod, metrics in metrics_data.items():
        for metric in plot_metrics:
            if metric in metrics:
                records.append({"Modality": Cfg.MODALITY_NAMES.get(mod, mod.capitalize()), "Metric": metric.upper(), "Value": metrics[metric]})
    if not records: logger.warning(f"Skipping metrics bar plot for {title_prefix}: No data to plot."); return
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]})
    r2_df = df[df['Metric'] == 'R2']
    sns.barplot(data=r2_df, x='Modality', y='Value', hue='Modality', legend=False, palette=list(Cfg.MODALITY_COLORS.values()), ax=axes[0])
    axes[0].set_title('R² Score'); axes[0].set_ylabel('Value (Higher is Better)'); axes[0].set_xlabel('')
    axes[0].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    if not r2_df.empty: axes[0].set_ylim(bottom=min(0, r2_df['Value'].min() - 0.1))
    error_df = df[df['Metric'].isin(['RMSE', 'MAE'])]
    sns.barplot(data=error_df, x='Modality', y='Value', hue='Metric', palette=[Cfg.METRIC_COLORS['rmse'], Cfg.METRIC_COLORS['mae']], ax=axes[1])
    axes[1].set_title('Reconstruction Errors'); axes[1].set_ylabel('Error Value (Lower is Better)'); axes[1].set_xlabel('')
    for ax in axes:
        for p in ax.patches:
            if p.get_height() != 0:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=Cfg.ANNOTATION_FONTSIZE)
    fig.suptitle(f'Reconstruction Performance ({title_prefix})', fontsize=Cfg.TITLE_FONTSIZE + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(save_path, bbox_inches='tight'); plt.close()

# ====== NEW: Calibration & Decision Curve plots ======
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_multiclass_calibration_agg(
    seeds_data,              # list of dicts: {'true': np.ndarray (N,), 'pred_scores': np.ndarray (N, C)}
    label_names,             # list like ['Stable','Active','Control'] (order = class index)
    title, save_path,
    n_bins: int = 10
):
    """
    Aggregate one-vs-rest calibration curves across seeds using fixed bins.
    """
    from .config import VisualizationConfig as Cfg
    Cfg.apply_style()

    C = len(label_names)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    mean_curves = []
    std_curves  = []

    plt.figure(figsize=(8, 7))
    for c in range(C):
        per_seed_curves = []
        for sd in seeds_data:
            y_true = sd['true']
            y_prob = sd['pred_scores'][:, c]
            # binary labels for class c
            y_bin = (y_true == c).astype(int)

            inds = np.digitize(y_prob, bin_edges) - 1
            inds = np.clip(inds, 0, n_bins-1)

            frac_pos = np.zeros(n_bins, dtype=float)
            mean_prob = np.zeros(n_bins, dtype=float)
            for b in range(n_bins):
                mask = (inds == b)
                if mask.any():
                    frac_pos[b] = y_bin[mask].mean()
                    mean_prob[b] = y_prob[mask].mean()
                else:
                    frac_pos[b] = np.nan
                    mean_prob[b] = np.nan

            per_seed_curves.append(np.vstack([mean_prob, frac_pos]))  # shape (2, n_bins)

        stacked = np.stack(per_seed_curves, axis=0)      # (S, 2, n_bins)
        mean_prob = np.nanmean(stacked[:, 0, :], axis=0)
        mean_frac = np.nanmean(stacked[:, 1, :], axis=0)
        std_frac  = np.nanstd (stacked[:, 1, :], axis=0)

        ok = ~np.isnan(mean_prob) & ~np.isnan(mean_frac)
        plt.plot(mean_prob[ok], mean_frac[ok],
                 label=f"{label_names[c]}", lw=2.5, color=Cfg.LABEL_COLORS.get(c, None))
        plt.fill_between(mean_prob[ok],
                         np.maximum(mean_frac[ok]-std_frac[ok], 0),
                         np.minimum(mean_frac[ok]+std_frac[ok], 1),
                         alpha=0.20, color=Cfg.LABEL_COLORS.get(c, None))

    xs = np.linspace(0, 1, 200)
    plt.plot(xs, xs, 'k--', lw=1.5, label='Perfect')

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=Cfg.DPI, bbox_inches='tight')
    plt.close()


def plot_decision_curve_active_agg(
    seeds_data,               # list of dicts: {'true': np.ndarray (N,), 'pred_scores': np.ndarray (N, C)}
    active_class_index: int,  # int, e.g., index of 'Active' in [Stable, Active, Control]
    title, save_path,
    thresholds: np.ndarray = None
):
    """
    Decision-curve analysis for Active versus other classes, summarized across
    seeds with treat-all and treat-none references.
    Treat-all: prevalence - (1-prevalence)*(pt/(1-pt))；Treat-none: 0
    """
    from .config import VisualizationConfig as Cfg
    Cfg.apply_style()

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    per_seed_nb = []   # list of (len(th),) arrays
    per_seed_prev = [] # prevalence for treat-all

    for sd in seeds_data:
        y_true = sd['true']
        y_bin  = (y_true == active_class_index).astype(int)
        p      = sd['pred_scores'][:, active_class_index]
        N = len(y_bin)
        prev = y_bin.mean()
        per_seed_prev.append(prev)

        nb = []
        for pt in thresholds:
            pred_pos = (p >= pt).astype(int)
            TP = np.sum((pred_pos == 1) & (y_bin == 1))
            FP = np.sum((pred_pos == 1) & (y_bin == 0))
            nb.append((TP/N) - (FP/N) * (pt / (1-pt)))
        per_seed_nb.append(np.array(nb))

    nb_stack = np.stack(per_seed_nb, axis=0)  # (S, T)
    nb_mean  = nb_stack.mean(axis=0)
    nb_std   = nb_stack.std(axis=0)

    prev_mean = np.mean(per_seed_prev)
    treat_all = prev_mean - (1 - prev_mean) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 7))
    plt.plot(thresholds, nb_mean, lw=2.5, label="Model (Active)", color=Cfg.LABEL_COLORS.get(1, '#CC6677'))
    plt.fill_between(thresholds, nb_mean - nb_std, nb_mean + nb_std,
                     alpha=0.2, color=Cfg.LABEL_COLORS.get(1, '#CC6677'))
    plt.plot(thresholds, treat_all, 'k--', lw=1.5, label='Treat-all')
    plt.plot(thresholds, treat_none, 'k-',  lw=1.5, label='Treat-none')

    plt.xlabel("Threshold probability")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=Cfg.DPI, bbox_inches='tight')
    plt.close()
