# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging
from pathlib import Path
from matplotlib.ticker import FuncFormatter

from .config import VisualizationConfig as Cfg

logger = logging.getLogger(__name__)


def _apply_prob_axis_style(ax, hide_y_zero=True):
    """Apply one-decimal tick labels to probability axes."""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    if hide_y_zero:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: "" if np.isclose(y, 0.0) else f"{y:.1f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))


def plot_learning_curve(history: dict, fold_name: str, save_path: Path):
    """Plot training and validation losses for one fold."""
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
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Skipping ROC plot for {title_prefix}: Only one class present.")
        return
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        plt.figure(figsize=(8, 8))
        for i in range(n_classes):
            if i not in label_names or np.sum(y_true_bin[:, i]) == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            name = label_names.get(i, str(i))
            plt.plot(fpr, tpr, color=Cfg.LABEL_COLORS.get(i), lw=2.5, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        _apply_prob_axis_style(plt.gca(), hide_y_zero=True)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title_prefix)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting ROC curve for {title_prefix}: {e}", exc_info=True)


def plot_pr_curves(y_true, y_score, n_classes, label_names, title_prefix, save_path):
    Cfg.apply_style()
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Skipping PR plot for {title_prefix}: Only one class present.")
        return
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        plt.figure(figsize=(8, 8))
        for i in range(n_classes):
            if i not in label_names or np.sum(y_true_bin[:, i]) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            pr_auc = auc(recall, precision)
            name = label_names.get(i, str(i))
            plt.plot(recall, precision, color=Cfg.LABEL_COLORS.get(i), lw=2.5, label=f"{name} (AUC = {pr_auc:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        _apply_prob_axis_style(plt.gca(), hide_y_zero=True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title_prefix)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting PR curve for {title_prefix}: {e}", exc_info=True)


def plot_confusion_matrix(cm, classes, title_prefix, save_path, normalize=False):
    Cfg.apply_style()
    plot_title = title_prefix
    fmt = 'd'
    matrix = cm.copy()
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        matrix = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
        fmt = '.2f'
    plt.figure(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=Cfg.HEATMAP_CMAP, xticklabels=classes, yticklabels=classes,
                annot_kws={"size": Cfg.ANNOTATION_FONTSIZE + 2})
    plt.title(plot_title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_classification_report_heatmap(report_dict, title_prefix, save_path):
    """Plot class-specific precision, recall and F1 scores."""
    Cfg.apply_style()
    class_keys = [key for key in report_dict.keys() if key in Cfg.LABEL_NAMES.values()]
    if not class_keys:
        logger.warning(f"No valid classes found in report_dict for {title_prefix}. Skipping report heatmap.")
        return
    report_df = pd.DataFrame(report_dict).loc[['precision', 'recall', 'f1-score'], class_keys].T

    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df, annot=True, cmap="Greens", fmt='.3f', linewidths=.5,
                annot_kws={"size": Cfg.ANNOTATION_FONTSIZE + 2})
    plt.title(title_prefix)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_reconstruction_scatter(
    y_true,
    y_pred,
    metrics,
    modality_name,
    title_prefix,
    save_path,
    max_points=2000
):
    """Plot observed and reconstructed feature values with marginal densities."""
    Cfg.apply_style()
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning(f"Skipping scatter plot for {modality_name} in {title_prefix}: No data.")
        return

    # -------- flatten + finite mask --------
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    finite_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_flat = y_true_flat[finite_mask]
    y_pred_flat = y_pred_flat[finite_mask]

    if y_true_flat.size == 0 or y_pred_flat.size == 0:
        logger.warning(f"Skipping scatter plot for {modality_name} in {title_prefix}: non-finite only.")
        return

    # -------- deterministic subsampling (for speed & reproducibility) --------
    if y_true_flat.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(y_true_flat.size, size=max_points, replace=False)
        y_true_flat = y_true_flat[idx]
        y_pred_flat = y_pred_flat[idx]

    # -------- modality color --------
    mod_key = (modality_name or "").lower().strip()
    color = getattr(Cfg, "RECON_SCATTER_COLORS", {}).get(mod_key, Cfg.RECON_SCATTER_COLOR)

    s = getattr(Cfg, "RECON_SCATTER_SIZE", 10)
    alpha = getattr(Cfg, "RECON_SCATTER_ALPHA", 0.35)
    bins = int(getattr(Cfg, "RECON_MARGINAL_BINS", 30))
    reg_lw = float(getattr(Cfg, "RECON_REG_LINE_WIDTH", 2.2))

    # -------- axis limits (shared, square-ish) --------
    min_val = float(np.min([y_true_flat.min(), y_pred_flat.min()]))
    max_val = float(np.max([y_true_flat.max(), y_pred_flat.max()]))
    pad = 0.02 * (max_val - min_val + 1e-9)
    xmin, xmax = min_val - pad, max_val + pad

    # -------- layout: main + marginals (top/right) --------
    fig = plt.figure(figsize=(8.2, 7.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4.8, 1.35],
        height_ratios=[1.35, 4.8],
        wspace=0.05,
        hspace=0.05
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_right)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    # -------- main: scatter + regression (with CI) --------
    sns.regplot(
        x=y_true_flat,
        y=y_pred_flat,
        ax=ax_main,
        ci=95,
        n_boot=500,
        truncate=False,
        scatter_kws=dict(s=s, alpha=alpha, color=color, edgecolors="none"),
        line_kws=dict(color=color, lw=reg_lw, alpha=0.95),
    )

    # ideal line y=x
    ax_main.plot(
        [xmin, xmax], [xmin, xmax],
        color=Cfg.RECON_IDEAL_LINE_COLOR,
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        label="Ideal (y=x)"
    )

    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(xmin, xmax)
    ax_main.set_xlabel("True Values")
    ax_main.set_ylabel("Predicted Values")
    ax_main.set_title(title_prefix)

    # legend (only ideal line)
    ax_main.legend(loc="lower right", frameon=False)

    # subtle grid (Nature-style)
    ax_main.grid(True, linestyle="--", linewidth=0.8, alpha=0.18)

    # -------- marginals: hist + KDE --------
    # top: distribution of True
    sns.histplot(
        x=y_true_flat,
        ax=ax_top,
        bins=bins,
        stat="density",
        kde=True,
        color=color,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.3,
        line_kws=dict(lw=1.4),
    )
    ax_top.set_ylabel("")
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(False)

    # right: distribution of Pred
    sns.histplot(
        y=y_pred_flat,
        ax=ax_right,
        bins=bins,
        stat="density",
        kde=True,
        color=color,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.3,
        line_kws=dict(lw=1.4),
    )
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(False)

    # tidy spines
    sns.despine(ax=ax_top, left=True)
    sns.despine(ax=ax_right, bottom=True)
    sns.despine(ax=ax_main)

    # -------- metrics box (top-left) --------
    def _fmt(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.3f}"

    r2 = _fmt(metrics.get("r2"))
    r2_sd = metrics.get("r2_sd")
    mae = _fmt(metrics.get("mae"))
    mae_sd = metrics.get("mae_sd")
    rmse = _fmt(metrics.get("rmse"))
    rmse_sd = metrics.get("rmse_sd")
    pearson = _fmt(metrics.get("pearson_corr"))
    pearson_sd = metrics.get("pearson_corr_sd")

    def _with_sd(mean_str, sd_val):
        return f"{mean_str} \u00B1 {sd_val:.3f}" if isinstance(sd_val, (float, int)) and not np.isnan(sd_val) else mean_str

    metrics_text = (
        f"R\u00b2 = {_with_sd(r2, r2_sd)}\n"
        f"MAE = {_with_sd(mae, mae_sd)}\n"
        f"RMSE = {_with_sd(rmse, rmse_sd)}\n"
        f"Pearson's r = {_with_sd(pearson, pearson_sd)}"
    )

    ax_main.text(
        0.03, 0.97,
        metrics_text,
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=Cfg.ANNOTATION_FONTSIZE - 1,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", lw=0.8, alpha=0.90),
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=Cfg.DPI)
    plt.close(fig)

def plot_reconstruction_residuals(y_pred, residuals, modality_name, title_prefix, save_path, metrics=None, max_points=3000):
    """Plot reconstruction residuals and summary errors."""
    Cfg.apply_style()
    if y_pred.size == 0 or residuals.size == 0:
        logger.warning(f"Skipping residual plot for {modality_name} in {title_prefix}: No data.")
        return
    y_pred_flat, residuals_flat = y_pred.flatten(), residuals.flatten()
    if len(y_pred_flat) > max_points:
        indices = np.random.choice(len(y_pred_flat), max_points, replace=False)
        y_pred_flat, residuals_flat = y_pred_flat[indices], residuals_flat[indices]
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred_flat, y=residuals_flat, lowess=True, scatter_kws={'alpha': 0.3},
                  line_kws={'color': Cfg.RECON_IDEAL_LINE_COLOR, 'lw': 2})
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title(title_prefix)

    mean_res = float(np.mean(residuals_flat))
    std_res = float(np.std(residuals_flat, ddof=1))
    def _fmt(v): return f"{v:.3f}"
    mae = metrics.get('mae') if metrics else None
    rmse = metrics.get('rmse') if metrics else None
    mae_sd = metrics.get('mae_sd') if metrics else None
    rmse_sd = metrics.get('rmse_sd') if metrics else None
    def _with_sd(mean_val, sd_val):
        if mean_val is None: return "N/A"
        if sd_val is None or np.isnan(sd_val): return f"{mean_val:.3f}"
        return f"{mean_val:.3f} \u00B1 {sd_val:.3f}"
    text = (f"MAE = {_with_sd(mae, mae_sd)}\n"
            f"RMSE = {_with_sd(rmse, rmse_sd)}\n"
            f"Mean Residual = {_fmt(mean_res)}\n"
            f"Std Residual = {_fmt(std_res)}")
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
             fontsize=Cfg.ANNOTATION_FONTSIZE-1, va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_multiclass_calibration(calib_df: pd.DataFrame, title_prefix: str, save_path: Path,
                                class_order, class_names, class_colors, annotation_text: str | None):
    """Plot one-vs-rest calibration curves with confidence intervals."""
    Cfg.apply_style()
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    ax.plot([0, 1], [0, 1], color='0.6', lw=1.2, ls='--', label='Identity')

    for i, cls_idx in enumerate(class_order):
        sub = calib_df[calib_df['class_index'] == cls_idx].copy()
        if sub.empty:
            continue
        sub.sort_values('prob_mean', inplace=True)
        color = class_colors[i]
        name = class_names[i]

        ax.fill_between(sub['prob_mean'], sub['ci_low'], sub['ci_high'],
                        color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(sub['prob_mean'], sub['frac_pos'], color=color, lw=2.2, alpha=0.95, zorder=2, label=name)
        ax.scatter(sub['prob_mean'], sub['frac_pos'], s=16, color=color, edgecolors='white', linewidths=0.6, zorder=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _apply_prob_axis_style(ax, hide_y_zero=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title_prefix)

    ax.legend(loc='upper left', frameon=False)

    if annotation_text:
        ax.text(0.98, 0.05, annotation_text, transform=ax.transAxes,
                ha='right', va='bottom', multialignment='left',
                fontsize=Cfg.ANNOTATION_FONTSIZE-2,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=Cfg.DPI)
    plt.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_temperature_scaling_effect(pre_probs, post_probs, temperature, n_classes, title_prefix, save_path):
    """Plot confidence values before and after temperature scaling."""
    Cfg.apply_style()
    c_pre = pre_probs.max(axis=1)
    c_post = post_probs.max(axis=1)

    grid = np.linspace(1e-4, 1 - 1e-4, 400)
    def ts_map(c, K, T):
        top = np.clip(c, 1e-12, 1-1e-12)
        rest = (1 - top) / (K - 1)
        vec = np.concatenate(([top], np.full(K - 1, rest)))
        logits = np.log(vec)
        scaled = np.exp(logits / T)
        scaled = scaled / scaled.sum()
        return scaled[0]
    map_vals = np.array([ts_map(x, n_classes, temperature) for x in grid])

    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(c_pre, bins) - 1, 0, len(bins) - 2)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_mean_post = np.array([c_post[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan for i in range(len(bin_centers))])

    plt.figure(figsize=(7, 7))
    plt.scatter(c_pre, c_post, s=10, alpha=0.12, color='0.5', edgecolors='none', label='Samples')
    plt.plot(bin_centers[~np.isnan(bin_mean_post)], bin_mean_post[~np.isnan(bin_mean_post)],
             color=Cfg.LEARNING_CURVE_COLORS['validation'], lw=2.2, marker='o', ms=4, label='Binned Mean')
    plt.plot(grid, map_vals, color=Cfg.LEARNING_CURVE_COLORS['train'], lw=2, linestyle='--',
             label=f'Theoretical (T={temperature:.2f})')
    plt.plot([0, 1], [0, 1], color='0.6', lw=1.2, linestyle=':', label='Identity')

    plt.xlabel("Original Confidence")
    plt.ylabel("Calibrated Confidence")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    _apply_prob_axis_style(plt.gca(), hide_y_zero=True)
    plt.title(title_prefix)
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=Cfg.DPI)
    plt.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    df_out = pd.DataFrame({
        'grid_x': grid, 'map_y': map_vals
    })
    df_out['temperature'] = temperature
    df_out['n_classes'] = n_classes
    df_out_bins = pd.DataFrame({'bin_center': bin_centers, 'binned_mean_y': bin_mean_post})
    return pd.concat([df_out, df_out_bins], axis=1)


def plot_reconstruction_metrics_bars(metrics_data: dict, title_prefix: str, save_path: Path):
    Cfg.apply_style()
    plot_metrics = ['r2', 'rmse', 'mae']
    records = []
    for mod, metrics in metrics_data.items():
        for metric in plot_metrics:
            if metric in metrics:
                records.append({"Modality": Cfg.MODALITY_NAMES.get(mod, mod.capitalize()), "Metric": metric.upper(), "Value": metrics[metric]})
    if not records:
        logger.warning(f"Skipping metrics bar plot for {title_prefix}: No data to plot.")
        return
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]})
    r2_df = df[df['Metric'] == 'R2']
    sns.barplot(data=r2_df, x='Modality', y='Value', hue='Modality', legend=False,
                palette=list(Cfg.MODALITY_COLORS.values()), ax=axes[0])
    axes[0].set_title('R\u00b2 Score')
    axes[0].set_ylabel('Value (Higher is Better)')
    axes[0].set_xlabel('')
    axes[0].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    if not r2_df.empty:
        axes[0].set_ylim(bottom=min(0, r2_df['Value'].min() - 0.1))
    error_df = df[df['Metric'].isin(['RMSE', 'MAE'])]
    sns.barplot(data=error_df, x='Modality', y='Value', hue='Metric',
                palette=[Cfg.METRIC_COLORS['rmse'], Cfg.METRIC_COLORS['mae']], ax=axes[1])
    axes[1].set_title('Reconstruction Errors')
    axes[1].set_ylabel('Error Value (Lower is Better)')
    axes[1].set_xlabel('')
    for ax in axes:
        for p in ax.patches:
            if p.get_height() != 0:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                            fontsize=Cfg.ANNOTATION_FONTSIZE)
    fig.suptitle(title_prefix, fontsize=Cfg.TITLE_FONTSIZE + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
