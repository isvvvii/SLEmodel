# collect_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import shutil
from sklearn.metrics import (classification_report, confusion_matrix, r2_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from . import config as cfg
from .visualization import (plot_confusion_matrix, plot_reconstruction_scatter,
                           plot_reconstruction_residuals, plot_reconstruction_metrics_bars,
                           plot_multiclass_calibration, plot_temperature_scaling_effect)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEEDS = [42, 100, 2025, 7, 123]
EXPERIMENT_TAG = cfg.EXPERIMENT_TAG
BASE_DIR = Path(cfg.BASE_EXPERIMENT_DIR)
OUTPUT_DIR = Path("analysis_results_foldmean")
SOURCE_DATA_DIR = OUTPUT_DIR / "source_data"

TARGET_NAMES = [cfg.VisualizationConfig.LABEL_NAMES[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())]
CLASS_COLORS = [cfg.VisualizationConfig.LABEL_COLORS[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())]
MODALITY_NAMES = ["gly", "mass", "rna"]
N_CLASSES = cfg.NUM_CLASSES


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
    eces = []
    for k in range(N_CLASSES):
        if y_true_bin[:, k].sum() == 0:
            continue
        prob = y_prob[:, k]
        true = y_true_bin[:, k]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            idx = (prob >= bins[i]) & (prob < bins[i + 1] if i < n_bins - 1 else prob <= bins[i + 1])
            if idx.any():
                acc = true[idx].mean()
                conf = prob[idx].mean()
                ece += np.abs(acc - conf) * idx.mean()
        eces.append(ece)
    return float(np.mean(eces)) if eces else np.nan


def multi_class_brier_score(y_true, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))


def temperature_scale_probabilities(y_prob, y_true, max_iter=1000, lr=0.05):
    logits = np.log(np.clip(y_prob, 1e-12, 1.0))
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(y_true, dtype=torch.long)

    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_temp], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperature = torch.exp(log_temp)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        optimizer.step()

    temperature = torch.exp(log_temp).item()
    temperature = max(temperature, 1e-3)
    scaled = F.softmax(logits_t / temperature, dim=1).cpu().numpy()
    return scaled, temperature


def _one_decimal(x, pos):
    return f"{x:.1f}"


def _one_decimal_blank_zero(x, pos):
    return "" if np.isclose(x, 0.0) else f"{x:.1f}"


def safe_pearsonr(x, y):
    """Return Pearson's r, including a defined result for constant arrays."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    return pearsonr(x, y)


def collect_slemodel_results():
    """Collect fold-level outputs from all five master seeds."""
    results = {
        'classification': [],
        'reconstruction_metrics': {mod: [] for mod in MODALITY_NAMES},
        'oof_data': {'training': [], 'validation': []},
        'recon_raw_data': {
            'training': {mod: {'true': [], 'pred': []} for mod in MODALITY_NAMES},
            'validation': {mod: {'true': [], 'pred': []} for mod in MODALITY_NAMES}
        }
    }

    for seed in SEEDS:
        run_name = f"SLEmodel_Run_Seed_{seed}"
        exp_path = BASE_DIR / EXPERIMENT_TAG / run_name / f"master_seed_{seed}"

        if not exp_path.exists():
            logging.warning(f"Results path not found for seed {seed}: {exp_path}")
            continue

        logging.info(f"Collecting results from seed {seed}...")

        metrics_dir = exp_path / "metrics"
        if (metrics_dir / "classification_metrics_per_fold.csv").exists():
            df = pd.read_csv(metrics_dir / "classification_metrics_per_fold.csv")
            df['seed'] = seed
            results['classification'].append(df)

        for mod in MODALITY_NAMES:
            if (metrics_dir / f"reconstruction_metrics_{mod}_per_fold.csv").exists():
                df = pd.read_csv(metrics_dir / f"reconstruction_metrics_{mod}_per_fold.csv")
                df['seed'] = seed
                df['modality'] = mod
                results['reconstruction_metrics'][mod].append(df)

        oof_dir = exp_path / "oof_predictions"
        for set_name in ['training', 'validation']:
            prefix = 'train' if set_name == 'training' else 'val'
            oof_file = oof_dir / f"oof_{prefix}_predictions_repeat_1.npz"
            if oof_file.exists():
                data = np.load(oof_file)
                results['oof_data'][set_name].append({
                    'seed': seed, 'true': data['true_labels'], 'pred_scores': data['pred_scores']
                })

        recon_dir = exp_path / "reconstruction_data"
        if recon_dir.exists():
            recon_files = list(recon_dir.glob("*.npz"))
            for recon_file in recon_files:
                set_name = 'training' if 'training' in recon_file.name else 'validation'
                with np.load(recon_file) as data:
                    for mod in MODALITY_NAMES:
                        if f"{mod}_true" in data and f"{mod}_pred" in data:
                            results['recon_raw_data'][set_name][mod]['true'].append(data[f"{mod}_true"])
                            results['recon_raw_data'][set_name][mod]['pred'].append(data[f"{mod}_pred"])

    results['classification'] = pd.concat(results['classification'], ignore_index=True) if results['classification'] else pd.DataFrame()
    for mod in MODALITY_NAMES:
        results['reconstruction_metrics'][mod] = pd.concat(results['reconstruction_metrics'][mod], ignore_index=True) if results['reconstruction_metrics'][mod] else pd.DataFrame()

    for set_name in ['training', 'validation']:
        for mod in MODALITY_NAMES:
            if results['recon_raw_data'][set_name][mod]['true']:
                results['recon_raw_data'][set_name][mod]['true'] = np.concatenate(results['recon_raw_data'][set_name][mod]['true'])
                results['recon_raw_data'][set_name][mod]['pred'] = np.concatenate(results['recon_raw_data'][set_name][mod]['pred'])

    return results


def plot_average_curves_with_ci(seeds_data, curve_type, set_name, save_dir, title, source_dir):
    """Plot mean ROC or precision-recall curves and export curve coordinates."""
    cfg.VisualizationConfig.apply_style()
    plt.figure(figsize=(8, 8))
    mean_axis = np.linspace(0, 1, 100)

    all_seeds_curves = {i: [] for i in range(N_CLASSES)}
    all_seeds_aucs = {i: [] for i in range(N_CLASSES)}
    summary_records = []
    curve_point_dfs = []

    for seed_data in seeds_data:
        y_true = seed_data['true']
        y_prob = seed_data['pred_scores']
        y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))

        for i in range(N_CLASSES):
            if np.sum(y_true_bin[:, i]) == 0:
                continue

            if curve_type == 'roc':
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                all_seeds_aucs[i].append(auc(fpr, tpr))
                all_seeds_curves[i].append(np.interp(mean_axis, fpr, tpr))

            else:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                all_seeds_aucs[i].append(auc(recall, precision))

                order = np.argsort(recall)
                recall_s = recall[order]
                prec_s = precision[order]
                recall_u, idx_u = np.unique(recall_s, return_index=True)
                prec_u = prec_s[idx_u]

                interp_precision = np.interp(mean_axis, recall_u, prec_u, left=np.nan, right=np.nan)
                all_seeds_curves[i].append(interp_precision)

    for i in range(N_CLASSES):
        if not all_seeds_aucs[i]:
            continue

        mean_auc = float(np.mean(all_seeds_aucs[i]))
        std_auc = float(np.std(all_seeds_aucs[i]))

        curves = np.array(all_seeds_curves[i], dtype=float)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        curves_upper = np.minimum(mean_curve + std_curve, 1)
        curves_lower = np.maximum(mean_curve - std_curve, 0)

        label = f"{TARGET_NAMES[i]} ({'AUC' if curve_type == 'roc' else 'AP'} = {mean_auc:.3f} \u00B1 {std_auc:.3f})"
        plt.plot(mean_axis, mean_curve, color=CLASS_COLORS[i], label=label, lw=2.5, alpha=0.85)
        plt.fill_between(mean_axis, curves_lower, curves_upper, color=CLASS_COLORS[i], alpha=0.15)

        summary_records.append({
            'set': set_name,
            'curve': curve_type.upper(),
            'class': TARGET_NAMES[i],
            'mean': mean_auc,
            'std': std_auc
        })

        curve_point_dfs.append(pd.DataFrame({
            'set': set_name,
            'curve': curve_type.upper(),
            'class': TARGET_NAMES[i],
            'x': mean_axis,
            'y_mean': mean_curve,
            'y_upper': curves_upper,
            'y_lower': curves_lower
        }))

    if curve_type == 'roc':
        plt.plot([0, 1], [0, 1], color='0.5', linestyle='--', lw=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    else:
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    ax = plt.gca()
    from matplotlib.ticker import FuncFormatter as _FF
    ax.xaxis.set_major_formatter(_FF(lambda x, pos: f"{x:.1f}"))
    ax.yaxis.set_major_formatter(_FF(lambda y, pos: "" if np.isclose(y, 0.0) else f"{y:.1f}"))

    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right" if curve_type == 'roc' else "lower left")
    plt.title(title)

    outfile = Path(save_dir) / f"average_{curve_type}_curve_{set_name}.png"
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.savefig(outfile.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

    curve_points = pd.concat(curve_point_dfs, ignore_index=True) if curve_point_dfs else pd.DataFrame()
    if not curve_points.empty:
        curve_points.to_csv(Path(source_dir) / f"{curve_type}_curve_points_{set_name}.csv", index=False)

    return summary_records, curve_points

def export_classification_tables(df, set_name, output_dir, source_dir):
    if df.empty:
        return
    filtered = df[df['set'] == set_name].copy()
    if filtered.empty:
        return
    filtered.to_csv(output_dir / f"classification_metrics_raw_{set_name}.csv", index=False)
    value_cols = [c for c in filtered.columns if c not in ('set', 'fold', 'seed')]
    long_df = filtered.melt(id_vars=['set', 'fold', 'seed'], value_vars=value_cols,
                            var_name='metric', value_name='value')
    long_df.to_csv(source_dir / f"classification_metrics_detailed_{set_name}.csv", index=False)
    summary = long_df.groupby('metric')['value'].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci95_low'] = summary['mean'] - 1.96 * summary['sem']
    summary['ci95_high'] = summary['mean'] + 1.96 * summary['sem']
    summary.to_csv(output_dir / f"classification_metrics_summary_{set_name}.csv", index=False)


def export_reconstruction_tables(df, set_name, output_dir, source_dir):
    if df.empty:
        return
    filtered = df[df['set'] == set_name].copy()
    if filtered.empty:
        return
    filtered.to_csv(output_dir / f"reconstruction_metrics_raw_{set_name}.csv", index=False)
    value_cols = [c for c in filtered.columns if c not in ('set', 'fold', 'seed', 'modality')]
    long_df = filtered.melt(id_vars=['set', 'fold', 'seed', 'modality'],
                            value_vars=value_cols, var_name='metric', value_name='value')
    long_df.to_csv(source_dir / f"reconstruction_metrics_detailed_{set_name}.csv", index=False)
    summary = long_df.groupby(['modality', 'metric'])['value'].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci95_low'] = summary['mean'] - 1.96 * summary['sem']
    summary['ci95_high'] = summary['mean'] + 1.96 * summary['sem']
    summary.to_csv(output_dir / f"reconstruction_metrics_summary_{set_name}.csv", index=False)


def _fmt_rel_change(pre, post):
    """Format a pre-to-post value and its relative change."""
    try:
        if pre is None or np.isclose(pre, 0.0):
            return f"{pre:.3f} \u2192 {post:.3f} (N/A)"
        rel = (post - pre) / pre * 100.0
        sign = "+" if rel >= 0 else "-"
        rel_abs = abs(rel)
        return f"{pre:.3f} \u2192 {post:.3f} ({sign}{rel_abs:.1f}%)"
    except Exception:
        return f"{pre:.3f} \u2192 {post:.3f} (N/A)"


def compute_multiclass_calibration(y_true, probs, label, n_bins=10):
    """Calculate one-vs-rest calibration bins for each class."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    records = []
    class_order = list(sorted(cfg.VisualizationConfig.LABEL_NAMES.keys()))
    for k in class_order:
        p_k = probs[:, k]
        y_k = (y_true == k).astype(float)
        bin_idx = np.clip(np.digitize(p_k, bins) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            if not np.any(mask):
                continue
            prob_mean = float(np.mean(p_k[mask]))
            frac_pos = float(np.mean(y_k[mask]))
            n = int(np.sum(mask))
            se = float(np.sqrt(max(frac_pos * (1.0 - frac_pos), 0.0) / max(n, 1)))
            ci_low = max(0.0, frac_pos - 1.96 * se)
            ci_high = min(1.0, frac_pos + 1.96 * se)
            records.append({
                'class_index': k,
                'class_name': cfg.VisualizationConfig.LABEL_NAMES[k],
                'bin_lower': float(bins[b]),
                'bin_upper': float(bins[b + 1]),
                'prob_mean': prob_mean,
                'frac_pos': frac_pos,
                'count': n,
                'se': se,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'label': label
            })
    return pd.DataFrame.from_records(records)

def set_publication_style():
    """Apply consistent typography and line settings."""
    import seaborn as sns
    sns.set_style("ticks")
    plt.style.use("default")

    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Sizes
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Lines
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        # ---- Save ----
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

def _prob_axis_formatter():
    return FuncFormatter(lambda x, pos: f"{x:.1f}")

def _panel_label(ax, letter: str):
    """Add a panel label to an axis."""
    ax.annotate(
        letter,
        xy=(0, 1), xycoords="axes fraction",
        xytext=(-18, 18), textcoords="offset points",
        ha="left", va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False
    )

def plot_average_curve_with_ci_on_ax(ax, seeds_data, curve_type: str, title: str,
                                     class_names, class_colors, n_classes: int):
    mean_axis = np.linspace(0, 1, 100)

    for i in range(n_classes):
        curves = []
        aucs = []

        for seed_data in seeds_data:
            y_true = seed_data["true"]
            y_prob = seed_data["pred_scores"]
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            if np.sum(y_true_bin[:, i]) == 0:
                continue

            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
                curves.append(np.interp(mean_axis, fpr, tpr))
            else:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                aucs.append(auc(recall, precision))

                order = np.argsort(recall)
                recall_s = recall[order]
                prec_s = precision[order]
                recall_u, idx_u = np.unique(recall_s, return_index=True)
                prec_u = prec_s[idx_u]
                curves.append(np.interp(mean_axis, recall_u, prec_u, left=np.nan, right=np.nan))

        if not aucs:
            continue

        curves = np.array(curves, dtype=float)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        upper = np.minimum(mean_curve + std_curve, 1)
        lower = np.maximum(mean_curve - std_curve, 0)

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        metric_name = "AUC" if curve_type == "roc" else "AP"

        label = f"{class_names[i]} ({metric_name} = {mean_auc:.3f} \u00B1 {std_auc:.3f})"
        ax.plot(mean_axis, mean_curve, color=class_colors[i], lw=2.0, alpha=0.90, label=label)
        ax.fill_between(mean_axis, lower, upper, color=class_colors[i], alpha=0.18, linewidth=0)

    ax.set_title(title, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.xaxis.set_major_formatter(FuncFormatter(_one_decimal))
    ax.yaxis.set_major_formatter(FuncFormatter(_one_decimal_blank_zero))

    ax.grid(True, ls="--", lw=0.6, alpha=0.18)
    sns.despine(ax=ax)

    legend_kw = dict(
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], color="0.35", lw=1.0, ls="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.06, 0.03),
            bbox_transform=ax.transAxes,
            **legend_kw,
        )
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.02, 0.03),
            bbox_transform=ax.transAxes,
            **legend_kw,
        )

def plot_multiclass_calibration_on_ax(ax, y_true, probs, title: str,
                                      class_order, class_names, class_colors,
                                      annotation_text: str | None):
    calib_df = compute_multiclass_calibration(y_true, probs, label="After", n_bins=10)

    ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="Identity")

    for j, cls_idx in enumerate(class_order):
        sub = calib_df[calib_df["class_index"] == cls_idx].copy()
        if sub.empty:
            continue
        sub.sort_values("prob_mean", inplace=True)
        color = class_colors[j]

        ax.fill_between(
            sub["prob_mean"], sub["ci_low"], sub["ci_high"],
            color=color, alpha=0.14, linewidth=0, zorder=1
        )
        ax.plot(
            sub["prob_mean"], sub["frac_pos"],
            color=color, lw=2.0, alpha=0.95,
            label=class_names[j], zorder=2
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.xaxis.set_major_formatter(FuncFormatter(_one_decimal))
    ax.yaxis.set_major_formatter(FuncFormatter(_one_decimal_blank_zero))

    ax.grid(True, ls="--", lw=0.6, alpha=0.18)
    sns.despine(ax=ax)

    legend_kw = dict(
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        **legend_kw,
    )

    if annotation_text:
        ax.text(
            0.97, 0.06, annotation_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7,
            linespacing=1.15,
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="0.65",
                lw=0.7,
                alpha=0.68,
            ),
        )


def _add_recon_metrics_box(ax, metrics_dict):
    """Add reconstruction means and standard deviations to an axis."""
    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.3f}"

    def _with_sd(mean_str, sd):
        return f"{mean_str} \u00B1 {sd:.3f}" if isinstance(sd, (float, int)) and not np.isnan(sd) else mean_str

    text = (
        f"R\u00b2 = {_with_sd(_fmt(metrics_dict.get('r2')), metrics_dict.get('r2_sd'))}\n"
        f"MAE = {_with_sd(_fmt(metrics_dict.get('mae')), metrics_dict.get('mae_sd'))}\n"
        f"RMSE = {_with_sd(_fmt(metrics_dict.get('rmse')), metrics_dict.get('rmse_sd'))}\n"
        f"Pearson's r = {_with_sd(_fmt(metrics_dict.get('pearson_corr')), metrics_dict.get('pearson_corr_sd'))}"
    )

    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7,
        linespacing=1.15,
        bbox=dict(
            boxstyle="round,pad=0.22",
            fc="white",
            ec="0.65",
            lw=0.7,
            alpha=0.68,
        ),
    )

def plot_reconstruction_joint_in_cell(
    fig,
    cell_gs,
    y_true,
    y_pred,
    metrics_dict,
    title: str,
    point_color: str,
    ideal_line_color: str = "#BF616A",
    max_points: int = 2000,
    show_marginals: bool = True,
    top_marginal_frac: float = 0.085,
    right_marginal_frac: float = 0.068,
    marginal_gap_frac_x: float = 0.0,
    marginal_gap_frac_y: float = 0.030,
    y_labelpad: float = 0.2,
):
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if yt.size == 0:
        ax = fig.add_subplot(cell_gs)
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return {"main": ax, "top": ax, "right": None}

    if yt.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(yt.size, size=max_points, replace=False)
        yt, yp = yt[idx], yp[idx]

    resid = yt - yp

    min_val = float(np.min([yt.min(), yp.min()]))
    max_val = float(np.max([yt.max(), yp.max()]))
    pad = 0.02 * (max_val - min_val + 1e-9)
    xmin, xmax = min_val - pad, max_val + pad

    ax_main = fig.add_subplot(cell_gs)

    sns.regplot(
        x=yt, y=yp, ax=ax_main,
        ci=95, n_boot=300, truncate=False,
        scatter_kws=dict(s=10, alpha=0.35, color=point_color, edgecolors="none"),
        line_kws=dict(color=point_color, lw=2.0, alpha=0.95),
    )
    ax_main.plot([xmin, xmax], [xmin, xmax], color=ideal_line_color, lw=1.5, ls="--", label="Ideal (y=x)")
    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(xmin, xmax)

    ax_main.set_xlabel("True Values")
    ax_main.set_ylabel("Predicted Values", labelpad=y_labelpad)

    if not show_marginals:
        ax_main.set_title(title, pad=4)

    ax_main.legend(loc="lower right", frameon=False)
    ax_main.grid(True, ls="--", lw=0.6, alpha=0.18)
    sns.despine(ax=ax_main)

    _add_recon_metrics_box(ax_main, metrics_dict)

    if not show_marginals:
        return {"main": ax_main, "top": ax_main, "right": None}

    bbox = ax_main.get_position()

    top_ratio = float(np.clip(top_marginal_frac, 0.05, 0.20))
    right_ratio = float(np.clip(right_marginal_frac, 0.04, 0.20))

    gap_x = bbox.width * float(np.clip(marginal_gap_frac_x, 0.0, 0.08))
    gap_y = bbox.height * float(np.clip(marginal_gap_frac_y, 0.0, 0.12))

    fig_xmax = 0.995
    fig_ymax = 0.995

    # top marginal (True)
    top_h = bbox.height * top_ratio
    top_y0 = bbox.y1 + gap_y
    top_y1 = min(fig_ymax, top_y0 + top_h)
    top_h_eff = max(0.0, top_y1 - top_y0)

    ax_top = None
    if top_h_eff > 0.0:
        ax_top = fig.add_axes([bbox.x0, top_y0, bbox.width, top_h_eff])
        sns.histplot(
            x=yt, ax=ax_top,
            bins=25, stat="density", kde=True,
            color=point_color, alpha=0.28,
            edgecolor="white", linewidth=0.3,
            line_kws=dict(lw=1.2),
        )
        ax_top.set_title(title, pad=6)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.set_xlabel(""); ax_top.set_ylabel("")
        ax_top.grid(False)
        sns.despine(ax=ax_top, left=True, bottom=True)
    else:
        ax_main.set_title(title, pad=4)

    # right marginal (Residual)
    right_w = bbox.width * right_ratio
    right_x0 = bbox.x1 + gap_x
    right_x1 = min(fig_xmax, right_x0 + right_w)
    right_w_eff = max(0.0, right_x1 - right_x0)

    ax_right = None
    if right_w_eff > 0.0:
        ax_right = fig.add_axes([right_x0, bbox.y0, right_w_eff, bbox.height])
        sns.histplot(
            y=resid, ax=ax_right,
            bins=25, stat="density", kde=True,
            color=point_color, alpha=0.22,
            edgecolor="white", linewidth=0.3,
            line_kws=dict(lw=1.2),
        )
        ax_right.axhline(0.0, color="0.35", lw=1.0, ls="--", alpha=0.8)
        ax_right.set_xticks([]); ax_right.set_yticks([])
        ax_right.set_xlabel(""); ax_right.set_ylabel("")
        ax_right.grid(False)
        sns.despine(ax=ax_right, left=True, bottom=True)

    top_for_label = ax_top if ax_top is not None else ax_main
    return {"main": ax_main, "top": top_for_label, "right": ax_right}


def plot_model_performance_summary(results, set_name: str, output_dir: Path):
    set_publication_style()

    fig = plt.figure(figsize=(8.27, 5.845))  # A4 width × half A4 height
    gs = fig.add_gridspec(2, 3, wspace=0.26, hspace=0.50)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.11)

    target_names = [cfg.VisualizationConfig.LABEL_NAMES[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())]
    class_colors = [cfg.VisualizationConfig.LABEL_COLORS[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())]
    n_classes = cfg.NUM_CLASSES

    seeds_data = results["oof_data"][set_name]
    if not seeds_data:
        raise RuntimeError(f"No OOF data found for set='{set_name}'")

    # a: PR
    ax_a = fig.add_subplot(gs[0, 0])
    plot_average_curve_with_ci_on_ax(
        ax_a, seeds_data, curve_type="pr",
        title="Average PR Curve",
        class_names=target_names, class_colors=class_colors, n_classes=n_classes
    )
    _panel_label(ax_a, "a")

    # b: ROC
    ax_b = fig.add_subplot(gs[0, 1])
    plot_average_curve_with_ci_on_ax(
        ax_b, seeds_data, curve_type="roc",
        title="Average ROC Curve",
        class_names=target_names, class_colors=class_colors, n_classes=n_classes
    )
    _panel_label(ax_b, "b")

    # c: Calibration
    ax_c = fig.add_subplot(gs[0, 2])
    all_true = np.concatenate([d["true"] for d in seeds_data])
    all_scores = np.concatenate([d["pred_scores"] for d in seeds_data])
    scaled_probs, temperature = temperature_scale_probabilities(all_scores, all_true)

    ece_pre = expected_calibration_error(all_true, all_scores, n_bins=10)
    ece_post = expected_calibration_error(all_true, scaled_probs, n_bins=10)
    brier_pre = (multi_class_brier_score(all_true, all_scores) / n_classes)
    brier_post = (multi_class_brier_score(all_true, scaled_probs) / n_classes)

    annot = (
        f"T = {temperature:.2f}\n"
        f"ECE: {_fmt_rel_change(ece_pre, ece_post)}\n"
        f"Brier: {_fmt_rel_change(brier_pre, brier_post)}"
    )
    plot_multiclass_calibration_on_ax(
        ax_c, all_true, scaled_probs,
        title="Calibration Curve",
        class_order=list(sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())),
        class_names=target_names,
        class_colors=class_colors,
        annotation_text=annot
    )
    _panel_label(ax_c, "c")

    # d/e/f: reconstruction
    recon_colors = getattr(cfg.VisualizationConfig, "RECON_SCATTER_COLORS", {
        "gly": "#6B5B95", "mass": "#2A9D8F", "rna": "#E07A5F"
    })

    RIGHT_MARGINAL_FRAC = 0.068
    TOP_MARGINAL_FRAC = 0.085

    for col, (mod, letter, title) in enumerate([
        ("gly", "d", "Glycan Reconstruction"),
        ("mass", "e", "Mass Reconstruction"),
        ("rna", "f", "RNA Reconstruction"),
    ]):
        cell = gs[1, col]
        y_true = results["recon_raw_data"][set_name][mod].get("true")
        y_pred = results["recon_raw_data"][set_name][mod].get("pred")

        if y_true is None or y_pred is None or np.size(y_true) == 0:
            ax = fig.add_subplot(cell)
            ax.set_title(title)
            ax.text(0.5, 0.5, "No reconstruction data", ha="center", va="center", transform=ax.transAxes)
            _panel_label(ax, letter)
            continue

        df_fold = results["reconstruction_metrics"][mod]
        metric_cols = ["r2", "mae", "rmse", "pearson_corr"]

        if df_fold is None or df_fold.empty:
            raise RuntimeError(
                f"No per-fold reconstruction metrics found for {mod}."
            )

        df_fold = df_fold.loc[df_fold["set"].eq(set_name)].copy()

        missing_cols = [
            metric for metric in metric_cols
            if metric not in df_fold.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Missing per-fold reconstruction metric(s) "
                f"for {mod}: {missing_cols}"
            )

        df_fold[metric_cols] = df_fold[metric_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )
        df_fold = df_fold.dropna(subset=metric_cols)

        expected_n = (
            len(SEEDS)
            * cfg.N_SPLITS_K_FOLD
            * cfg.N_REPEATS
        )

        if len(df_fold) != expected_n:
            logging.warning(
                "Reconstruction panel %s: expected %d seed-fold rows for %s, found %d.",
                letter,
                expected_n,
                mod,
                len(df_fold)
            )

        if df_fold.empty:
            raise RuntimeError(
                f"No complete per-fold reconstruction metrics "
                f"for {mod}, set={set_name}."
            )

        metrics = {}
        for metric in metric_cols:
            metrics[metric] = float(df_fold[metric].mean())
            metrics[f"{metric}_sd"] = (
                float(df_fold[metric].std(ddof=1))
                if len(df_fold) > 1
                else np.nan
            )

        y_labelpad = -6.0 if mod == "rna" else 0.2

        axes = plot_reconstruction_joint_in_cell(
            fig, cell, y_true, y_pred, metrics_dict=metrics,
            title=title,
            point_color=recon_colors.get(mod, "#6B5B95"),
            ideal_line_color=cfg.VisualizationConfig.RECON_IDEAL_LINE_COLOR,
            show_marginals=True,
            top_marginal_frac=TOP_MARGINAL_FRAC,
            right_marginal_frac=RIGHT_MARGINAL_FRAC,
            marginal_gap_frac_x=0.0,
            marginal_gap_frac_y=0.030,
            y_labelpad=y_labelpad,
        )

        _panel_label(axes["top"], letter)

    out_png = Path(output_dir) / f"model_performance_summary_{set_name}.png"
    out_pdf = Path(output_dir) / f"model_performance_summary_{set_name}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def generate_final_summary(results):
    """Generate summary tables and figures from fold-level results."""
    logging.info("\n" + "=" * 20 + " Generating Final Summary Report " + "=" * 20)

    roc_summary_all, pr_summary_all, curve_point_records = [], [], []
    calibration_metrics_records = []
    agg_reconstruction_records = []

    for set_name in ['training', 'validation']:
        set_name_cap = set_name.capitalize()
        logging.info(f"\n--- Analyzing {set_name_cap} Set ---")

        if not results['classification'].empty:
            export_classification_tables(results['classification'], set_name, OUTPUT_DIR, SOURCE_DATA_DIR)
            set_cls_df = results['classification'][results['classification']['set'] == set_name]
            if not set_cls_df.empty:
                logging.info(f"\n--- Overall Classification Metrics ({set_name_cap}) ---")
                summary_stats = set_cls_df.drop(columns=['set', 'fold', 'seed']).describe().loc[['mean', 'std']]
                print(summary_stats.to_string(float_format="%.4f"))
                summary_stats.to_csv(OUTPUT_DIR / f"classification_summary_stats_{set_name}.csv")

        if results['oof_data'][set_name]:
            all_true = np.concatenate([d['true'] for d in results['oof_data'][set_name]])
            all_scores = np.concatenate([d['pred_scores'] for d in results['oof_data'][set_name]])
            all_pred = np.argmax(all_scores, axis=1)

            logging.info(f"\n--- Overall Ensembled Classification Report ({set_name_cap} OOF) ---")
            report = classification_report(all_true, all_pred, target_names=TARGET_NAMES, digits=4, zero_division=0)
            print(report)
            (OUTPUT_DIR / f"classification_report_ensembled_{set_name}.txt").write_text(report)

            cm = confusion_matrix(all_true, all_pred, labels=range(N_CLASSES))
            cm_df = pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES)
            cm_df.to_csv(SOURCE_DATA_DIR / f"confusion_matrix_{set_name}.csv")

            plot_confusion_matrix(cm, TARGET_NAMES, "Confusion Matrix",
                                  OUTPUT_DIR / f"confusion_matrix_ensembled_{set_name}.png")

            plot_confusion_matrix(cm, TARGET_NAMES, "Normalized Confusion Matrix",
                                  OUTPUT_DIR / f"confusion_matrix_ensembled_normalized_{set_name}.png", normalize=True)

            plot_confusion_matrix(cm, TARGET_NAMES, "Normalized Confusion Matrix",
                                  OUTPUT_DIR / f"confusion_matrix_ensembled_normalized_{set_name}.pdf", normalize=True)

            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            pd.DataFrame(cm_norm, index=TARGET_NAMES, columns=TARGET_NAMES).to_csv(
                SOURCE_DATA_DIR / f"confusion_matrix_normalized_{set_name}.csv")

            roc_summary, roc_points = plot_average_curves_with_ci(
                results['oof_data'][set_name], 'roc', set_name, OUTPUT_DIR, "Average ROC Curve", SOURCE_DATA_DIR)
            pr_summary, pr_points = plot_average_curves_with_ci(
                results['oof_data'][set_name], 'pr', set_name, OUTPUT_DIR, "Average PR Curve", SOURCE_DATA_DIR)
            roc_summary_all.extend(roc_summary)
            pr_summary_all.extend(pr_summary)
            if not roc_points.empty:
                curve_point_records.append(roc_points)
            if not pr_points.empty:
                curve_point_records.append(pr_points)
            logging.info(f"Average ROC/PR curves for {set_name_cap} saved to {OUTPUT_DIR}")

            ece_pre = expected_calibration_error(all_true, all_scores, n_bins=10)
            scaled_probs, temperature = temperature_scale_probabilities(all_scores, all_true)
            ece_post = expected_calibration_error(all_true, scaled_probs, n_bins=10)

            brier_pre_raw = multi_class_brier_score(all_true, all_scores)
            brier_post_raw = multi_class_brier_score(all_true, scaled_probs)
            brier_pre = brier_pre_raw / N_CLASSES          # normalized
            brier_post = brier_post_raw / N_CLASSES        # normalized

            calib_pre_df = compute_multiclass_calibration(all_true, all_scores, label='Before', n_bins=10)
            calib_post_df = compute_multiclass_calibration(all_true, scaled_probs, label='After', n_bins=10)

            calib_pre_df.to_csv(SOURCE_DATA_DIR / f"calibration_multiclass_before_{set_name}.csv", index=False)
            calib_post_df.to_csv(SOURCE_DATA_DIR / f"calibration_multiclass_after_{set_name}.csv", index=False)

            annot_post = (
                f"T = {temperature:.2f}\n"
                f"ECE: {_fmt_rel_change(ece_pre, ece_post)}\n"
                f"Brier: {_fmt_rel_change(brier_pre, brier_post)}"
            )
            plot_multiclass_calibration(
                calib_post_df,
                title_prefix="Calibration Curve (Temperature-Scaled)",
                save_path=OUTPUT_DIR / f"calibration_multiclass_after_{set_name}.png",
                class_order=list(sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())),
                class_names=[cfg.VisualizationConfig.LABEL_NAMES[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())],
                class_colors=[cfg.VisualizationConfig.LABEL_COLORS[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())],
                annotation_text=annot_post
            )

            annot_pre = (
                f"ECE: {ece_pre:.3f}\n"
                f"Brier: {brier_pre:.3f}"
            )
            plot_multiclass_calibration(
                calib_pre_df,
                title_prefix="Calibration Curve (Uncalibrated)",
                save_path=OUTPUT_DIR / f"calibration_multiclass_before_{set_name}.png",
                class_order=list(sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())),
                class_names=[cfg.VisualizationConfig.LABEL_NAMES[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())],
                class_colors=[cfg.VisualizationConfig.LABEL_COLORS[i] for i in sorted(cfg.VisualizationConfig.LABEL_NAMES.keys())],
                annotation_text=annot_pre
            )

            ts_points = plot_temperature_scaling_effect(all_scores, scaled_probs, temperature, N_CLASSES,
                                                        "Temperature Scaling Mapping",
                                                        OUTPUT_DIR / f"ts_effect_{set_name}.png")
            ts_points.to_csv(SOURCE_DATA_DIR / f"ts_mapping_points_{set_name}.csv", index=False)

            calibration_metrics_records.append({
                'set': set_name,
                'temperature': temperature,
                'ece_pre': ece_pre,
                'ece_post': ece_post,
                'brier_pre': brier_pre,               # normalized
                'brier_post': brier_post,             # normalized
                'brier_pre_raw': brier_pre_raw,
                'brier_post_raw': brier_post_raw
            })

            pd.DataFrame({
                'set': set_name,
                'sample_index': np.arange(len(all_true)),
                'true_label': all_true,
                'pred_label': all_pred,
                'confidence': all_scores.max(axis=1),
                'confidence_post_ts': scaled_probs.max(axis=1)
            }).to_csv(SOURCE_DATA_DIR / f"oof_predictions_summary_{set_name}.csv", index=False)

        metric_cols = ["r2", "mae", "rmse", "pearson_corr"]
        expected_n = (
            len(SEEDS)
            * cfg.N_SPLITS_K_FOLD
            * cfg.N_REPEATS
        )

        recon_frames = [
            results["reconstruction_metrics"][mod]
            for mod in MODALITY_NAMES
            if results["reconstruction_metrics"][mod] is not None
            and not results["reconstruction_metrics"][mod].empty
        ]

        if recon_frames:
            all_recon_df = pd.concat(
                recon_frames,
                ignore_index=True
            )
            export_reconstruction_tables(
                all_recon_df,
                set_name,
                OUTPUT_DIR,
                SOURCE_DATA_DIR
            )

        per_mod_sd = {mod: {} for mod in MODALITY_NAMES}

        for mod in MODALITY_NAMES:
            recon_df = results["reconstruction_metrics"][mod]

            if recon_df is None or recon_df.empty:
                continue

            set_recon_df = recon_df.loc[
                recon_df["set"].eq(set_name)
            ].copy()

            missing_cols = [
                metric for metric in metric_cols
                if metric not in set_recon_df.columns
            ]
            if missing_cols:
                raise KeyError(
                    f"Missing per-fold reconstruction metric(s) "
                    f"for {mod}: {missing_cols}"
                )

            set_recon_df[metric_cols] = set_recon_df[
                metric_cols
            ].apply(pd.to_numeric, errors="coerce")

            set_recon_df = set_recon_df.dropna(
                subset=metric_cols
            )

            if len(set_recon_df) != expected_n:
                logging.warning(
                    "Expected %d seed-fold rows for %s/%s, found %d.",
                    expected_n,
                    set_name,
                    mod,
                    len(set_recon_df)
                )

            for metric in metric_cols:
                mean_value = float(
                    set_recon_df[metric].mean()
                )

                per_mod_sd[mod][metric] = mean_value
                per_mod_sd[mod][f"{metric}_mean"] = mean_value
                per_mod_sd[mod][f"{metric}_sd"] = float(
                    set_recon_df[metric].std(ddof=1)
                )

        overall_metrics_by_mod = {}
        for mod in MODALITY_NAMES:
            all_true = results['recon_raw_data'][set_name][mod].get('true')
            all_pred = results['recon_raw_data'][set_name][mod].get('pred')

            if all_true is not None and all_pred is not None and all_true.size > 1:
                metrics = {
                    'r2': r2_score(all_true, all_pred),
                    'mae': mean_absolute_error(all_true, all_pred),
                    'rmse': np.sqrt(mean_squared_error(all_true, all_pred)),
                    'pearson_corr': safe_pearsonr(all_true.flatten(), all_pred.flatten())[0]
                }
                metrics.update(per_mod_sd.get(mod, {}))
                overall_metrics_by_mod[mod] = metrics

                title_scatter = f"{cfg.VisualizationConfig.MODALITY_NAMES.get(mod, mod.upper())} Reconstruction"
                plot_reconstruction_scatter(all_true, all_pred, metrics, mod, title_scatter,
                                            OUTPUT_DIR / f"recon_scatter_{mod}_{set_name}.png")
                residuals = all_true - all_pred
                title_resid = f"{cfg.VisualizationConfig.MODALITY_NAMES.get(mod, mod.upper())} Residuals"
                plot_reconstruction_residuals(all_pred, residuals, mod, title_resid,
                                              OUTPUT_DIR / f"recon_residuals_{mod}_{set_name}.png", metrics=metrics)

        if overall_metrics_by_mod:
            plot_reconstruction_metrics_bars(overall_metrics_by_mod, "Reconstruction Performance",
                                             OUTPUT_DIR / f"recon_metrics_bars_{set_name}.png")
            logging.info(f"Overall reconstruction plots for {set_name_cap} saved to {OUTPUT_DIR}")
            agg_reconstruction_records.extend([{
                'set': set_name,
                'modality': mod,
                **metrics
            } for mod, metrics in overall_metrics_by_mod.items()])

    if roc_summary_all:
        pd.DataFrame(roc_summary_all).to_csv(SOURCE_DATA_DIR / "roc_summary_statistics.csv", index=False)
    if pr_summary_all:
        pd.DataFrame(pr_summary_all).to_csv(SOURCE_DATA_DIR / "pr_summary_statistics.csv", index=False)
    if curve_point_records:
        pd.concat(curve_point_records, ignore_index=True).to_csv(
            SOURCE_DATA_DIR / "curve_points_all_sets.csv", index=False)
    if calibration_metrics_records:
        pd.DataFrame(calibration_metrics_records).to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False)
    if agg_reconstruction_records:
        pd.DataFrame(agg_reconstruction_records).to_csv(
            SOURCE_DATA_DIR / "reconstruction_aggregated_metrics.csv", index=False)

def main():
    """Collect and summarize the primary model results."""
    if OUTPUT_DIR.exists():
        logging.warning(f"Removing existing analysis directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    SOURCE_DATA_DIR.mkdir(exist_ok=True)

    logging.info("=== SLEmodel Results Collection and Analysis ===")

    results = collect_slemodel_results()

    if results['classification'].empty and not results['oof_data']['validation']:
        logging.error("No results found for any seed. Run 'slemodel-train' first.")
        return

    generate_final_summary(results)
    plot_model_performance_summary(results, set_name="validation", output_dir=OUTPUT_DIR)

    logging.info("\n" + "=" * 60)
    logging.info(f"Analysis complete. All summary reports and plots are saved in: {OUTPUT_DIR.absolute()}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
