#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_kegg_ora_biological_direction.py
=====================================
KEGG over-representation analysis for mass-spectrometry features, with pathway
direction determined from abundance fold changes relative to Control.
"""

from __future__ import annotations

import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from slemodel import config as cfg

from .enrichment_plot_utils import set_publication_style, save_publication_figure
set_publication_style()

# ============================================================================
# ============================================================================
@dataclass
class ORAConfig:
    min_pathway_size: int = 3
    max_pathway_size: int = 500
    min_overlap: int = 1
    fdr_threshold: float = 0.10
    p_threshold: float = 0.05
    require_both: bool = True
    reference_class: str = "Control"
    log2fc_threshold: float = 0.1
    exclude_large_pathways: List[str] = None

    def __post_init__(self):
        if self.exclude_large_pathways is None:
            self.exclude_large_pathways = [
                "Metabolic pathways",
                "Biosynthesis of secondary metabolites",
                "Microbial metabolism in diverse environments"
            ]

# ============================================================================
# ============================================================================

def load_enhanced_annotation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['final_category'].isin(['ANNOTATED', 'ISOTOPE'])].copy()
    return df


def load_shap_results(npz_path: Path):
    dat = np.load(npz_path, allow_pickle=True)
    return dat["phi_stack"], dat["y_true"], dat["feature_names"].tolist()


def load_feature_to_cids(json_path: Path) -> Dict[str, List[str]]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {feat: [item["cid"] for item in lst if "cid" in item]
            for feat, lst in data.items()}


def load_pathway_to_cids(json_path: Path) -> Dict[str, List[str]]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_mass_abundance_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the mass-spectrometry abundance matrix and group labels.
    """
    df = pd.read_csv(csv_path)

    if 'ID' in df.columns:
        df = df.set_index('ID')

    y = df['Group'].copy()
    X = df.drop(columns=['Group'], errors='ignore')

    mass_cols = [c for c in X.columns if c.startswith('mz_') or 'mz_' in c]
    if mass_cols:
        X = X[mass_cols]

    return X, y

def _apply_rcparams_frame(ax):
    """
    Apply axis frame style using current matplotlib rcParams.
    This keeps the biological-interpretation plots visually consistent.
    """
    edge_color = plt.rcParams.get("axes.edgecolor", "#333333")
    edge_lw = float(plt.rcParams.get("axes.linewidth", 0.8))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(edge_lw)
        spine.set_color(edge_color)

    ax.tick_params(width=edge_lw, colors=edge_color)
    return edge_color, edge_lw

# ============================================================================
# ============================================================================

def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()


def run_ora_single_class(
    query_cids: set,
    pathway_to_cids: Dict[str, List[str]],
    background_cids: set,
    config: ORAConfig
) -> List[Dict]:
    results = []

    M = len(background_cids)
    query_in_bg = query_cids & background_cids
    N = len(query_in_bg)

    if M == 0 or N == 0:
        return results

    for pathway, path_cids in pathway_to_cids.items():
        if any(excl in pathway for excl in config.exclude_large_pathways):
            continue

        path_in_bg = set(path_cids) & background_cids
        n = len(path_in_bg)

        if n < config.min_pathway_size or n > config.max_pathway_size:
            continue

        overlap = query_in_bg & path_in_bg
        k = len(overlap)

        if k < config.min_overlap:
            continue

        a, b, c, d = k, N - k, n - k, M - n - N + k

        if any(x < 0 for x in [a, b, c, d]):
            continue

        try:
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
        except:
            odds_ratio, p_value = 1.0, 1.0

        expected = (n * N) / M if M > 0 else 0
        fold_enrichment = k / expected if expected > 0 else 0

        results.append({
            "pathway": pathway,
            "k_overlap": k,
            "pathway_size": n,
            "query_size": N,
            "background_size": M,
            "expected": round(expected, 3),
            "fold_enrichment": round(fold_enrichment, 3),
            "odds_ratio": round(odds_ratio, 3) if np.isfinite(odds_ratio) else float('inf'),
            "p_fisher": p_value,
            "overlap_cids": ",".join(sorted(overlap)) if overlap else ""
        })

    if results:
        pvals = [r["p_fisher"] for r in results]
        fdrs = bh_fdr(pvals)
        for r, fdr in zip(results, fdrs):
            r["fdr"] = fdr

    results.sort(key=lambda x: x["p_fisher"])
    return results


# ============================================================================
# ============================================================================

def build_cid_to_features(feature_to_cids: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Build a reverse mapping from KEGG compound identifiers to features."""
    cid_to_features = defaultdict(list)
    for feat, cids in feature_to_cids.items():
        for cid in cids:
            cid_to_features[cid].append(feat)
    return dict(cid_to_features)


def feature_name_to_column(feature_name: str) -> str:
    """
    Convert prefixed feature names to abundance-matrix column names.
    """
    if feature_name.startswith("mass::"):
        return feature_name.replace("mass::", "")
    return feature_name


def calculate_pathway_log2fc(
    overlap_cids: List[str],
    cid_to_features: Dict[str, List[str]],
    X_abundance: pd.DataFrame,
    y_labels: pd.Series,
    target_class: str,
    reference_class: str = "Control",
    log2fc_threshold: float = 0.1
) -> Tuple[float, str, Dict]:
    """
    Calculate pathway direction from the mean feature-level log2 fold change
    between the target and reference groups.
    """
    features_in_pathway = []
    for cid in overlap_cids:
        if cid in cid_to_features:
            for feat in cid_to_features[cid]:
                col_name = feature_name_to_column(feat)
                if col_name in X_abundance.columns:
                    features_in_pathway.append(col_name)

    features_in_pathway = list(set(features_in_pathway))

    if not features_in_pathway:
        return 0.0, 'neutral', {}

    target_mask = (y_labels == target_class)
    ref_mask = (y_labels == reference_class)

    if target_mask.sum() == 0 or ref_mask.sum() == 0:
        return 0.0, 'neutral', {}

    log2fcs = []
    details = {}

    for feat in features_in_pathway:
        target_vals = X_abundance.loc[target_mask, feat].values
        ref_vals = X_abundance.loc[ref_mask, feat].values

        target_mean = np.mean(target_vals)
        ref_mean = np.mean(ref_vals)

        pseudo = 1e-10
        if target_mean <= 0:
            target_mean = pseudo
        if ref_mean <= 0:
            ref_mean = pseudo

        log2fc = np.log2(target_mean / ref_mean)
        log2fcs.append(log2fc)
        details[feat] = {
            'log2fc': log2fc,
            'target_mean': target_mean,
            'ref_mean': ref_mean
        }

    mean_log2fc = float(np.mean(log2fcs))

    if mean_log2fc > log2fc_threshold:
        direction = 'up'
    elif mean_log2fc < -log2fc_threshold:
        direction = 'down'
    else:
        direction = 'neutral'

    return mean_log2fc, direction, details


def add_biological_direction(
    results_df: pd.DataFrame,
    feature_to_cids: Dict[str, List[str]],
    X_abundance: pd.DataFrame,
    y_labels: pd.Series,
    config: ORAConfig
) -> pd.DataFrame:
    """Add abundance-based direction estimates to the ORA results."""

    cid_to_features = build_cid_to_features(feature_to_cids)

    log2fcs = []
    directions = []
    n_up_metabolites = []
    n_down_metabolites = []

    for _, row in results_df.iterrows():
        target_class = row['class']

        if row['overlap_cids']:
            overlap_cids = row['overlap_cids'].split(',')
        else:
            overlap_cids = []

        mean_log2fc, direction, details = calculate_pathway_log2fc(
            overlap_cids=overlap_cids,
            cid_to_features=cid_to_features,
            X_abundance=X_abundance,
            y_labels=y_labels,
            target_class=target_class,
            reference_class=config.reference_class,
            log2fc_threshold=config.log2fc_threshold
        )

        log2fcs.append(mean_log2fc)
        directions.append(direction)

        n_up = sum(1 for d in details.values() if d['log2fc'] > config.log2fc_threshold)
        n_down = sum(1 for d in details.values() if d['log2fc'] < -config.log2fc_threshold)
        n_up_metabolites.append(n_up)
        n_down_metabolites.append(n_down)

    results_df = results_df.copy()
    results_df['log2fc'] = log2fcs
    results_df['direction'] = directions
    results_df['n_up_metabolites'] = n_up_metabolites
    results_df['n_down_metabolites'] = n_down_metabolites

    results_df['signed_fe'] = results_df.apply(
        lambda r: r['fold_enrichment'] if r['direction'] != 'down' else -r['fold_enrichment'],
        axis=1
    )

    return results_df


# ============================================================================
# ============================================================================

def run_classwise_ora_with_biological_direction(
    phi_stack: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    annotated_features: pd.DataFrame,
    feature_to_cids: Dict[str, List[str]],
    pathway_to_cids: Dict[str, List[str]],
    X_abundance: pd.DataFrame,
    y_abundance_labels: pd.Series,
    config: ORAConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Run ORA and annotate pathways with abundance-based direction."""

    all_results = []
    debug_info = {"classes": {}}

    mass_idx = [i for i, fn in enumerate(feature_names) if fn.startswith("mass::")]

    if not mass_idx:
        return pd.DataFrame(), debug_info

    annotated_set = set(annotated_features['feature'].tolist())

    background_cids = set()
    for cids in pathway_to_cids.values():
        background_cids.update(cids)

    all_mapped_cids = set()
    for feat in annotated_set:
        if feat in feature_to_cids:
            all_mapped_cids.update(feature_to_cids[feat])

    debug_info['total_background'] = len(background_cids)
    debug_info['all_mapped_cids'] = len(all_mapped_cids)
    debug_info['reference_class'] = config.reference_class

    print(f"\n[ORA] Background: {len(background_cids)} CIDs")
    print(f"[ORA] Mapped CIDs: {len(all_mapped_cids)}")
    print(f"[ORA] Reference class for FC: {config.reference_class}")

    C = phi_stack.shape[2]

    for c in range(C):
        class_name = cfg.VisualizationConfig.LABEL_NAMES.get(c, str(c))
        class_mask = (y_true == c)

        if class_mask.sum() == 0:
            continue

        print(f"\n[ORA] Class: {class_name}")

        phi_class = phi_stack[class_mask, :, c]
        signed_mean = np.mean(phi_class, axis=0)

        positive_features = []
        negative_features = []

        for local_i, global_i in enumerate(mass_idx):
            feat = feature_names[global_i]
            if feat in annotated_set:
                val = signed_mean[global_i]
                if val > 0:
                    positive_features.append((feat, val))
                else:
                    negative_features.append((feat, abs(val)))

        positive_features.sort(key=lambda x: x[1], reverse=True)

        if len(positive_features) >= 3:
            selected_features = [f for f, _ in positive_features]
            selection_method = "positive_shap"
        else:
            abs_importance = {}
            for local_i, global_i in enumerate(mass_idx):
                feat = feature_names[global_i]
                if feat in annotated_set:
                    abs_importance[feat] = float(np.mean(np.abs(phi_class[:, global_i])))
            sorted_feats = sorted(abs_importance.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in sorted_feats]
            selection_method = "abs_shap"

        print(f"  Selected features: {len(selected_features)} ({selection_method})")

        query_cids = set()
        for feat in selected_features:
            if feat in feature_to_cids:
                query_cids.update(feature_to_cids[feat])

        debug_info['classes'][class_name] = {
            'n_positive': len(positive_features),
            'n_selected': len(selected_features),
            'n_cids': len(query_cids),
            'selection_method': selection_method
        }

        class_results = run_ora_single_class(
            query_cids=query_cids,
            pathway_to_cids=pathway_to_cids,
            background_cids=background_cids,
            config=config
        )

        for r in class_results:
            r["class"] = class_name
            r["n_input_features"] = len(selected_features)

        all_results.extend(class_results)

        sig = len([r for r in class_results if r["fdr"] < config.fdr_threshold])
        print(f"  FDR < {config.fdr_threshold}: {sig}")

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        print(f"\n[ORA] Calculating biological direction (vs {config.reference_class})...")
        results_df = add_biological_direction(
            results_df,
            feature_to_cids,
            X_abundance,
            y_abundance_labels,
            config
        )

    return results_df, debug_info

# ============================================================================
# ============================================================================
def plot_ora_dotplot_biological(
    df: pd.DataFrame,
    save_path: Path,
    top_n: int = 12,
    fdr_threshold: float = 0.10,
    p_threshold: float = 0.05,
    min_overlap: int = 2,
    require_both: bool = True,
    reference_class: str = "Control",
    exclude_reference: bool = True,
    selected_pathways: list = None,
) -> bool:
    """
    Biological-direction ORA dotplot

    Tick labels and colorbar text retain the configured text color, while the
    axes and FDR legend use a shared frame style.
    """
    from .enrichment_plot_utils import set_publication_style
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    import textwrap

    # Apply the shared biological-interpretation style.
    set_publication_style()

    # Keep only size tweaks here; DO NOT override color-related rcParams.
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
    })

    if df is None or df.empty:
        warnings.warn("No data to plot")
        return False

    # Read canonical style tokens from rcParams.
    edge_color = plt.rcParams.get("axes.edgecolor", "#333333")
    edge_lw = float(plt.rcParams.get("axes.linewidth", 0.8))
    text_color = plt.rcParams.get("text.color", "#111111")

    def _apply_rcparams_frame(ax):
        # Only set spines; DO NOT force tick label colors (that caused fading).
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(edge_lw)
            spine.set_color(edge_color)
        # Make tick marks consistent thickness; keep default label color
        ax.tick_params(width=edge_lw)
        return

    MORANDI_CMAP = LinearSegmentedColormap.from_list(
        "morandi_pink_blue", ["#4A6A91", "#A8B8D0", "#F0EBE8", "#E8C7C3", "#B36A6F"]
    )

    df = df.copy()
    if exclude_reference and ("class" in df.columns):
        df = df[df["class"] != reference_class].copy()
    if df.empty:
        warnings.warn("No non-reference rows available to plot")
        return False

    # -------- pathway selection --------
    if require_both:
        eligible = df[
            (df["fdr"] < fdr_threshold)
            & (df["p_fisher"] < p_threshold)
            & (df["k_overlap"] >= min_overlap)
        ].copy()
    else:
        eligible = df[
            (df["fdr"] < fdr_threshold)
            & (df["k_overlap"] >= min_overlap)
        ].copy()

    if selected_pathways is not None and len(selected_pathways) > 0:
        matched_pathways = []
        for sel in selected_pathways:
            sel_lower = str(sel).lower()
            for pw in eligible["pathway"].unique():
                pw_lower = str(pw).lower()
                if sel_lower in pw_lower or pw_lower in sel_lower:
                    matched_pathways.append(pw)
        matched_pathways = list(set(matched_pathways))
        df_plot = eligible[eligible["pathway"].isin(matched_pathways)].copy()
    else:
        df_plot = eligible.copy()

    if df_plot.empty:
        warnings.warn(
            f"No pathways met FDR < {fdr_threshold:g}"
            + (
                f" and P < {p_threshold:g}"
                if require_both else ""
            )
        )
        return False
    # -------- pathway name cleanup/wrap --------
    def clean_and_wrap_name(s, max_width: int = 25):
        s = str(s).split(" - ")[0]
        replacements = {
            "Valine, leucine and isoleucine": "BCAA",
            "biosynthesis": "biosynth.",
            "degradation": "degrad.",
            "metabolism": "metab.",
            "Pantothenate and CoA": "Pantothenate/CoA",
            "Central carbon metabolism in cancer": "Central carbon metab.\n(cancer)",
            "Phenylalanine, tyrosine and tryptophan": "Phe/Tyr/Trp",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        if len(s) > max_width and "\n" not in s:
            s = "\n".join(textwrap.wrap(s, width=max_width))
        return s

    df_plot["pathway_clean"] = df_plot["pathway"].apply(clean_and_wrap_name)

    pathway_order = (
        df_plot.groupby("pathway_clean")["p_fisher"]
        .min()
        .sort_values()
        .head(top_n)
        .index.tolist()
    )
    df_plot = df_plot[df_plot["pathway_clean"].isin(pathway_order)].copy()
    if df_plot.empty:
        warnings.warn("No pathways remained after pathway selection")
        return False

    classes = ["Active", "Stable"]
    classes = [c for c in classes if c in df_plot["class"].unique()]
    if not classes:
        warnings.warn("No Active or Stable rows available to plot")
        return False

    # -------- layout --------
    fig_h = max(4.5, len(pathway_order) * 0.42 + 0.8)
    fig_w = 5.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[0.46, 0.54],
        wspace=0.12,
        left=0.36, right=0.98, top=0.88, bottom=0.08
    )

    ax = fig.add_subplot(gs[0, 0])

    x_gap = 0.24
    x_pos = {cls: i * x_gap for i, cls in enumerate(classes)}
    y_pos = {pw: i for i, pw in enumerate(pathway_order)}

    df_plot["nlog_fdr"] = -np.log10(df_plot["fdr"].clip(lower=1e-10))
    max_nlog = max(2, float(df_plot["nlog_fdr"].max()))

    max_log2fc = max(0.5, float(df_plot["log2fc"].abs().max()))
    norm = plt.Normalize(vmin=-max_log2fc, vmax=max_log2fc)

    # -------- draw points --------
    for _, row in df_plot.iterrows():
        x = x_pos.get(row["class"])
        y = y_pos.get(row["pathway_clean"])
        if x is None or y is None:
            continue

        size = 70 + 250 * (row["nlog_fdr"] / max_nlog)
        color = MORANDI_CMAP(norm(row["log2fc"]))

        # Use one opaque Line2D circle per result.  This preserves the same
        # area mapping while preventing translucent scatter PathCollections
        # from becoming white/double-edged compound paths in Illustrator.
        ax.plot(
            [x], [y],
            marker="o",
            linestyle="None",
            markersize=float(np.sqrt(size)),
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            alpha=1.0,
            zorder=3,
        )

    # -------- axes formatting --------
    xticks = [x_pos[c] for c in classes]
    ax.set_xticks(xticks)
    contrast_labels = {
        "Active": "Active −\nControl",
        "Stable": "Stable −\nControl",
    }
    ax.set_xticklabels(
        [contrast_labels.get(c, c) for c in classes],
        fontweight="bold",
        fontsize=9,
    )

    ax.set_yticks(range(len(pathway_order)))
    ax.set_yticklabels(pathway_order, fontsize=8)

    # Balance the horizontal panel margins.
    # The statistical values and the Active/Stable minus Control contrasts are
    # unchanged; this affects presentation only.
    ax.set_xlim(min(xticks) - x_gap * 0.60, max(xticks) + x_gap * 0.60)
    ax.set_ylim(-0.5, len(pathway_order) - 0.5)
    ax.invert_yaxis()

    ax.grid(True, axis="both", linestyle=":", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    # Match the frame used in the other biological-interpretation plots.
    _apply_rcparams_frame(ax)

    ax.set_title("KEGG pathway over-representation", fontweight="bold", fontsize=10, pad=6)

    # -------- right area: legend + colorbar --------
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.axis("off")

    # -------- FDR size legend --------
    legend_kw = dict(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.55,
        handletextpad=0.5,
    )

    legend_fdrs = [0.01, 0.05, 0.10]
    legend_sizes_raw = [70 + 250 * (-np.log10(f) / max_nlog) for f in legend_fdrs]
    # Scatter uses area in pt^2; Line2D uses marker diameter in pt.
    # sqrt(area) therefore reproduces the exact point-size mapping.
    marker_sizes = [np.sqrt(s) for s in legend_sizes_raw]

    size_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=ms,
            markerfacecolor="#a0a0a0",
            markeredgecolor="#444444",
            markeredgewidth=0.5,
        )
        for ms in marker_sizes
    ]

    leg_fdr = fig.legend(
        size_handles,
        [f"q = {f:.2f}" for f in legend_fdrs],
        title="FDR q value",
        loc="upper left",
        bbox_to_anchor=(0.64, 0.85),
        bbox_transform=fig.transFigure,
        fontsize=7,
        title_fontsize=8,
        handlelength=1.2,
        handleheight=2.2,
        **legend_kw
    )

    # Use the same border style as the other biological-interpretation plots.
    frame = leg_fdr.get_frame()
    frame.set_edgecolor(edge_color)
    frame.set_linewidth(edge_lw)

    leg_fdr.get_title().set_color(text_color)
    for t in leg_fdr.get_texts():
        t.set_color(text_color)

    # colorbar (keep ticks readable; do not tie to edge_color)
    cax = fig.add_axes([0.68, 0.15, 0.025, 0.45])
    sm = ScalarMappable(norm=norm, cmap=MORANDI_CMAP)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")

    cbar.ax.tick_params(labelsize=7, length=2, width=edge_lw, colors=text_color, labelcolor=text_color)
    cbar.outline.set_linewidth(edge_lw)
    cbar.outline.set_edgecolor(edge_color)

    cbar.set_label("log₂FC", fontsize=9, labelpad=8, color=text_color)

    cbar.ax.text(
        2.8, 1.02, "↑Up",
        transform=cbar.ax.transAxes,
        fontsize=7, ha="left", va="bottom",
        color="#B36A6F", fontweight="bold"
    )
    cbar.ax.text(
        2.8, -0.02, "↓Down",
        transform=cbar.ax.transAxes,
        fontsize=7, ha="left", va="top",
        color="#4A6A91", fontweight="bold"
    )

    # -------- save --------
    save_path = Path(save_path)
    pdf_path, png_path = save_publication_figure(
        fig,
        save_path,
        preview_png=True,
        png_dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
        bbox_extra_artists=(leg_fdr,),
    )
    plt.close(fig)

    print(f"[Plot] Saved PDF: {pdf_path}")
    if png_path is not None:
        print(f"[Plot] Saved PNG: {png_path}")
    print(f"[Plot] {len(pathway_order)} pathways × {len(classes)} classes")
    return True

def plot_ora_lollipop_biological(
    df: pd.DataFrame,
    save_path: Path,
    top_n: int = 10,
    fdr_threshold: float = 0.10,
    p_threshold: float = 0.05,
    min_overlap: int = 2,
    reference_class: str = "Control"
):
    """
    Plot abundance-based pathway log2 fold changes by clinical class.
    """
    plt.rcParams.update({
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
    })

    if df.empty:
        return

    df_plot = df[
        (df['fdr'] < fdr_threshold)
        & (df['p_fisher'] < p_threshold)
        & (df['k_overlap'] >= min_overlap)
    ].copy()

    df_plot = df_plot[df_plot['class'] != reference_class]

    classes = ["Active", "Stable"]
    classes = [c for c in classes if c in df_plot['class'].unique()]

    if not classes:
        print("[Warning] No disease classes to plot")
        return

    def clean_name(s):
        s = str(s).split(" - ")[0]
        replacements = {
            "Valine, leucine and isoleucine": "BCAA",
            "biosynthesis": "biosynth.",
            "degradation": "degrad.",
            "metabolism": "metab.",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s[:32] + "..." if len(s) > 32 else s

    df_plot['pathway_clean'] = df_plot['pathway'].apply(clean_name)

    fig, axes = plt.subplots(1, len(classes), figsize=(5*len(classes), max(5, top_n*0.45)))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        cls_df = df_plot[df_plot['class'] == cls].nsmallest(top_n, 'p_fisher').copy()

        if cls_df.empty:
            ax.set_title(f'{cls}\n(No significant pathways)')
            ax.set_xlim(-1, 1)
            continue

        cls_df = cls_df.sort_values('log2fc', ascending=True)

        pathways = cls_df['pathway_clean'].tolist()
        y_positions = range(len(pathways))

        colors = ['#d73027' if x > 0 else '#4575b4' for x in cls_df['log2fc']]

        ax.hlines(y=y_positions, xmin=0, xmax=cls_df['log2fc'],
                 colors=colors, linewidth=2.5, alpha=0.8)

        sizes = 50 + 150 * (-np.log10(cls_df['fdr'].clip(1e-10)) / 3)
        ax.scatter(cls_df['log2fc'], y_positions, c=colors, s=sizes,
                  edgecolors='black', linewidths=0.5, zorder=3)

        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(pathways, fontsize=8)
        ax.set_xlabel(f'log₂FC (vs {reference_class})', fontsize=10)
        ax.set_title(f'{cls}', fontweight='bold', fontsize=12)

        max_abs = max(abs(cls_df['log2fc'].min()), abs(cls_df['log2fc'].max()), 0.5)
        ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)

        ax.grid(True, axis='x', linestyle=':', alpha=0.3)

    fig.text(0.5, 0.01, f'← Down-regulated in patients     Up-regulated in patients →',
            ha='center', fontsize=9, style='italic')

    plt.suptitle(f'Metabolic Pathway Changes in SLE\n(Compared to {reference_class})',
                fontweight='bold', fontsize=12, y=1.02)

    plt.tight_layout()

    pdf_path, png_path = save_publication_figure(
        fig,
        save_path,
        preview_png=True,
        png_dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)

    print(f"[Plot] Saved PDF: {pdf_path}")
    if png_path is not None:
        print(f"[Plot] Saved PNG: {png_path}")

def generate_results_table(
    df: pd.DataFrame,
    output_path: Path,
    reference_class: str = "Control",
    fdr_threshold: float = 0.10,
    p_threshold: float = 0.05,
    min_overlap: int = 2,
):
    """Generate the reported pathway-results table."""

    df = df[
        (df['fdr'] < fdr_threshold)
        & (df['p_fisher'] < p_threshold)
        & (df['k_overlap'] >= min_overlap)
    ].copy()

    cols = {
        'class': 'Class',
        'pathway': 'Pathway',
        'k_overlap': 'Overlap',
        'pathway_size': 'Pathway Size',
        'fold_enrichment': 'Fold Enrichment',
        'log2fc': f'log2FC (vs {reference_class})',
        'direction': 'Direction',
        'n_up_metabolites': 'N Up',
        'n_down_metabolites': 'N Down',
        'p_fisher': 'P-value',
        'fdr': 'FDR',
        'overlap_cids': 'Overlap CIDs'
    }

    table_df = df[[c for c in cols.keys() if c in df.columns]].copy()
    table_df = table_df.rename(columns=cols)

    table_df['Pathway'] = table_df['Pathway'].apply(lambda s: s.split(" - ")[0])

    table_df['P-value'] = table_df['P-value'].apply(lambda x: f'{x:.2e}')
    # Four decimals avoid displaying values just below 0.10 as 0.100.
    table_df['FDR'] = table_df['FDR'].apply(lambda x: f'{x:.4f}')
    table_df['Fold Enrichment'] = table_df['Fold Enrichment'].apply(lambda x: f'{x:.1f}')
    fc_col = f'log2FC (vs {reference_class})'
    table_df[fc_col] = table_df[fc_col].apply(lambda x: f'{x:.3f}')

    table_df['Direction'] = table_df['Direction'].map({
        'up': '↑ Up', 'down': '↓ Down', 'neutral': '- Neutral'
    })

    table_df.to_csv(output_path, index=False)
    print(f"[Table] Saved: {output_path}")


# ============================================================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Mass ORA with Biological Direction (Dual Threshold)")
    parser.add_argument("--shap_npz",
                       default="clinical_explain_results/shap_outputs/mor_shap_outputs.npz")
    parser.add_argument("--annotation",
                       default="clinical_explain_results/enrichment_results/mass_kegg/enhanced_annotation/enhanced_annotation_results.csv")
    parser.add_argument("--feature_to_cpd",
                       default="clinical_explain_results/enrichment_results/mass_kegg/audit/mapping/feature_to_cpd.json")
    parser.add_argument("--kegg_pathway_to_cids",
                       default="kegg_cache/kegg_hsa_pathway_to_cids.json")
    parser.add_argument("--mass_abundance",
                       default="data/mass/mass_features_fingerprint_tic_log.csv",
                       help="CSV containing the mass-spectrometry abundance matrix")
    parser.add_argument("--outdir",
                       default="clinical_explain_results/enrichment_results/mass_kegg_ora_biological")

    parser.add_argument("--fdr", type=float, default=0.10,
                       help="FDR threshold (default: 0.10)")
    parser.add_argument("--p_threshold", type=float, default=0.05,
                       help="Nominal P-value threshold (default: 0.05)")
    parser.add_argument("--require_both", action="store_true", default=True,
                       help="Require both nominal P and FDR thresholds")
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--reference", type=str, default="Control",
                       help="Reference group used for fold-change calculation")
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ORAConfig(
        fdr_threshold=args.fdr,
        p_threshold=args.p_threshold,
        require_both=args.require_both,
        reference_class=args.reference
    )

    print("=" * 70)
    print("MASS ORA WITH BIOLOGICAL DIRECTION (DUAL THRESHOLD)")
    print("=" * 70)
    print(f"  FDR threshold: {args.fdr}")
    print(f"  p-value threshold: {args.p_threshold}")
    print(f"  Require both: {args.require_both}")

    print("\n[1/6] Loading data...")
    phi_stack, y_true, feature_names = load_shap_results(Path(args.shap_npz))
    annotated_df = load_enhanced_annotation(Path(args.annotation))
    feature_to_cids = load_feature_to_cids(Path(args.feature_to_cpd))
    pathway_to_cids = load_pathway_to_cids(Path(args.kegg_pathway_to_cids))

    print(f"  Annotated features: {len(annotated_df)}")
    print(f"  KEGG pathways: {len(pathway_to_cids)}")

    print("\n[2/6] Loading abundance data...")
    X_abundance, y_abundance = load_mass_abundance_data(Path(args.mass_abundance))
    print(f"  Samples: {len(X_abundance)}")
    print(f"  Features: {X_abundance.shape[1]}")
    print(f"  Groups: {y_abundance.value_counts().to_dict()}")

    print("\n[3/6] Running ORA with biological direction...")
    results_df, debug_info = run_classwise_ora_with_biological_direction(
        phi_stack=phi_stack,
        y_true=y_true,
        feature_names=feature_names,
        annotated_features=annotated_df,
        feature_to_cids=feature_to_cids,
        pathway_to_cids=pathway_to_cids,
        X_abundance=X_abundance,
        y_abundance_labels=y_abundance,
        config=config
    )

    if results_df.empty:
        print("[Warning] No results!")
        return

    results_df.to_csv(output_dir / "mass_ora_biological_direction.csv", index=False)
    print(f"\n[4/6] Results saved: mass_ora_biological_direction.csv")

    print("\n  Direction summary (vs Control):")
    for cls in ['Active', 'Stable']:
        cls_df = results_df[(results_df['class'] == cls) & (results_df['fdr'] < args.fdr)]
        if cls_df.empty:
            continue
        up = (cls_df['direction'] == 'up').sum()
        down = (cls_df['direction'] == 'down').sum()
        neutral = (cls_df['direction'] == 'neutral').sum()
        print(f"    {cls}: ↑{up} pathways up, ↓{down} pathways down, {neutral} neutral")

    print("\n[5/6] Generating visualizations...")

    plot_ora_dotplot_biological(
        results_df,
        output_dir / "mass_ora_dotplot_biological.png",
        top_n=args.top_n,
        fdr_threshold=args.fdr,
        reference_class=args.reference
    )

    plot_ora_lollipop_biological(
        results_df,
        output_dir / "mass_ora_lollipop_biological.png",
        top_n=10,
        fdr_threshold=args.fdr,
        reference_class=args.reference
    )

    print("\n[6/6] Generating pathway-results table...")
    generate_results_table(
        results_df,
        output_dir / "mass_ora_reported_pathways.csv",
        reference_class=args.reference,
        fdr_threshold=args.fdr,
        p_threshold=args.p_threshold,
        min_overlap=2,
    )

    report = f"""
================================================================================
MASS ORA ANALYSIS REPORT (Biological Direction)
================================================================================

Reference Class: {args.reference}
FDR Threshold: {args.fdr}

DIRECTION INTERPRETATION:
- Up (Red): Metabolites in this pathway are HIGHER in patients vs {args.reference}
- Down (Blue): Metabolites in this pathway are LOWER in patients vs {args.reference}

RESULTS SUMMARY:
"""
    for cls in ['Active', 'Stable', 'Control']:
        cls_df = results_df[(results_df['class'] == cls) & (results_df['fdr'] < args.fdr)]
        if cls_df.empty:
            continue
        report += f"\n{cls}:\n"
        report += f"  Significant pathways: {len(cls_df)}\n"
        if cls != args.reference:
            up = (cls_df['direction'] == 'up').sum()
            down = (cls_df['direction'] == 'down').sum()
            report += f"  Up-regulated pathways: {up}\n"
            report += f"  Down-regulated pathways: {down}\n"

    report += f"""
OUTPUT FILES:
- mass_ora_biological_direction.csv: Complete results
- mass_ora_dotplot_biological.png/pdf: Dotplot visualization
- mass_ora_lollipop_biological.png/pdf: Lollipop plot
- mass_ora_reported_pathways.csv: Pathways meeting the reporting thresholds

================================================================================
"""

    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report)

    print(report)
    print(f"\n{'='*70}")
    print(f"[Done] Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
