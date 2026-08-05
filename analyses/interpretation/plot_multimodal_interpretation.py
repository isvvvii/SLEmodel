# plot_multimodal_interpretation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

from slemodel import config as cfg
from .gly_traits_analysis import compute_gly_traits
from .feature_annotations import build_feature_label_maps

from . import enrichment_plot_utils as epu

try:
    import gseapy as gp
    HAVE_GSEAPY = True
except Exception:
    HAVE_GSEAPY = False

try:
    from . import mass_kegg_ora_biological_direction as mass_ora_mod
    HAVE_MASS_ORA_MODULE = True
except Exception:
    mass_ora_mod = None
    HAVE_MASS_ORA_MODULE = False


# ==============================================================================
# Global PDF / font configuration for Illustrator-editable output
# ==============================================================================

def configure_matplotlib_for_editable_vector() -> None:
    """
    Force vector-friendly / Illustrator-friendly output.
    """
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,              # Type 42 / TrueType
        "ps.fonttype": 42,               # Type 42 / TrueType
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",          # keep text as text in SVG
        "text.usetex": False,            # avoid TeX text path conversion
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


configure_matplotlib_for_editable_vector()

# Patch enrichment_plot_utils.set_publication_style so that any internal calls
# also restore the vector-friendly rcParams afterwards.
_ORIGINAL_EPU_SET_PUBLICATION_STYLE = epu.set_publication_style


def set_publication_style(*args, **kwargs):
    _ORIGINAL_EPU_SET_PUBLICATION_STYLE(*args, **kwargs)
    configure_matplotlib_for_editable_vector()


epu.set_publication_style = set_publication_style
plot_rna_stacked_dotplot = epu.plot_rna_stacked_dotplot
plot_mass_dotplot_contrast_vs_ref = epu.plot_mass_dotplot_contrast_vs_ref

# If the mass ORA helper module defines its own style function, patch it too.
if HAVE_MASS_ORA_MODULE and hasattr(mass_ora_mod, "set_publication_style"):
    _ORIGINAL_MASS_ORA_SET_PUBLICATION_STYLE = mass_ora_mod.set_publication_style

    def _patched_mass_ora_style(*args, **kwargs):
        _ORIGINAL_MASS_ORA_SET_PUBLICATION_STYLE(*args, **kwargs)
        configure_matplotlib_for_editable_vector()

    mass_ora_mod.set_publication_style = _patched_mass_ora_style


GLOBAL_SEED = 202510
np.random.seed(GLOBAL_SEED)

CLASS_COLORS = {"Active": "#E68B81", "Stable": "#8aab82", "Control": "#7DA6C6"}
MODALITY_COLORS = {"rna": "#52854C", "mass": "#4E84C4", "gly": "#D16103"}


# ==============================================================================
# Save helpers
# ==============================================================================

def normalize_pdf_path(path: Path | str) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    return path


def save_figure_pdf_and_preview(
    fig: plt.Figure,
    save_path: Path | str,
    *,
    preview_png: bool = True,
    preview_dpi: int = 600,
    bbox_inches="tight",
    pad_inches: float = 0.02,
    facecolor: str = "white",
    transparent: bool = False,
    extra_artists=None,
) -> Path:
    """
    Save the figure as editable PDF and optionally as PNG preview.
    """
    configure_matplotlib_for_editable_vector()

    pdf_path = normalize_pdf_path(save_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        facecolor=facecolor,
        transparent=transparent,
    )
    if extra_artists is not None:
        save_kwargs["bbox_extra_artists"] = tuple(extra_artists)

    fig.savefig(pdf_path, format="pdf", **save_kwargs)

    if preview_png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=preview_dpi, **save_kwargs)

    return pdf_path


def save_placeholder_panel(
    save_path: Path | str,
    message: str,
    figsize: Tuple[float, float] = (5.2, 5.6),
    preview_png: bool = True,
) -> Path:
    configure_matplotlib_for_editable_vector()
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color="#666")
    ax.axis("off")
    out_pdf = save_figure_pdf_and_preview(
        fig,
        save_path,
        preview_png=preview_png,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return out_pdf


def ensure_panel_pdf_exists(panel_pdf: Path | str, panel_name: str) -> Path:
    panel_pdf = normalize_pdf_path(panel_pdf)
    if not panel_pdf.exists():
        raise FileNotFoundError(
            f"{panel_name} did not produce {panel_pdf}. "
            f"This usually means the corresponding helper still saved only PNG "
            f"or used a raster-only export path."
        )
    return panel_pdf


# ==============================================================================
# Basic utilities
# ==============================================================================

def load_modality_csv(path: Path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].values
    labels = df.iloc[:, 1].astype(str).values
    X = df.iloc[:, 2:].values
    featnames = df.columns[2:].tolist()
    return ids, labels, X, featnames


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = a.size
    n_b = b.size
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    if pooled <= 1e-12:
        return 0.0
    return (mean_a - mean_b) / math.sqrt(pooled)


def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    mean = values.mean()
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(mean), float(lo), float(hi)


def compute_diff_stats(X, labels, group_a, group_b, feature_names):
    rows = []
    for idx, feat in enumerate(feature_names):
        a = X[labels == group_a, idx]
        b = X[labels == group_b, idx]
        if a.size == 0 or b.size == 0:
            continue
        try:
            pval = mannwhitneyu(a, b, alternative="two-sided").pvalue
        except Exception:
            pval = 1.0
        d = cohen_d(a, b)
        rows.append({"feature": feat, "cohen_d": d, "p_value": pval})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = bh_adjust(df["p_value"].values)
    df["rank_score"] = np.sign(df["cohen_d"]) * -np.log10(np.clip(df["p_value"], 1e-300, 1.0))
    return df


def aggregate_by_gene(gene_names, scores):
    df = pd.DataFrame({"gene": gene_names, "score": scores})
    df["abs_score"] = np.abs(df["score"])
    df = df.sort_values("abs_score", ascending=False).drop_duplicates("gene")
    return df


def build_rna_gene_names(rna_featnames):
    full = [f"rna::{n}" for n in rna_featnames]
    display_map, _, _ = build_feature_label_maps(full)
    names = [display_map.get(f"rna::{n}", n) for n in rna_featnames]
    return names

def resolve_gly_trait_name(traits_df: pd.DataFrame, trait_name: str) -> Optional[str]:
    """
    Resolve displayed glycan-trait names to data-frame column names.
    Supports both short names and internal calculated names.
    """
    alias_map = {
        "STotal": ["STotal", "S_total_calc"],
        "S1Total": ["S1Total", "S1_total_calc"],
        "FG1S1": ["FG1S1", "FG1S1_ratio_pct_calc"],
        "FG2S1": ["FG2S1", "FG2S1_ratio_pct_calc"],
        "GI": ["GI"],
        "bisecting": ["bisecting"],
        "G0n": ["G0n"],
        "G1n": ["G1n"],
        "G2n": ["G2n"],
    }

    col_map = {str(c).lower(): str(c) for c in traits_df.columns}

    for cand in alias_map.get(trait_name, [trait_name]):
        if cand in traits_df.columns:
            return cand
        if str(cand).lower() in col_map:
            return col_map[str(cand).lower()]

    return None


def gly_trait_display_name(trait_name: str) -> str:
    """
    Convert internal column names to cleaner display names.
    """
    display_map = {
        "S_total_calc": "STotal",
        "S1_total_calc": "S1Total",
        "FG1S1_ratio_pct_calc": "FG1S1",
        "FG2S1_ratio_pct_calc": "FG2S1",
    }
    return display_map.get(trait_name, trait_name)

# ==============================================================================
# Panel A: RNA GSEA
# ==============================================================================

def run_rna_gsea(rna_csv: Path, gmt_dict: Dict[str, Path], outdir: Path, permutations: int = 500):
    if not HAVE_GSEAPY:
        warnings.warn("gseapy is not installed. Skipping RNA GSEA.")
        return None

    _, labels, X, rna_featnames = load_modality_csv(rna_csv)
    X = np.log1p(X)
    gene_names = build_rna_gene_names(rna_featnames)

    contrasts = [("Active", "Control"), ("Stable", "Control")]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_dict = {k: [] for k in gmt_dict.keys()}

    for grp_a, grp_b in contrasts:
        df = compute_diff_stats(X, labels, grp_a, grp_b, gene_names)
        if df.empty:
            continue
        df = aggregate_by_gene(df["feature"].values, df["rank_score"].values)
        rank_series = pd.Series(df["score"].values, index=df["gene"]).dropna()
        if len(rank_series) < 50:
            continue

        for tag, gmt in gmt_dict.items():
            gmt = Path(gmt)
            if not gmt.exists():
                continue

            out_sub = outdir / f"gsea_{tag}_{grp_a}_vs_{grp_b}"
            try:
                enr = gp.prerank(
                    rnk=rank_series,
                    gene_sets=str(gmt),
                    threads=4,
                    permutation_num=permutations,
                    outdir=str(out_sub),
                    seed=7,
                    format="png",
                    no_plot=True,
                    verbose=False,
                )
                res = enr.res2d.reset_index().rename(columns={"Term": "pathway"})
                if "NES" not in res.columns:
                    continue
                res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
                res["FDR"] = pd.to_numeric(res.get("FDR q-val", np.nan), errors="coerce")
                res["p-val"] = pd.to_numeric(res.get("NOM p-val", np.nan), errors="coerce")
                res = res.dropna(subset=["NES", "FDR", "p-val"])
                if res.empty:
                    continue
                res["class"] = grp_a
                data_dict[tag].append(res[["class", "pathway", "NES", "p-val", "FDR"]])
            except Exception as e:
                warnings.warn(f"GSEA failed for {tag} {grp_a} vs {grp_b}: {e}")

    merged = {}
    for tag, lst in data_dict.items():
        merged[tag] = pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(
            columns=["class", "pathway", "NES", "p-val", "FDR"]
        )
    return merged


# ==============================================================================
# Panel B: Gly forest + distributions
# ==============================================================================

def plot_gly_trait_forest_with_distribution(
    traits_df: pd.DataFrame,
    trait_cols: List[str],
    save_path: Path,
    ref_group: str = "Control",
    comp_group: str = "Active",
    n_traits_for_dist: int = 4,
    dist_traits: Optional[List[str]] = None,
):
    """
    Gly forest plot + distributions
    Retain STotal in the distribution panel and replace S1Total with GI when available.
    """
    set_publication_style()
    configure_matplotlib_for_editable_vector()

    group_col = traits_df.columns[1]
    groups_all = [g for g in ["Control", "Stable", "Active"] if g in traits_df[group_col].unique()]

    sub = traits_df[traits_df[group_col].isin([ref_group, comp_group])].copy()
    if sub.empty:
        warnings.warn("No data for requested groups in gly forest plot.")
        return

    rows = []
    rng = np.random.default_rng(GLOBAL_SEED)

    for t in trait_cols:
        if t not in sub.columns:
            continue
        a = pd.to_numeric(sub.loc[sub[group_col] == comp_group, t], errors="coerce").dropna().values
        c = pd.to_numeric(sub.loc[sub[group_col] == ref_group, t], errors="coerce").dropna().values
        if a.size < 3 or c.size < 3:
            continue

        diff = a.mean() - c.mean()

        boots = []
        for _ in range(2500):
            aa = rng.choice(a, size=a.size, replace=True)
            cc = rng.choice(c, size=c.size, replace=True)
            boots.append(aa.mean() - cc.mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])

        try:
            # The plotted effect is a difference in means, so use the
            # corresponding unequal-variance two-sample test.
            p = ttest_ind(a, c, equal_var=False, alternative="two-sided").pvalue
        except Exception:
            p = np.nan

        rows.append({
            "trait": t,
            "n_comp": int(a.size),
            "n_ref": int(c.size),
            "diff": float(diff),
            "lo": float(lo),
            "hi": float(hi),
            "p": float(p) if np.isfinite(p) else np.nan,
        })

    if not rows:
        warnings.warn("No valid traits for gly forest plot.")
        return

    P = np.array([r["p"] for r in rows], dtype=float)
    q = np.full_like(P, np.nan)
    mask = np.isfinite(P)
    if np.any(mask):
        q[mask] = bh_adjust(P[mask])
    for i, qi in enumerate(q):
        rows[i]["q"] = float(qi) if np.isfinite(qi) else np.nan

    D = pd.DataFrame(rows).sort_values("diff", ascending=False).reset_index(drop=True)

    # Export the statistics used in the glycan-trait plot.
    stats_table = D.copy()
    stats_table.insert(
        1,
        "trait_display",
        [gly_trait_display_name(t) for t in stats_table["trait"]],
    )
    stats_table["comparison"] = f"{comp_group} - {ref_group}"
    stats_table["effect"] = "difference in means"
    stats_table["confidence_interval"] = "percentile bootstrap 95% CI"
    stats_table["test"] = "two-sided Welch's t-test"
    stats_table["multiplicity_adjustment"] = "Benjamini-Hochberg"
    stats_path = Path(save_path).with_name(
        f"{Path(save_path).stem}_statistics.csv"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_table.to_csv(stats_path, index=False, encoding="utf-8")

    D_sorted_by_diff = D.copy()
    D_sorted_by_diff["abs_diff"] = D_sorted_by_diff["diff"].abs()

    # If distribution-panel traits are specified,
    # use them directly; otherwise keep the original auto-selection logic.
    if dist_traits is not None and len(dist_traits) > 0:
        selected = []
        for t in dist_traits:
            resolved = resolve_gly_trait_name(traits_df, t)
            if resolved is None:
                warnings.warn(f"[Gly] Requested distribution trait '{t}' not found in traits_df.columns")
                continue
            selected.append(resolved)
        selected = list(dict.fromkeys(selected))[:n_traits_for_dist]
    else:
        selected = D_sorted_by_diff.sort_values("abs_diff", ascending=False)["trait"].head(n_traits_for_dist).tolist()

        gi_col = resolve_gly_trait_name(traits_df, "GI")
        s1_col = resolve_gly_trait_name(traits_df, "S1Total")
        stotal_col = resolve_gly_trait_name(traits_df, "STotal")

        if s1_col in selected and gi_col is not None:
            selected = [(gi_col if t == s1_col else t) for t in selected]
            selected = list(dict.fromkeys(selected))

        if stotal_col is not None and stotal_col not in selected:
            if len(selected) >= n_traits_for_dist:
                diffs = {row.trait: abs(row.diff) for row in D_sorted_by_diff.itertuples()}
                protected = {stotal_col}
                if gi_col is not None:
                    protected.add(gi_col)
                drop_candidates = [t for t in selected if t not in protected]
                if drop_candidates:
                    drop = min(drop_candidates, key=lambda t: diffs.get(t, 0.0))
                    selected.remove(drop)
            selected.append(stotal_col)

        selected = selected[:n_traits_for_dist]

    print(f"[Gly] Selected traits for distribution: {[gly_trait_display_name(t) for t in selected]}")

    fig = plt.figure(figsize=(13.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.15)

    ax_f = fig.add_subplot(gs[0, 0])

    y = np.arange(len(D))
    ax_f.hlines(y, D["lo"], D["hi"], color="#444", lw=2.0, zorder=1)

    colors = np.where(D["diff"].values >= 0, "#C96963", "#4C8C8A")
    ax_f.scatter(
        D["diff"], y,
        s=110,
        c=colors,
        edgecolors="black",
        linewidths=1.1,
        zorder=3
    )
    xmin = float(np.nanmin(D["lo"].values))
    xmax = float(np.nanmax(D["hi"].values))
    span = max(1e-6, xmax - xmin)
    ax_f.set_xlim(xmin - 0.30 * span, xmax + 0.30 * span)

    ax_f.axvline(0, color="grey", ls="--", lw=1.2)

    def _significance_prefix(q_value: float) -> str:
        if not np.isfinite(q_value):
            return ""
        if q_value < 0.001:
            return "*** "
        if q_value < 0.01:
            return "** "
        if q_value < 0.05:
            return "* "
        return ""

    ylabels = [
        f"{_significance_prefix(row.q)}{gly_trait_display_name(row.trait)}"
        for row in D.itertuples()
    ]

    ax_f.set_yticks(y)
    ax_f.set_yticklabels(ylabels, fontsize=10)
    ax_f.invert_yaxis()
    ax_f.set_xlabel(
        f"Δ glycan trait value ({comp_group} − {ref_group})",
        fontsize=11
    )
    ax_f.set_title(
        "Glycan trait differences",
        fontsize=12,
        fontweight="bold",
        pad=8
    )

    xpad = 0.03 * span

    def _fmt2(value: float) -> str:
        # Avoid displaying the visually confusing value "-0.00".
        value = 0.0 if abs(float(value)) < 0.005 else float(value)
        return f"{value:.2f}"

    for yi, row in enumerate(D.itertuples()):
        txt = f"{_fmt2(row.diff)} [{_fmt2(row.lo)}, {_fmt2(row.hi)}]"
        if row.diff >= 0:
            text_x, text_ha = row.lo - xpad, "right"
        else:
            text_x, text_ha = row.hi + xpad, "left"
        ax_f.text(
            text_x,
            yi,
            txt,
            ha=text_ha,
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.15),
            clip_on=False,
        )

    ax_placeholder = fig.add_subplot(gs[0, 1])
    ax_placeholder.remove()

    if selected:
        n = min(4, len(selected))
        gs_inner = gs[0, 1].subgridspec(2, 2, hspace=0.32, wspace=0.22)

        for k in range(n):
            ax = fig.add_subplot(gs_inner[k // 2, k % 2])
            t = selected[k]
            df_plot = traits_df[[group_col, t]].copy()
            df_plot = df_plot[df_plot[group_col].isin(groups_all)]
            df_plot[t] = pd.to_numeric(df_plot[t], errors="coerce")
            df_plot = df_plot.dropna(subset=[t])

            if df_plot.empty:
                ax.text(0.5, 0.5, f"No data for {gly_trait_display_name(t)}", ha="center", va="center")
                continue

            sns.violinplot(
                data=df_plot, x=group_col, y=t, order=groups_all,
                hue=group_col, hue_order=groups_all,
                palette=CLASS_COLORS, inner=None, cut=0, linewidth=0.9, ax=ax, alpha=0.75,
                legend=False
            )
            sns.boxplot(
                data=df_plot, x=group_col, y=t, order=groups_all,
                hue=group_col, hue_order=groups_all,
                palette=CLASS_COLORS, width=0.13, showfliers=False, ax=ax,
                boxprops=dict(alpha=0.85), whiskerprops=dict(linewidth=1.0),
                legend=False
            )
            sns.stripplot(
                data=df_plot, x=group_col, y=t, order=groups_all,
                color="black", size=2.3, alpha=0.35, jitter=0.12, ax=ax
            )

            for i, grp in enumerate(groups_all):
                vals = df_plot.loc[df_plot[group_col] == grp, t].values
                if vals.size:
                    mean_val, lo_val, hi_val = bootstrap_mean_ci(vals)
                    ax.errorbar(
                        i, mean_val,
                        yerr=[[mean_val - lo_val], [hi_val - mean_val]],
                        fmt="D", color="black", markersize=4.8, capsize=3, lw=1.4, zorder=5
                    )

            ax.set_title(gly_trait_display_name(t), fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")

            for side in ["left", "right", "top", "bottom"]:
                ax.spines[side].set_visible(True)

            if k % 2 == 0:
                ax.yaxis.set_ticks_position("left")
                ax.yaxis.set_label_position("left")
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=True,
                    right=False,
                    labelleft=True,
                    labelright=False
                )

            else:
                ax.yaxis.set_ticks_position("right")
                ax.yaxis.set_label_position("right")
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=True,
                    labelleft=False,
                    labelright=True,
                    pad=3
                )

            if k < 2:
                ax.set_xticklabels([])

        fig.text(0.79, 0.98, "Trait distributions", ha="center", va="top", fontsize=12, fontweight="bold")

    fig.text(
        0.30,
        0.018,
        "Two-sided Welch's t-test; BH-adjusted q: "
        "*** <0.001, ** <0.01, * <0.05",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.055, 1, 1])
    save_figure_pdf_and_preview(
        fig,
        save_path,
        preview_png=True,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


# ==============================================================================
# Panel D: circular network
# ==============================================================================

def get_default_module_definitions():
    return {
        "IFN-I": {"modality": "rna", "keywords": ["IFIT", "IFI", "ISG", "MX1", "MX2", "OAS", "IRF7", "STAT1", "RSAD2"]},
        "Inflammation": {"modality": "rna", "keywords": ["TNF", "IL1", "IL6", "CXCL", "CCL", "NFKB", "NLRP"]},
        "B Cell": {"modality": "rna", "keywords": ["MS4A1", "CD79", "CD19", "BANK1", "BLK", "PAX5"]},
        "T Cell": {"modality": "rna", "keywords": ["CD3", "CD4", "CD8", "TRAC", "LCK", "ZAP70"]},
        "Sialylation": {"modality": "gly", "keywords": ["STOTAL", "S1TOTAL", "S2TOTAL", "SIA", "S1", "S2", "GP16", "GP17", "GP18", "GP19", "GP21", "GP22", "GP23", "GP24"]},
        "Galactosylation": {"modality": "gly", "keywords": ["G0", "G1", "G2", "GAL", "GI"]},
        "Core Fucose": {"modality": "gly", "keywords": ["FUC", "FG", "FB", "FNTOTAL"]},
        "Amino Acid": {"modality": "mass", "keywords": ["VALINE", "LEUCINE", "ISOLEUCINE", "GLYCINE", "ALANINE", "SERINE", "PROLINE", "GLUTAM", "ARGININE", "LYSINE", "TYROSINE", "TRYPTOPHAN"]},
        "Energy/TCA": {"modality": "mass", "keywords": ["CITRATE", "SUCCIN", "MALATE", "FUMARATE", "PYRUVATE", "LACTATE", "GLUCOSE"]},
    }


def plot_module_interaction_network_circular_style(
    shap_npz_path: Path,
    save_path: Path,
    module_definitions: Optional[dict] = None,
    class_index: int = None,
    bootstrap_n: int = 500,
    edge_fdr: float = 0.05,
    dashed_edge_fdr: float = 0.10,
    min_features_per_module: int = 2,
    rna_gmt_path: Optional[Path] = None,
    rna_pathway_csv: Optional[Path] = None,
    rna_top_k: int = 4,
    rna_min_genes: int = 5,
    mass_kegg_csv: Optional[Path] = None,
    feature_to_cpd_json: Optional[Path] = None,
    pathway_to_cids_json: Optional[Path] = None,
    mass_fdr_threshold: float = 0.05,
    mass_top_k: int = 3,
    node_color_metric: str = "standardized_active_minus_control",
    edge_scope: str = "cross_modal",
):
    """
    Plot pooled sample-wise SHAP associations between pre-defined modules.

    All candidate modules are retained in the audit tables. The main network
    shows solid edges for BH-adjusted q < ``edge_fdr`` and dashed exploratory
    edges for ``edge_fdr`` <= q < ``dashed_edge_fdr``. Bootstrap sign stability
    is exported as a sensitivity diagnostic but is not an additional display
    threshold.
    """
    set_publication_style()
    configure_matplotlib_for_editable_vector()

    shap_npz_path = Path(shap_npz_path)
    if not shap_npz_path.exists():
        warnings.warn(f"SHAP file not found: {shap_npz_path}")
        save_placeholder_panel(save_path, "Network not available", figsize=(6, 5), preview_png=True)
        return

    dat = np.load(shap_npz_path, allow_pickle=True)
    phi_stack = dat["phi_stack"]
    feature_names = dat["feature_names"].tolist()
    y_true = np.asarray(dat["y_true"]) if "y_true" in dat.files else None

    valid_node_color_metrics = {
        "standardized_active_minus_control",
        "active_minus_control",
        "active_only",
        "mean_signed_all",
    }
    if node_color_metric not in valid_node_color_metrics:
        raise ValueError(
            f"node_color_metric must be one of {sorted(valid_node_color_metrics)}; "
            f"got {node_color_metric!r}"
        )

    valid_edge_scopes = {"cross_modal", "all"}
    if edge_scope not in valid_edge_scopes:
        raise ValueError(
            f"edge_scope must be one of {sorted(valid_edge_scopes)}; "
            f"got {edge_scope!r}"
        )
    if not (0 < float(edge_fdr) < float(dashed_edge_fdr) <= 1):
        raise ValueError(
            "Require 0 < edge_fdr < dashed_edge_fdr <= 1; got "
            f"edge_fdr={edge_fdr!r}, dashed_edge_fdr={dashed_edge_fdr!r}."
        )

    if class_index is None:
        active_class_idx = None
        for k, v in cfg.VisualizationConfig.LABEL_NAMES.items():
            if v == "Active":
                active_class_idx = k
                break
        class_index = 1 if active_class_idx is None else active_class_idx

    if not (0 <= class_index < phi_stack.shape[2]):
        raise ValueError(f"class_index={class_index} out of range")

    display_map, _, _ = build_feature_label_maps(feature_names)

    def modality(fn: str) -> str:
        return fn.split("::", 1)[0] if "::" in fn else "?"

    def _load_gmt(gmt_path: Path) -> Dict[str, List[str]]:
        mp = {}
        with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                term = parts[0].strip()
                genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                mp[term] = genes
        return mp

    def _short_rna_label(p: str) -> str:
        s = p.replace("HALLMARK_", "").replace("REACTOME_", "").replace("GOBP_", "")
        s = s.replace("_", " ").title()
        repl = {
            "Interferon Alpha Response": "IFN-α",
            "Interferon Gamma Response": "IFN-γ",
            "Inflammatory Response": "inflammation",
            "Complement": "Complement",
            "Tnfa Signaling Via Nfkb": "TNFα→NF-κB",
            "Interferon Signaling": "IFN signaling",
        }
        s = repl.get(s, s)
        return f"RNA: {s}"

    def _short_mass_label(p: str) -> Optional[str]:
        s = p.lower()
        if "metabolic pathways" in s:
            return None
        if "biosynthesis of cofactors" in s:
            return "Mass: cofactor biosynthesis"
        if "pantothenate" in s or "coa" in s:
            return "Mass: pantothenate/CoA biosynthesis"
        if "arginine and proline metabolism" in s:
            return "Mass: arginine/proline metabolism"
        if "pyrimidine metabolism" in s:
            return "Mass: pyrimidine metabolism"
        if "tyrosine metabolism" in s:
            return "Mass: tyrosine metabolism"
        if "biosynthesis of amino acids" in s:
            return "Mass: amino-acid biosynthesis"
        if "valine, leucine and isoleucine biosynthesis" in s:
            return "Mass: BCAA biosynthesis"
        if "valine, leucine and isoleucine degradation" in s:
            return "Mass: BCAA degradation"
        if "butanoate metabolism" in s:
            return "Mass: butanoate metabolism"
        words = re.sub(r"\s+homo sapiens.*$", "", p, flags=re.IGNORECASE).split()
        return "Mass: " + " ".join(words[:4])

    phi_class = phi_stack[:, :, class_index]

    module_vec: Dict[str, np.ndarray] = {}
    module_imp: Dict[str, float] = {}
    module_val: Dict[str, float] = {}
    module_mod: Dict[str, str] = {}
    module_n_features: Dict[str, int] = {}

    # =========================
    # RNA pathway modules (GMT)
    # =========================
    rna_idx = [i for i, fn in enumerate(feature_names) if modality(fn) == "rna"]
    gene_to_indices: Dict[str, List[int]] = {}
    for i in rna_idx:
        g = str(display_map.get(feature_names[i], feature_names[i])).strip().upper()
        if not g or g in {"NAN", "NONE", "NULL"}:
            continue
        gene_to_indices.setdefault(g, []).append(i)

    def _indices_for_genes(genes: List[str]) -> List[int]:
        idxs = []
        for g in genes:
            gg = str(g).strip().upper()
            if gg in gene_to_indices:
                idxs.extend(gene_to_indices[gg])
        return sorted(set(idxs))

    def _add_rna_module(term: str, genes: List[str]) -> bool:
        idxs = _indices_for_genes(genes)
        if len(idxs) < int(rna_min_genes):
            return False
        label = _short_rna_label(term)
        if label in module_vec:
            return False
        v = phi_class[:, idxs].mean(axis=1)
        module_vec[label] = v
        module_imp[label] = float(np.mean(np.abs(v)))
        module_val[label] = float(np.mean(v))
        module_mod[label] = "rna"
        module_n_features[label] = len(idxs)
        return True

    rna_modules_added = 0
    if rna_gmt_path and Path(rna_gmt_path).exists():
        gmt = _load_gmt(Path(rna_gmt_path))

        selected_paths: List[str] = []
        if rna_pathway_csv and Path(rna_pathway_csv).exists():
            try:
                dfp = pd.read_csv(rna_pathway_csv)
                if "pathway" in dfp.columns:
                    if "class" in dfp.columns:
                        dfa = dfp[dfp["class"].astype(str).str.lower() == "active"].copy()
                        if dfa.empty:
                            dfa = dfp.copy()
                    else:
                        dfa = dfp.copy()

                    if "importance" in dfa.columns:
                        dfa["importance"] = pd.to_numeric(dfa["importance"], errors="coerce")
                        dfa = dfa.dropna(subset=["importance"]).sort_values("importance", ascending=False)
                        selected_paths = dfa["pathway"].head(rna_top_k * 3).astype(str).tolist()
                    elif "NES" in dfa.columns:
                        dfa["NES"] = pd.to_numeric(dfa["NES"], errors="coerce")
                        dfa = dfa.dropna(subset=["NES"])
                        dfa["score"] = dfa["NES"].abs()
                        dfa = dfa.sort_values("score", ascending=False)
                        selected_paths = dfa["pathway"].head(rna_top_k * 3).astype(str).tolist()
            except Exception:
                selected_paths = []

        selected_paths = [p for p in selected_paths if p in gmt] if selected_paths else []
        if not selected_paths:
            selected_paths = [
                "HALLMARK_INTERFERON_ALPHA_RESPONSE",
                "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                "HALLMARK_COMPLEMENT",
                "HALLMARK_INFLAMMATORY_RESPONSE",
                "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
            ]
            selected_paths = [p for p in selected_paths if p in gmt]

        for p in selected_paths:
            if _add_rna_module(p, gmt.get(p, [])):
                rna_modules_added += 1

        if rna_modules_added < max(2, int(rna_top_k)):
            candidates = []
            for term, genes in gmt.items():
                idxs = _indices_for_genes(genes)
                if len(idxs) < int(rna_min_genes):
                    continue
                v = phi_class[:, idxs].mean(axis=1)
                imp = float(np.mean(np.abs(v)))
                candidates.append((imp, term, genes))
            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, term, genes in candidates:
                if rna_modules_added >= int(rna_top_k):
                    break
                if _add_rna_module(term, genes):
                    rna_modules_added += 1

    # =========================
    # Gly modules
    # =========================
    gly_features = [fn for fn in feature_names if modality(fn) == "gly"]
    gly_raw = {fn: fn.split("::", 1)[1].upper() for fn in gly_features}
    gly_modules = {
        "Gly: sialylation": [
            "STOTAL", "S1TOTAL", "S2TOTAL", "SIA", "SA", "SIA/GAL",
            "FG1S1_RATIO_PCT_CALC", "FG2S1_RATIO_PCT_CALC",
        ],
        "Gly: fucosylation": [
            "FNTOTAL", "FG0VSG0", "FG1VSG1", "FG2VSG2",
            "FBN", "FBG0VSG0", "FBG1VSG1", "FBG2VSG2",
        ],
        "Gly: galactosylation": ["G0N", "G1N", "G2N", "GAL", "GI", "G2FS2"],
    }
    for mod_name, feats in gly_modules.items():
        idxs = []
        for fn, raw in gly_raw.items():
            if raw in feats:
                idxs.append(feature_names.index(fn))
        if len(idxs) < int(min_features_per_module):
            continue
        v = phi_class[:, idxs].mean(axis=1)
        module_vec[mod_name] = v
        module_imp[mod_name] = float(np.mean(np.abs(v)))
        module_val[mod_name] = float(np.mean(v))
        module_mod[mod_name] = "gly"
        module_n_features[mod_name] = len(idxs)

    # =========================
    # Mass modules (KEGG)
    # =========================
    def _find_first_existing(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p and Path(p).exists():
                return Path(p)
        return None

    if feature_to_cpd_json is None:
        feature_to_cpd_json = _find_first_existing([
            Path("clinical_explain_results/enrichment_results/mass_kegg/audit/mapping/feature_to_cpd.json"),
            Path("clinical_explain_results/enrichment_results/mass_kegg/mapping/feature_to_cpd.json"),
            Path("clinical_explain_results/enrichment_results/mass_kegg/feature_to_cpd.json"),
        ])
    if pathway_to_cids_json is None:
        pathway_to_cids_json = _find_first_existing([
            Path("kegg_cache/kegg_hsa_pathway_to_cids.json"),
            Path("clinical_explain_results/enrichment_results/kegg_cache/kegg_hsa_pathway_to_cids.json"),
            Path("clinical_explain_results/enrichment_results/mass_kegg/kegg_cache/kegg_hsa_pathway_to_cids.json"),
        ])

    def _candidate_pathways_from_mass_csv(csv_path: Optional[Path], fdr_thr: float, topn: int = 12) -> Optional[List[str]]:
        if csv_path is None or not Path(csv_path).exists():
            return None
        dfm = pd.read_csv(csv_path)
        if not {"pathway", "class", "NES", "FDR"}.issubset(dfm.columns):
            return None
        dfm["NES"] = pd.to_numeric(dfm["NES"], errors="coerce")
        dfm["FDR"] = pd.to_numeric(dfm["FDR"], errors="coerce")
        dfm = dfm.dropna(subset=["NES", "FDR"])
        if dfm.empty:
            return None
        dfm = dfm[~dfm["pathway"].str.lower().str.contains("metabolic pathways", na=False)]
        df_sig = dfm[dfm["FDR"] <= fdr_thr].copy()
        if df_sig.empty:
            df_sig = dfm.copy()
        paths = []
        for cls in df_sig["class"].unique():
            dfc = df_sig[df_sig["class"] == cls].copy()
            dfc["score"] = dfc["NES"].abs()
            paths.extend(dfc.sort_values("score", ascending=False)["pathway"].head(topn).tolist())
        return list(dict.fromkeys(paths))

    feat2cpd = {}
    path2cids = {}
    if feature_to_cpd_json and pathway_to_cids_json:
        try:
            with open(feature_to_cpd_json, "r", encoding="utf-8") as f:
                feat2cpd = json.load(f)
            with open(pathway_to_cids_json, "r", encoding="utf-8") as f:
                path2cids = json.load(f)
        except Exception:
            feat2cpd = {}
            path2cids = {}

    if feat2cpd and path2cids:
        candidate_paths = _candidate_pathways_from_mass_csv(mass_kegg_csv, mass_fdr_threshold, topn=12)
        if candidate_paths is None:
            candidate_paths = list(path2cids.keys())

        feat_idx = {fn: i for i, fn in enumerate(feature_names)}

        mass_modules = []
        for p in candidate_paths:
            if "metabolic pathways" in p.lower():
                continue
            cids = set(path2cids.get(p, []))
            if not cids:
                continue

            idxs, weights = [], []
            for feat, lst in feat2cpd.items():
                if feat not in feat_idx:
                    continue
                w_sum = 0.0
                for item in lst:
                    cid = item.get("cid")
                    w = float(item.get("weight", 1.0))
                    if cid in cids:
                        w_sum += w
                if w_sum > 0:
                    idxs.append(feat_idx[feat])
                    weights.append(w_sum)

            if len(idxs) < int(min_features_per_module):
                continue

            W = np.array(weights, dtype=float)
            W = W / (W.sum() + 1e-12)
            v = (phi_class[:, idxs] * W[None, :]).sum(axis=1)
            mass_modules.append((
                p,
                v,
                float(np.mean(np.abs(v))),
                float(np.mean(v)),
                len(idxs),
            ))

        mass_modules = sorted(mass_modules, key=lambda x: x[2], reverse=True)[:mass_top_k]
        for p, v, imp, val, n_features in mass_modules:
            label = _short_mass_label(p)
            if not label:
                continue
            module_vec[label] = v
            module_imp[label] = imp
            module_val[label] = val
            module_mod[label] = "mass"
            module_n_features[label] = n_features

    modules = list(module_vec.keys())
    if len(modules) < 4:
        warnings.warn(f"Only {len(modules)} modules found")
        save_placeholder_panel(save_path, f"Only {len(modules)} modules", figsize=(6, 5), preview_png=True)
        return

    mod_order = {"rna": 0, "mass": 1, "gly": 2}
    modules = sorted(modules, key=lambda m: (mod_order.get(module_mod[m], 9), -module_imp[m], m))

    def _bh_fdr(pvals):
        p = np.asarray(pvals, dtype=float)
        m = p.size
        if m == 0:
            return []
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * m / (np.arange(1, m + 1))
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)
        out = np.empty_like(q)
        out[order] = q
        return out.tolist()

    edges = []
    rng = np.random.default_rng(GLOBAL_SEED)
    n_samples = phi_class.shape[0]

    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            mi, mj = modules[i], modules[j]
            # Cross-modal mode excludes within-modality pairs before testing
            # and multiple correction. In all-pairs mode, BH correction is
            # applied to the complete module-pair family.
            if edge_scope == "cross_modal" and module_mod[mi] == module_mod[mj]:
                continue
            vec_i, vec_j = module_vec[mi], module_vec[mj]
            rho, p = spearmanr(vec_i, vec_j)
            if np.isnan(rho):
                continue

            boot_rhos = []
            for _ in range(bootstrap_n):
                boot_idx = rng.integers(0, n_samples, size=n_samples)
                r_boot, _ = spearmanr(vec_i[boot_idx], vec_j[boot_idx])
                if np.isfinite(r_boot):
                    boot_rhos.append(r_boot)
            stability = np.mean(np.sign(boot_rhos) == np.sign(rho)) if boot_rhos else 0.0

            edges.append({"i": i, "j": j, "rho": float(rho), "p": float(p), "stab": float(stability)})

    if not edges:
        warnings.warn("No valid edges found")
        save_placeholder_panel(save_path, "No valid edges found", figsize=(6, 5), preview_png=True)
        return

    qvals = _bh_fdr([e["p"] for e in edges])
    for e, q in zip(edges, qvals):
        e["q"] = float(q)

    imp = np.array([module_imp[m] for m in modules], dtype=float)

    class_label = str(
        cfg.VisualizationConfig.LABEL_NAMES.get(class_index, f"class {class_index}")
    )
    label_to_index = {
        str(label): int(idx)
        for idx, label in cfg.VisualizationConfig.LABEL_NAMES.items()
    }
    active_label_index = label_to_index.get("Active")
    stable_label_index = label_to_index.get("Stable")
    control_label_index = label_to_index.get("Control")

    if y_true is not None and len(y_true) != phi_class.shape[0]:
        warnings.warn(
            "y_true length does not match SHAP samples; group-specific node "
            "colour metrics are unavailable."
        )
        y_true = None

    def _group_mean(vector: np.ndarray, label_index: Optional[int]) -> float:
        if y_true is None or label_index is None:
            return np.nan
        mask = y_true == label_index
        if not np.any(mask):
            return np.nan
        return float(np.mean(vector[mask]))

    def _hedges_g_active_vs_control(vector: np.ndarray) -> float:
        """Bias-corrected standardized Active-Control difference."""
        if (
            y_true is None
            or active_label_index is None
            or control_label_index is None
        ):
            return np.nan
        active_values = np.asarray(vector[y_true == active_label_index], dtype=float)
        control_values = np.asarray(vector[y_true == control_label_index], dtype=float)
        n_active = active_values.size
        n_control = control_values.size
        if n_active < 2 or n_control < 2:
            return np.nan
        pooled_variance = (
            (n_active - 1) * np.var(active_values, ddof=1)
            + (n_control - 1) * np.var(control_values, ddof=1)
        ) / (n_active + n_control - 2)
        if pooled_variance <= 0 or not np.isfinite(pooled_variance):
            return np.nan
        cohen_d_value = (
            float(np.mean(active_values)) - float(np.mean(control_values))
        ) / math.sqrt(pooled_variance)
        correction = 1.0 - 3.0 / (4.0 * (n_active + n_control) - 9.0)
        return float(correction * cohen_d_value)

    module_group_stats = {}
    for module_name in modules:
        vector = module_vec[module_name]
        active_mean = _group_mean(vector, active_label_index)
        stable_mean = _group_mean(vector, stable_label_index)
        control_mean = _group_mean(vector, control_label_index)
        module_group_stats[module_name] = {
            "mean_signed_all": float(np.mean(vector)),
            "mean_signed_active": active_mean,
            "mean_signed_stable": stable_mean,
            "mean_signed_control": control_mean,
            "active_minus_control": (
                active_mean - control_mean
                if np.isfinite(active_mean) and np.isfinite(control_mean)
                else np.nan
            ),
            "hedges_g_active_minus_control": _hedges_g_active_vs_control(vector),
        }

    if node_color_metric == "standardized_active_minus_control":
        if y_true is None or active_label_index is None or control_label_index is None:
            raise ValueError(
                "node_color_metric='standardized_active_minus_control' requires "
                "y_true and Active/Control entries in VisualizationConfig.LABEL_NAMES."
            )
        val = np.array([
            module_group_stats[m]["hedges_g_active_minus_control"]
            for m in modules
        ], dtype=float)
        node_color_description = "Active vs Control"
    elif node_color_metric == "active_minus_control":
        if y_true is None or active_label_index is None or control_label_index is None:
            raise ValueError(
                "node_color_metric='active_minus_control' requires y_true and "
                "Active/Control entries in VisualizationConfig.LABEL_NAMES."
            )
        val = np.array([
            module_group_stats[m]["active_minus_control"] for m in modules
        ], dtype=float)
        node_color_description = "Active − Control"
    elif node_color_metric == "active_only":
        if y_true is None or active_label_index is None:
            raise ValueError(
                "node_color_metric='active_only' requires y_true and an Active "
                "entry in VisualizationConfig.LABEL_NAMES."
            )
        val = np.array([
            module_group_stats[m]["mean_signed_active"] for m in modules
        ], dtype=float)
        node_color_description = "Active samples"
    else:
        val = np.array([
            module_group_stats[m]["mean_signed_all"] for m in modules
        ], dtype=float)
        node_color_description = "all samples"

    module_table = pd.DataFrame({
        "module": modules,
        "modality": [module_mod[m] for m in modules],
        "n_input_features": [module_n_features.get(m, np.nan) for m in modules],
        "mean_absolute_module_shap": imp,
        "mean_signed_module_shap_all_samples": [
            module_group_stats[m]["mean_signed_all"] for m in modules
        ],
        "mean_signed_module_shap_active": [
            module_group_stats[m]["mean_signed_active"] for m in modules
        ],
        "mean_signed_module_shap_stable": [
            module_group_stats[m]["mean_signed_stable"] for m in modules
        ],
        "mean_signed_module_shap_control": [
            module_group_stats[m]["mean_signed_control"] for m in modules
        ],
        "delta_mean_module_shap_active_minus_control": [
            module_group_stats[m]["active_minus_control"] for m in modules
        ],
        "hedges_g_module_shap_active_vs_control": [
            module_group_stats[m]["hedges_g_active_minus_control"]
            for m in modules
        ],
        "node_color_metric": [node_color_metric] * len(modules),
        "node_color_value": val,
        "explained_output": [f"{class_label}-class logit"] * len(modules),
    })
    module_table_path = Path(save_path).with_name(
        f"{Path(save_path).stem}_modules.csv"
    )

    smin, smax = 320, 1850
    R = 1.08
    label_r = 1.17

    x_offset = 0.34
    xlim, ylim = (-1.72, 2.24), (-1.55, 1.55)

    solid_alpha = 0.90
    dashed_alpha = 0.68

    imp_min = float(np.min(imp))
    imp_max = float(np.max(imp))

    def _importance_to_area(values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if imp_max - imp_min > 1e-12:
            return (
                smin
                + (values - imp_min) / (imp_max - imp_min) * (smax - smin)
            )
        return np.full_like(values, (smin + smax) / 2.0)

    sizes = _importance_to_area(imp)

    # Display node colours in a power-of-ten scaled unit for legibility. This
    # rescales only colourbar labels; relative colours/statistics are unchanged.
    vmax_node_raw = float(np.max(np.abs(val))) if val.size else 0.0
    node_display_exponent = (
        0
        if node_color_metric == "standardized_active_minus_control"
        else int(np.floor(np.log10(vmax_node_raw))) if vmax_node_raw > 0 else 0
    )
    node_display_scale = 10.0 ** (-node_display_exponent)
    val_display = val * node_display_scale
    vmax_node = max(1e-6, float(np.max(np.abs(val_display))))
    node_norm = Normalize(vmin=-vmax_node, vmax=vmax_node)
    node_cmap = plt.get_cmap("PiYG")
    edge_norm = Normalize(vmin=-1, vmax=1)
    edge_cmap = plt.get_cmap("PuOr")

    # Pre-defined, auditable line-style rule. The BH family is unchanged:
    # all candidate pairs within ``edge_scope`` are corrected together before
    # either display category is assigned.
    solid_edges = [e for e in edges if e["q"] < float(edge_fdr)]
    dashed_edges = [
        e for e in edges
        if float(edge_fdr) <= e["q"] < float(dashed_edge_fdr)
    ]
    displayed_edges = solid_edges + dashed_edges

    display_module_indices = sorted({
        idx
        for edge in displayed_edges
        for idx in (edge["i"], edge["j"])
    })
    display_modules = [modules[i] for i in display_module_indices]
    omitted_modules = [m for m in modules if m not in set(display_modules)]

    module_table["shown_in_network"] = module_table["module"].isin(display_modules)
    module_table["node_display_rule"] = (
        "participates in >=1 displayed pooled association: "
        f"solid BH q < {edge_fdr:g}; dashed {edge_fdr:g} <= BH q "
        f"< {dashed_edge_fdr:g}"
    )
    module_table_path.parent.mkdir(parents=True, exist_ok=True)
    module_table.to_csv(module_table_path, index=False, encoding="utf-8")

    def _edge_display_category(edge: dict) -> str:
        if edge["q"] < float(edge_fdr):
            return "solid"
        if edge["q"] < float(dashed_edge_fdr):
            return "dashed"
        return "not shown"

    edge_table = pd.DataFrame([
        {
            "source": modules[e["i"]],
            "target": modules[e["j"]],
            "source_modality": module_mod[modules[e["i"]]],
            "target_modality": module_mod[modules[e["j"]]],
            "edge_type": (
                "within-modality"
                if module_mod[modules[e["i"]]] == module_mod[modules[e["j"]]]
                else "cross-modal"
            ),
            "rho": e["rho"],
            "p_value": e["p"],
            "q_value_bh": e["q"],
            "bootstrap_sign_stability": e["stab"],
            "multiple_testing_scope": edge_scope,
            "n_pairs_in_bh_family": len(edges),
            "passes_bh_fdr": e["q"] < float(edge_fdr),
            "in_dashed_q_interval": (
                float(edge_fdr) <= e["q"] < float(dashed_edge_fdr)
            ),
            "line_style": _edge_display_category(e),
            "display_rule": (
                f"solid: BH q < {edge_fdr:g}; dashed: {edge_fdr:g} "
                f"<= BH q < {dashed_edge_fdr:g}"
            ),
            "shown": e in displayed_edges,
        }
        for e in edges
    ])
    edge_table_path = Path(save_path).with_name(f"{Path(save_path).stem}_edges.csv")
    edge_table_path.parent.mkdir(parents=True, exist_ok=True)
    edge_table.to_csv(edge_table_path, index=False, encoding="utf-8")

    if not displayed_edges:
        warnings.warn(
            f"No {edge_scope} pooled SHAP associations met BH q < "
            f"{dashed_edge_fdr:g}."
        )
        save_placeholder_panel(
            save_path,
            f"No pooled SHAP associations at BH q < {dashed_edge_fdr:g}",
            figsize=(7.0, 5.5),
            preview_png=True,
        )
        return

    display_angles = (
        np.linspace(0, 2 * np.pi, len(display_module_indices), endpoint=False)
        + np.pi / 2
    )
    angle_by_index = dict(zip(display_module_indices, display_angles))
    pos = {
        old_index: (
            x_offset + R * np.cos(angle_by_index[old_index]),
            R * np.sin(angle_by_index[old_index]),
        )
        for old_index in display_module_indices
    }

    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_position([0.18, 0.08, 0.64, 0.84])

    for e in displayed_edges:
        x1, y1 = pos[e["i"]]
        x2, y2 = pos[e["j"]]
        lw = 1.0 + 5.0 * abs(e["rho"])
        is_solid = e["q"] < float(edge_fdr)

        rad = 0.25 if (e["i"] + e["j"]) % 2 == 0 else -0.25
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=float(lw),
            linestyle="-" if is_solid else (0, (4.0, 3.0)),
            color=edge_cmap(edge_norm(e["rho"])),
            alpha=float(solid_alpha if is_solid else dashed_alpha),
            zorder=2 if is_solid else 1.8,
        ))

    for i in display_module_indices:
        m = modules[i]
        x, y = pos[i]
        ax.scatter(
            [x], [y],
            s=float(sizes[i]),
            c=[node_cmap(node_norm(val_display[i]))],
            edgecolors=MODALITY_COLORS.get(module_mod[m], "black"),
            linewidths=2.2,
            zorder=3
        )

    for i in display_module_indices:
        m = modules[i]
        a = angle_by_index[i]
        lx = x_offset + label_r * R * np.cos(a)
        ly = label_r * R * np.sin(a)
        ha = "left" if lx >= x_offset else "right"
        display_label = m
        if m == "Mass: pantothenate/CoA biosynthesis":
            display_label = "Mass: pantothenate/CoA\nbiosynthesis"
        elif m == "Mass: arginine/proline metabolism":
            display_label = "Mass: arginine/proline\nmetabolism"
        fz = 9 if module_mod[m] == "mass" else 10
        ax.text(
            lx,
            ly,
            display_label,
            ha=ha,
            va="center",
            fontsize=fz,
            fontweight="bold",
            linespacing=0.95,
        )

    legend_kw = dict(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.75,
        handletextpad=0.7,
    )

    lw_vals = [0.12, 0.28, 0.50]
    lw_handles = [
        Line2D([0], [0], color="#444", lw=1.0 + 5.0 * v, linestyle="-")
        for v in lw_vals
    ]
    leg1 = fig.legend(
        lw_handles, [f"|ρ| = {v:.2f}" for v in lw_vals],
        title="|Spearman's ρ|",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.82),
        bbox_transform=fig.transFigure,
        fontsize=9,
        title_fontsize=10,
        handlelength=2.0,
        **legend_kw
    )

    q_style_handles = [
        Line2D([0], [0], color="#555", lw=2.2, linestyle="-"),
        Line2D([0], [0], color="#555", lw=2.2, linestyle=(0, (4.0, 3.0))),
    ]
    leg_q = fig.legend(
        q_style_handles,
        [
            rf"$q < {edge_fdr:.2f}$",
            rf"${edge_fdr:.2f} \leq q < {dashed_edge_fdr:.2f}$",
        ],
        title="BH q value",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.58),
        bbox_transform=fig.transFigure,
        fontsize=8.5,
        title_fontsize=9.5,
        handlelength=2.4,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        borderpad=0.7,
        labelspacing=0.65,
        handletextpad=0.7,
    )

    display_imp = imp[display_module_indices]
    imp_q = np.percentile(display_imp, [15, 50, 85])
    # Matplotlib scatter uses marker area (pt^2), whereas Line2D legend
    # markers use diameter (pt). Taking sqrt preserves the exact visual map.
    node_imp_areas = _importance_to_area(imp_q)
    node_imp_marker_sizes = np.sqrt(node_imp_areas)
    size_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=ms,
            markerfacecolor="#8f8f8f",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        for ms in node_imp_marker_sizes
    ]

    # The marker diameters retain the exact sqrt(area) mapping used by scatter.
    # A deliberately tall handle box keeps the largest key from overlapping
    # either the adjacent key or its label.
    leg2 = fig.legend(
        size_handles,
        [f"{r:.1e}" for r in imp_q],
        title="Mean |aggregated module SHAP|",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.04),
        bbox_transform=fig.transFigure,
        ncol=1,
        fontsize=8,
        title_fontsize=9,
        # Reserve enough horizontal room for the largest bubble before text.
        handlelength=4.5,
        handleheight=5.4,
        handletextpad=0.9,
        labelspacing=1.2,
        borderpad=0.9,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )

    cax_edge = fig.add_axes([0.89, 0.53, 0.020, 0.24])
    cb1 = fig.colorbar(ScalarMappable(norm=edge_norm, cmap=edge_cmap), cax=cax_edge)
    cb1.set_label("Pooled Spearman's ρ", rotation=270, labelpad=14, fontsize=10)
    cb1.ax.tick_params(labelsize=8)

    cax_node = fig.add_axes([0.89, 0.20, 0.020, 0.24])
    cb2 = fig.colorbar(ScalarMappable(norm=node_norm, cmap=node_cmap), cax=cax_node)
    if node_color_metric == "standardized_active_minus_control":
        node_colorbar_label = (
            "Hedges' g of module SHAP\n"
            f"({node_color_description})"
        )
    elif node_color_metric == "active_minus_control":
        node_colorbar_label = (
            "Δ mean module SHAP\n"
            f"({node_color_description}; ×10$^{{{node_display_exponent}}}$)"
        )
    else:
        node_colorbar_label = (
            "Mean signed module SHAP\n"
            f"({node_color_description}; ×10$^{{{node_display_exponent}}}$)"
        )
    cb2.set_label(
        node_colorbar_label,
        rotation=270,
        labelpad=24,
        fontsize=9,
    )
    cb2.ax.tick_params(labelsize=8)

    network_title = (
        "Pooled cross-modal SHAP association network"
        if edge_scope == "cross_modal"
        else "Pooled SHAP association network"
    )
    ax.set_title(network_title, fontsize=16, fontweight="bold", pad=10)

    save_figure_pdf_and_preview(
        fig,
        save_path,
        preview_png=True,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        extra_artists=(leg1, leg_q, leg2),
    )
    plt.close(fig)

    print(
        f"[Module Network] Modules tested: {len(modules)}, "
        f"modules displayed: {len(display_modules)}, "
        f"edge scope: {edge_scope}, edges tested: {len(edges)}, "
        f"solid edges: {len(solid_edges)}, dashed edges: {len(dashed_edges)}"
    )
    if omitted_modules:
        print(
            "[Module Network] Modules omitted from the displayed-association "
            "subnetwork: " + "; ".join(omitted_modules)
        )
    print(f"[Module Network] Edge statistics saved to: {edge_table_path}")
    print(f"[Module Network] Module statistics saved to: {module_table_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate multimodal biological interpretation plots."
    )
    ap.add_argument("--rna_csv", type=str, default=cfg.RNA_PATH)
    ap.add_argument("--gly_csv", type=str, default=cfg.GLY_PATH)

    ap.add_argument("--msigdb_h", type=str, default=None)
    ap.add_argument("--msigdb_reactome", type=str, default=None)
    ap.add_argument("--msigdb_gobp", type=str, default=None)
    ap.add_argument("--gsea_perm", type=int, default=500)
    ap.add_argument("--skip_gsea", action="store_true")

    ap.add_argument(
        "--mass_kegg_csv",
        type=str,
        default="clinical_explain_results/enrichment_results/mass_kegg/mass_kegg_classwise.csv",
    )
    ap.add_argument(
        "--shap_npz",
        type=str,
        default="clinical_explain_results/shap_outputs/mor_shap_outputs.npz",
    )
    ap.add_argument(
        "--network_node_color",
        choices=[
            "standardized_active_minus_control",
            "active_minus_control",
            "active_only",
            "mean_signed_all",
        ],
        default="standardized_active_minus_control",
        help=(
            "Node-colour definition for the module network. The default is "
            "the standardized Active-Control difference (Hedges' g) in module SHAP."
        ),
    )
    ap.add_argument(
        "--network_edge_scope",
        choices=["cross_modal", "all"],
        default="cross_modal",
        help=(
            "Test and display cross-modal module pairs only or all module pairs. "
            "BH correction follows this scope."
        ),
    )
    ap.add_argument(
        "--network_dashed_fdr",
        type=float,
        default=0.10,
        help=(
            "Upper BH-q boundary for dashed exploratory network edges. Solid "
            "edges use q < 0.05; dashed edges use 0.05 <= q < this value."
        ),
    )

    ap.add_argument("--outdir", type=str, default="clinical_explain_results/multimodal_interpretation")
    args = ap.parse_args()

    configure_matplotlib_for_editable_vector()

    outdir = Path(args.outdir)
    panels = outdir / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # RNA pathway enrichment
    # ------------------------------------------------------------------
    panel_a = panels / "rna_pathway_enrichment.pdf"

    gmt_dict = {}
    if args.msigdb_h and Path(args.msigdb_h).exists():
        gmt_dict["Hallmark"] = Path(args.msigdb_h)
    if args.msigdb_reactome and Path(args.msigdb_reactome).exists():
        gmt_dict["Reactome"] = Path(args.msigdb_reactome)
    if args.msigdb_gobp and Path(args.msigdb_gobp).exists():
        gmt_dict["GO-BP"] = Path(args.msigdb_gobp)

    panel_a_done = False
    if not args.skip_gsea and gmt_dict and HAVE_GSEAPY:
        gsea_data = run_rna_gsea(
            Path(args.rna_csv),
            gmt_dict,
            outdir / "rna_gsea",
            permutations=args.gsea_perm,
        )
        if gsea_data:
            # A requested pathway is retained only
            # when q < 0.10 in at least one contrast; both available contrast
            # rows are then plotted. The order below is the display order.
            RNA_PATHWAYS_TO_DISPLAY = {
                "Hallmark": [
                    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
                    "HALLMARK_INFLAMMATORY_RESPONSE",
                    # These pathways are omitted unless q < 0.10.
                    "HALLMARK_COMPLEMENT",
                    "HALLMARK_P53_PATHWAY",
                    "HALLMARK_ALLOGRAFT_REJECTION",
                ],
                "Reactome": [
                    "REACTOME_INTERFERON_SIGNALING",
                    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM",
                    "REACTOME_TRANSLATION",
                    "REACTOME_EUKARYOTIC_TRANSLATION_INITIATION",
                    "REACTOME_RRNA_PROCESSING",
                    "REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION",
                ],
                "GO-BP": [
                    "GOBP_DEFENSE_RESPONSE_TO_VIRUS",
                    "GOBP_RESPONSE_TO_VIRUS",
                    "GOBP_REGULATION_OF_VIRAL_LIFE_CYCLE",
                    "GOBP_ANTIVIRAL_INNATE_IMMUNE_RESPONSE",
                    "GOBP_NEGATIVE_REGULATION_OF_VIRAL_GENOME_REPLICATION",
                    "GOBP_VIRAL_GENOME_REPLICATION",
                    "GOBP_VIRAL_LIFE_CYCLE",
                ],
            }
            panel_a_done = bool(plot_rna_stacked_dotplot(
                data_dict=gsea_data,
                save_path=panel_a,
                top_n_per_db=6,
                sort_by_class="Active",
                fdr_threshold=0.10,
                fdr_display_floor=1.0 / (args.gsea_perm + 1.0),
                force_top_n=False,
                group_gap=0.08,
                # Use compact panel geometry.
                fig_w=3.2,
                max_label_len=38,
                selected_pathways_by_db=RNA_PATHWAYS_TO_DISPLAY,
                selected_require_any_significant=True,
                audit_csv_path=panels / "rna_gsea_displayed_pathways.csv",
            ))

    if not panel_a_done:
        save_placeholder_panel(
            panel_a,
            "RNA enrichment\n(not computed)",
            figsize=(5.2, 5.6),
            preview_png=True,
        )

    ensure_panel_pdf_exists(panel_a, "RNA pathway enrichment")

    # ------------------------------------------------------------------
    # Glycan traits
    # ------------------------------------------------------------------
    panel_b = panels / "glycan_trait_summary.pdf"

    traits = compute_gly_traits(Path(args.gly_csv))
    candidate_traits = [
        t for t in [
            "STotal", "S1Total", "bisecting", "G0n", "G1n", "G2n", "GI",
            "S_total_calc", "S1_total_calc", "FG1S1_ratio_pct_calc", "FG2S1_ratio_pct_calc"
        ]
        if t in traits.columns
    ]

    if candidate_traits:
        plot_gly_trait_forest_with_distribution(
            traits_df=traits,
            trait_cols=candidate_traits,
            save_path=panel_b,
            ref_group="Control",
            comp_group="Active",
            n_traits_for_dist=4,
            dist_traits=["GI", "bisecting", "STotal", "FG1S1"],
        )
    else:
        save_placeholder_panel(
            panel_b,
            "Glycan traits\n(not available)",
            figsize=(8.2, 4.8),
            preview_png=True,
        )

    ensure_panel_pdf_exists(panel_b, "Glycan trait summary")

    # ------------------------------------------------------------------
    # Mass-spectrometry pathway enrichment
    # ------------------------------------------------------------------
    panel_c = panels / "mass_pathway_enrichment.pdf"
    mass_ora_new_path = Path(
        "clinical_explain_results/enrichment_results/mass_kegg_ora_biological/mass_ora_biological_direction.csv"
    )

    panel_c_done = False
    if mass_ora_new_path.exists() and HAVE_MASS_ORA_MODULE and hasattr(mass_ora_mod, "plot_ora_dotplot_biological"):
        print("[Mass pathway enrichment] Using the prespecified pathway list")
        df_mass = pd.read_csv(mass_ora_new_path)

        SELECTED_PATHWAYS = [
            "Valine, leucine and isoleucine biosynthesis",
            "Valine, leucine and isoleucine degradation",
            "Tyrosine metabolism",
            "Central carbon metabolism in cancer",
            "Pantothenate and CoA biosynthesis",
            "Phenylalanine, tyrosine and tryptophan biosynthesis",
            "Butanoate metabolism",
        ]

        panel_c_done = bool(mass_ora_mod.plot_ora_dotplot_biological(
            df=df_mass,
            save_path=panel_c,
            top_n=12,
            fdr_threshold=0.10,
            p_threshold=0.05,
            require_both=True,
            reference_class="Control",
            exclude_reference=True,
            selected_pathways=SELECTED_PATHWAYS,
        ))

    if not panel_c_done:
        save_placeholder_panel(
            panel_c,
            "Mass KEGG\n(not available)",
            figsize=(6.8, 5.2),
            preview_png=True,
        )

    ensure_panel_pdf_exists(panel_c, "Mass pathway enrichment")

    # ------------------------------------------------------------------
    # Cross-modal module network
    # ------------------------------------------------------------------
    panel_d = panels / "cross_modal_module_network.pdf"

    plot_module_interaction_network_circular_style(
        shap_npz_path=Path(args.shap_npz),
        save_path=panel_d,
        class_index=None,
        bootstrap_n=500,
        edge_fdr=0.05,
        dashed_edge_fdr=args.network_dashed_fdr,
        mass_fdr_threshold=0.05,
        node_color_metric=args.network_node_color,
        edge_scope=args.network_edge_scope,
        rna_gmt_path=(
            Path(args.msigdb_h)
            if args.msigdb_h
            else Path("ref/msigdb/h.all.v2025.1.Hs.symbols.gmt")
        ),
        rna_pathway_csv=Path("clinical_explain_results/enrichment_results/rna/rna_all_pathway_importance.csv"),
        mass_kegg_csv=Path(args.mass_kegg_csv),
    )

    ensure_panel_pdf_exists(panel_d, "Cross-modal module network")

    print(f"[OK] Multimodal interpretation plots saved in: {panels}")


if __name__ == "__main__":
    main()
