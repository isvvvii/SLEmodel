# enrichment_plot_utils.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import textwrap
import re
from typing import Tuple, Dict, List, Optional

try:
    from pycirclize import Circos
    _HAVE_PYCIRCLIZE = True
except ImportError:
    _HAVE_PYCIRCLIZE = False

try:
    from adjustText import adjust_text
    _HAVE_ADJUSTTEXT = True
except ImportError:
    _HAVE_ADJUSTTEXT = False
try:
    import networkx as nx
    _HAVE_NX = True
except ImportError:
    _HAVE_NX = False


GLOBAL_SEED = 202510
MORANDI_CMAP = LinearSegmentedColormap.from_list(
    "morandi_pink_blue", ["#4A6A91", "#A8B8D0", "#F0EBE8", "#E8C7C3", "#B36A6F"]
)

def set_publication_style():
    """
    Apply consistent figure styling and preserve editable text in PDF output.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,

        # ---- font / text ----
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "text.usetex": False,
        "axes.unicode_minus": False,

        # ---- editable vector output ----
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",

        # ---- general style ----
        "axes.labelsize": 11,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def save_publication_figure(
    fig,
    save_path: Path,
    *,
    preview_png: bool = True,
    png_dpi: int = 600,
    bbox_inches="tight",
    pad_inches: float = 0.02,
    facecolor: str = "white",
    transparent: bool = False,
    bbox_extra_artists=None,
):
    """
    Save a vector PDF and, when requested, a PNG preview.

    Returns the PDF and PNG paths.
    """
    set_publication_style()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = save_path.suffix.lower()

    if suffix == ".pdf":
        pdf_path = save_path
        png_path = save_path.with_suffix(".png") if preview_png else None
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        pdf_path = save_path.with_suffix(".pdf")
        png_path = save_path if preview_png else None
    else:
        pdf_path = save_path.with_suffix(".pdf")
        png_path = save_path.with_suffix(".png") if preview_png else None

    common_kwargs = dict(
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        facecolor=facecolor,
        transparent=transparent,
    )
    if bbox_extra_artists is not None:
        common_kwargs["bbox_extra_artists"] = tuple(bbox_extra_artists)

    fig.savefig(pdf_path, format="pdf", **common_kwargs)

    if png_path is not None:
        fig.savefig(png_path, format="png", dpi=png_dpi, **common_kwargs)

    return pdf_path, png_path

def _cleanup_pathway_name(name: str, max_len: int = 45) -> str:
    name = re.sub(r"^(HALLMARK_|REACTOME_|GOBP_)", "", name)
    name = name.replace("_", " ").capitalize()
    return "\n".join(textwrap.wrap(name, max_len))

def plot_rna_pathway_importance_bars(
    importance_df: pd.DataFrame,
    save_path: Path,
    top_n_pathways: int = 12,
    sort_by_class: str = "Active",
    figsize: Tuple[float, float] = (10, 8),
    class_order: Optional[List[str]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    max_label_len: int = 40,
):
    """
    RNA pathway importance plot using grouped horizontal bars.

    Expected columns in importance_df:
      - pathway (str)
      - class (str): Active/Stable/Control
      - importance (float): e.g., mean |SHAP|
    Optional:
      - database (str)

    The plot shows, for each pathway, the importance across classes.
    """
    set_publication_style()

    df = importance_df.copy()
    if df.empty:
        warnings.warn("Empty importance_df, skipping plot.")
        return

    need_cols = {"pathway", "class", "importance"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"importance_df must contain columns {need_cols}, got {set(df.columns)}")

    df["pathway"] = df["pathway"].astype(str)
    df["class"] = df["class"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["importance"])
    if df.empty:
        warnings.warn("importance_df has no numeric importance values, skipping plot.")
        return

    if class_order is None:
        class_order = [c for c in ["Active", "Stable", "Control"] if c in df["class"].unique().tolist()]
    if not class_order:
        warnings.warn("No valid classes found, skipping plot.")
        return

    if class_colors is None:
        class_colors = {"Active": "#E68B81", "Stable": "#8aab82", "Control": "#7DA6C6"}

    pivot = df.pivot_table(index="pathway", columns="class", values="importance", aggfunc="mean").fillna(0.0)

    # pick top pathways
    if sort_by_class in pivot.columns:
        top_paths = pivot.sort_values(by=sort_by_class, ascending=False).head(top_n_pathways).index.tolist()
    else:
        top_paths = pivot.sum(axis=1).sort_values(ascending=False).head(top_n_pathways).index.tolist()

    pivot = pivot.loc[top_paths, [c for c in class_order if c in pivot.columns]].copy()

    # nicer labels
    def _clean_pw(p: str) -> str:
        return _cleanup_pathway_name(p, max_len=max_label_len)

    pivot.index = [_clean_pw(p) for p in pivot.index]
    pivot = pivot.iloc[::-1]  # top at top (barh)

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(pivot.shape[0])
    n_cls = pivot.shape[1]
    bar_h = 0.8 / max(n_cls, 1)
    offsets = (np.arange(n_cls) - (n_cls - 1) / 2.0) * bar_h

    for j, cls in enumerate(pivot.columns.tolist()):
        ax.barh(
            y + offsets[j],
            pivot[cls].values,
            height=bar_h * 0.95,
            color=class_colors.get(cls, "#999999"),
            edgecolor="white",
            linewidth=0.6,
            label=cls,
            alpha=0.95,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_xlabel("Mean |SHAP| (pathway importance)")
    ax.set_title("RNA Pathway Importance", fontsize=12, fontweight="bold", pad=8)

    ax.grid(True, axis="x", ls="--", lw=0.6, alpha=0.25)
    sns.despine(ax=ax, left=False, bottom=True)

    ax.legend(loc="lower right", frameon=False, fontsize=9)

    save_path = Path(save_path)
    save_publication_figure(
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

def plot_rna_stacked_dotplot(
    data_dict: dict[str, pd.DataFrame],
    save_path: Path,
    top_n_per_db: int = 7,
    sort_by_class: str = "Active",
    fdr_threshold: float = 0.10,
    fdr_display_floor: float = 1e-3,
    group_gap: float = 0.08,
    force_top_n: bool = False,
    fig_w: float = 3.2,
    max_label_len: int = 38,
    exclude_keywords_by_db: Optional[Dict[str, List[str]]] = None,
    exclude_pathways_exact_by_db: Optional[Dict[str, List[str]]] = None,
    exclude_pathways_regex_by_db: Optional[Dict[str, List[str]]] = None,
    selected_pathways_by_db: Optional[Dict[str, List[str]]] = None,
    selected_require_any_significant: bool = True,
    audit_csv_path: Optional[Path] = None,
) -> bool:
    """
    RNA GSEA stacked dotplot.

    ``selected_pathways_by_db`` is an explicit, auditable display whitelist.
    Matching is case/punctuation/prefix insensitive. When
    ``selected_require_any_significant`` is True, a requested pathway is shown
    only if it has FDR < ``fdr_threshold`` in at least one displayed contrast;
    rows for all available contrasts are then retained for comparison.

    The exclusion arguments remain available for automatic-selection use:
      * exclude_keywords_by_db: case-insensitive substring filters
      * exclude_pathways_exact_by_db: case-insensitive exact-name filters
      * exclude_pathways_regex_by_db: case-insensitive regular-expression filters
    """
    set_publication_style()
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.2, 'grid.linestyle': ':'})

    cmap = plt.get_cmap("RdBu_r")

    default_exclude = {
        "Hallmark": [
            "MYC_TARGETS", "INFECTIOUS_DISEASE", "RESPONSES_TO_STIMULI",
        ],
        "Reactome": [
            "Nervous system development".upper(),
            "SRP_DEPENDENT".upper(),
            "NONSENSE_MEDIATED".upper(),
        ],
        "GO-BP": [
            "INFECTIOUS_DISEASE".upper(),
        ],
    }

    base_exclude_keywords = [
        "SELENOAMINO", "ROBO_RECEPTORS", "SLITS_AND_ROBOS",
        "INFLUENZA", "SARS_COV_2", "HIV", "HERPES",
        "SRP_DEPENDENT", "NONSENSE_MEDIATED",
        "SUPRAMOLECULAR", "VASCULATURE", "EPITHELIAL_CELL",
        "MYC_TARGETS", "INFECTIOUS_DISEASE", "RESPONSES_TO_STIMULI",
        "EIF2AK4", "ALPHA_BETA_SIGNALING"
    ]

    def _merge_map(base: Dict[str, List[str]], extra: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        out = {k: list(v) for k, v in base.items()}
        if extra:
            for k, v in extra.items():
                out.setdefault(k, [])
                out[k].extend(list(v))
        return out

    exclude_keywords_by_db = _merge_map(default_exclude, exclude_keywords_by_db or {})
    exclude_pathways_exact_by_db = exclude_pathways_exact_by_db or {}
    exclude_pathways_regex_by_db = exclude_pathways_regex_by_db or {}
    selected_pathways_by_db = selected_pathways_by_db or {}

    def _normalise_pathway_name(value: str) -> str:
        """Normalise MSigDB display names and identifiers for robust matching."""
        value = str(value).strip().upper()
        value = re.sub(r"^(HALLMARK|REACTOME|GOBP|GO_BP)[_:\s-]+", "", value)
        value = re.sub(r"[^A-Z0-9]+", "_", value)
        return value.strip("_")

    all_plot_dfs, y_offset, db_y_ranges = [], 0, {}
    db_order = ["Hallmark", "Reactome", "GO-BP"]

    for db in db_order:
        df = data_dict.get(db)
        if df is None or df.empty:
            continue
        df = df.copy()

        df['pathway'] = df['pathway'].astype(str)
        df['class'] = df['class'].astype(str)

        df['NES'] = pd.to_numeric(df['NES'], errors='coerce')
        df['FDR'] = pd.to_numeric(df['FDR'], errors='coerce')
        df.dropna(subset=['NES', 'FDR'], inplace=True)
        if df.empty:
            continue

        patt_kw = "|".join([re.escape(k) for k in base_exclude_keywords])
        if patt_kw:
            df = df[~df['pathway'].str.upper().str.contains(patt_kw, na=False)]

        kws = [k.upper() for k in exclude_keywords_by_db.get(db, []) if k]
        if kws:
            patt = "|".join([re.escape(k) for k in kws])
            df = df[~df['pathway'].str.upper().str.contains(patt, na=False)]

        exacts = [x.strip().upper() for x in exclude_pathways_exact_by_db.get(db, []) if str(x).strip()]
        if exacts:
            df = df[~df['pathway'].str.upper().isin(exacts)]

        regex_list = [r for r in exclude_pathways_regex_by_db.get(db, []) if str(r).strip()]
        for rx in regex_list:
            try:
                df = df[~df['pathway'].str.contains(rx, case=False, regex=True, na=False)]
            except re.error:
                pass

        if df.empty:
            continue

        # ---- Select pathways ----
        # Explicit selection takes precedence over all automatic ranking.
        pathway_order_override = None
        requested = [
            str(x).strip()
            for x in selected_pathways_by_db.get(db, [])
            if str(x).strip()
        ]

        if requested:
            actual_by_normalised_name = {}
            for actual in df["pathway"].drop_duplicates().tolist():
                actual_by_normalised_name.setdefault(
                    _normalise_pathway_name(actual), actual
                )

            selected_actual = []
            missing = []
            for requested_name in requested:
                actual = actual_by_normalised_name.get(
                    _normalise_pathway_name(requested_name)
                )
                if actual is None:
                    missing.append(requested_name)
                elif actual not in selected_actual:
                    selected_actual.append(actual)

            if missing:
                warnings.warn(
                    f"[{db}] Requested RNA pathways not found and omitted: "
                    + "; ".join(missing)
                )

            if selected_require_any_significant:
                significant_pathways = set(
                    df.loc[df["FDR"] < float(fdr_threshold), "pathway"]
                )
                non_significant = [
                    p for p in selected_actual if p not in significant_pathways
                ]
                if non_significant:
                    warnings.warn(
                        f"[{db}] Requested pathways without FDR < "
                        f"{fdr_threshold:g} in either contrast were omitted: "
                        + "; ".join(non_significant)
                    )
                selected_actual = [
                    p for p in selected_actual if p in significant_pathways
                ]

            df = df[df["pathway"].isin(selected_actual)].copy()
            pathway_order_override = selected_actual

        elif force_top_n:
            df['abs_NES'] = df['NES'].abs()
            top_pathways = set()
            for cls in df['class'].unique():
                cls_df = df[df['class'] == cls]
                sig_df = cls_df[cls_df['FDR'] < fdr_threshold]
                top_pathways.update(
                    sig_df.nlargest(top_n_per_db, 'abs_NES')['pathway'].tolist()
                )
            df = df[df['pathway'].isin(top_pathways)].copy()
            df.drop(columns=['abs_NES'], inplace=True, errors='ignore')
        else:
            # Do not silently fall back to non-significant pathways.
            df = df[df['FDR'] < fdr_threshold].copy()

        if df.empty:
            continue

        pivot = df.pivot_table(index='pathway', columns='class', values='NES', aggfunc='mean')
        if pivot.empty:
            continue

        if pathway_order_override is not None:
            pathway_order = pd.Index([
                p for p in pathway_order_override if p in pivot.index
            ])
        elif sort_by_class in pivot.columns:
            pathway_order = pivot.sort_values(by=sort_by_class, ascending=False).index
        else:
            pathway_order = pivot.max(axis=1).sort_values(ascending=False).index

        plot_df = df.copy()
        plot_df['pathway_cat'] = pd.Categorical(plot_df['pathway'], categories=pathway_order, ordered=True)
        # Display floor only; original FDR values are unchanged.
        plot_df['FDR_for_size'] = plot_df['FDR'].clip(
            lower=float(fdr_display_floor),
            upper=1.0
        )
        plot_df['-log10FDR'] = -np.log10(plot_df['FDR_for_size'])
        plot_df['y_pos'] = plot_df['pathway_cat'].cat.codes + y_offset
        plot_df['database'] = db

        all_plot_dfs.append(plot_df)
        db_y_ranges[db] = (y_offset, y_offset + len(pathway_order) - 1)
        y_offset += len(pathway_order) + max(1, int(round(group_gap * 10)))

    if not all_plot_dfs:
        warnings.warn("No RNA data to plot.")
        return False

    final_df = pd.concat(all_plot_dfs, ignore_index=True)

    if audit_csv_path is not None:
        audit_csv_path = Path(audit_csv_path)
        audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
        audit_cols = ["database", "class", "pathway", "NES", "FDR"]
        final_df[audit_cols].to_csv(audit_csv_path, index=False, encoding="utf-8")

    fig_h = min(18.0, max(6.0, 0.35 * y_offset + 1.5))
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.22, 0.12], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_db = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    classes_order = [c for c in ["Active", "Stable", "Control"] if c in final_df['class'].unique()]
    n_cls = len(classes_order)

    if n_cls == 3:
        pos_vals = np.array([0.0, 0.16, 0.32])
    elif n_cls == 2:
        pos_vals = np.array([0.0, 0.20])
    else:
        pos_vals = np.array([0.0])

    x_pos_map = dict(zip(classes_order, pos_vals))
    final_df['x_num'] = final_df['class'].map(x_pos_map)

    ax.set_xlim(pos_vals.min() - 0.08, pos_vals.max() + 0.08)
    ax.margins(x=0)

    max_abs_nes = max(2.5, float(final_df['NES'].abs().max()))
    norm = Normalize(vmin=-max_abs_nes, vmax=max_abs_nes)

    max_nlog = max(1e-2, float(final_df['-log10FDR'].max()))
    sizes = 12 + 120 * (final_df['-log10FDR'] / max_nlog)

    # Draw every bubble as one opaque Line2D marker.  Unlike a translucent
    # scatter PathCollection, this stays a simple editable vector circle when
    # the PDF is opened in Adobe Illustrator and avoids white/double rims.
    for (_, row), area in zip(final_df.iterrows(), sizes.to_numpy()):
        ax.plot(
            [float(row['x_num'])],
            [float(row['y_pos'])],
            marker='o',
            linestyle='None',
            markersize=float(np.sqrt(area)),
            markerfacecolor=cmap(norm(float(row['NES']))),
            markeredgecolor='black',
            markeredgewidth=0.5,
            alpha=1.0,
            zorder=3,
        )

    def _fix_typos(name: str) -> str:
        name = re.sub(r'\bRrna\b', 'rRNA', name, flags=re.IGNORECASE)
        name = re.sub(r'\bRna\b', 'RNA', name, flags=re.IGNORECASE)
        name = re.sub(r'\bMrna\b', 'mRNA', name, flags=re.IGNORECASE)
        return name

    y_ticks_map = final_df.groupby('pathway_cat', observed=True)['y_pos'].first()
    ax.set_yticks(y_ticks_map.values)
    clean_labels = [_cleanup_pathway_name(p, max_label_len) for p in y_ticks_map.index]
    ax.set_yticklabels([_fix_typos(l) for l in clean_labels], fontsize=9)
    ax.tick_params(axis='y', length=0)

    ax.set_ylim(-1.15, y_offset)
    ax.invert_yaxis()

    ax.set_xticks(pos_vals)
    contrast_labels = {
        "Active": "Active\nvs Control",
        "Stable": "Stable\nvs Control",
        "Control": "Control",
    }
    ax.set_xticklabels(
        [contrast_labels.get(c, c) for c in classes_order],
        fontsize=10,
        fontweight='bold',
        rotation=0,
        ha='center'
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    for i in range(len(pos_vals) - 1):
        mid_x = (pos_vals[i] + pos_vals[i + 1]) / 2
        ax.axvline(mid_x, color='grey', linestyle=':', alpha=0.35, linewidth=0.8)

    ax_db.set_ylim(ax.get_ylim())
    ax_db.axis('off')
    for db, (start, end) in db_y_ranges.items():
        center_y = (start + end) / 2.0
        span = end - start
        h = max(span + 0.8, 2.8)
        rect_top = center_y - h / 2.0
        rect = Rectangle((0.08, rect_top), 0.84, h, facecolor='#F0F0F0', edgecolor='#888888', lw=0.5)
        ax_db.add_patch(rect)
        ax_db.text(0.5, center_y, db, ha='center', va='center',
                   rotation=270, fontsize=9, fontweight='bold', color='#444444')

    # NES colour scale (kept separate from the FDR size legend).
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('NES', rotation=270, labelpad=10, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    def _size_from_fdr(fdr: float) -> float:
        fdr_for_size = np.clip(
            float(fdr),
            float(fdr_display_floor),
            1.0
        )
        return float(
            12 + 120 * (-np.log10(fdr_for_size) / max_nlog)
        )

    # Use a compact three-key legend layout.
    # These are representative size references; the plotted points still use
    # their continuous q values, including values between 0.05 and 0.10.
    legend_fdrs = [float(fdr_display_floor), 0.01, 0.05]
    size_scat = [_size_from_fdr(fdr) for fdr in legend_fdrs]

    handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markersize=float(np.sqrt(s)),
            markerfacecolor='gray',
            markeredgecolor='black',
            markeredgewidth=0.5,
            alpha=1.0,
        )
        for s in size_scat
    ]

    labels = [f"q ≤ {fdr_display_floor:.3f}"] + [
        f"q = {fdr:.2f}" for fdr in legend_fdrs[1:]
    ]

    ax.legend(
        handles,
        labels,
        title="FDR q value",
        bbox_to_anchor=(0.50, -0.07),
        loc='upper center',
        ncol=len(labels),
        frameon=False,
        prop={'size': 8},
        title_fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2
    )

    ax.set_title("RNA Pathway Enrichment", fontsize=12, fontweight='bold', pad=8)

    fig.tight_layout(rect=[0, 0.08, 1.0, 0.98])

    save_publication_figure(
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
    return True

def _pick_best_direction_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain one direction per pathway and class, prioritizing lower FDR and then
    larger absolute NES.
    """
    if df.empty:
        return df
    x = df.copy()
    x["absNES"] = x["NES"].abs()
    x["FDR_sort"] = x["FDR"].fillna(1.0)
    x = x.sort_values(["pathway", "class", "FDR_sort", "absNES"], ascending=[True, True, True, False])
    x = x.drop_duplicates(subset=["pathway", "class"], keep="first")
    x = x.drop(columns=["absNES", "FDR_sort"], errors="ignore")
    return x

def plot_mass_dotplot_contrast_vs_ref(df: pd.DataFrame, save_path: Path,
                                      ref_class: str = "Control",
                                      top_n: int = 12,
                                      top_n_per_class: Optional[int] = 8,
                                      max_total: Optional[int] = 20,
                                      include_ref: bool = False,
                                      cmap=None,
                                      fdr_threshold: float = 0.25,
                                      mark_single_hit: bool = True,
                                      x_gap: float = 0.65,
                                      x_labelsize: int = 11,
                                      exclude_pathways: Optional[List[str]] = None):
    set_publication_style()
    if df.empty:
        warnings.warn("No Mass data to plot.")
        return

    if exclude_pathways is None:
        exclude_pathways = ["metabolic pathways"]

    palette = cmap if cmap is not None else MORANDI_CMAP

    x = _pick_best_direction_per_class(df).copy()
    x["NES"] = pd.to_numeric(x["NES"], errors="coerce")
    x["FDR"] = pd.to_numeric(x["FDR"], errors="coerce").clip(lower=1e-300)
    if "k_overlap" in x.columns:
        x["k_overlap"] = pd.to_numeric(x["k_overlap"], errors="coerce")
    else:
        x["k_overlap"] = np.nan
    x.dropna(subset=["NES", "FDR"], inplace=True)
    if x.empty:
        return

    if exclude_pathways:
        patt = "|".join([re.escape(p.lower()) for p in exclude_pathways])
        x = x[~x["pathway"].str.lower().str.contains(patt, na=False)]
        if x.empty:
            warnings.warn("All pathways filtered out after exclusion.")
            return

    pivot_nes = x.pivot_table(index="pathway", columns="class", values="NES", aggfunc="first")
    pivot_fdr = x.pivot_table(index="pathway", columns="class", values="FDR", aggfunc="first")
    pivot_k = x.pivot_table(index="class", columns="pathway", values="k_overlap", aggfunc="first").T

    if ref_class not in pivot_nes.columns:
        warnings.warn(f"Reference class '{ref_class}' not found")
        return

    classes = [c for c in ["Active", "Stable"] if c in pivot_nes.columns]
    if include_ref and ref_class not in classes:
        classes.append(ref_class)
    if not classes:
        return

    recs = []
    for p in pivot_nes.index:
        nes_ref = pivot_nes.loc[p, ref_class]
        for cls in classes:
            if cls == ref_class:
                delta = 0.0
                fdr = float(pivot_fdr.loc[p, ref_class]) if pd.notna(pivot_fdr.loc[p, ref_class]) else 1.0
                k = float(pivot_k.loc[p, ref_class]) if pd.notna(pivot_k.loc[p, ref_class]) else np.nan
            else:
                nes_cls = pivot_nes.loc[p, cls]
                if pd.isna(nes_cls) or pd.isna(nes_ref):
                    continue
                delta = float(nes_cls - nes_ref)
                fdr = float(pivot_fdr.loc[p, cls]) if pd.notna(pivot_fdr.loc[p, cls]) else 1.0
                k = float(pivot_k.loc[p, cls]) if pd.notna(pivot_k.loc[p, cls]) else np.nan
            recs.append({"pathway": p, "class": cls, "DeltaNES": delta, "FDR": fdr, "k_overlap": k})

    D = pd.DataFrame(recs)
    if D.empty:
        return

    D["-log10FDR"] = -np.log10(D["FDR"].clip(lower=1e-300))

    D_rank = D
    if fdr_threshold is not None:
        D_sig = D[D["FDR"] <= fdr_threshold].copy()
        if D_sig["pathway"].nunique() >= 6:
            D_rank = D_sig

    def _select_paths(df_rank):
        if top_n_per_class:
            paths = []
            for cls in classes:
                dfc = df_rank[df_rank["class"] == cls].copy()
                if dfc.empty:
                    continue
                dfc["score"] = dfc["DeltaNES"].abs()
                paths.extend(dfc.sort_values("score", ascending=False)["pathway"].head(top_n_per_class).tolist())
            uniq = list(dict.fromkeys(paths))
            if max_total:
                score_map = (df_rank.assign(score=df_rank["DeltaNES"].abs())
                             .groupby("pathway")["score"].max()
                             .sort_values(ascending=False))
                uniq = [p for p in score_map.index if p in uniq][:max_total]
            return uniq

        if "Active" in df_rank["class"].unique():
            return (df_rank[df_rank["class"] == "Active"]
                    .assign(absDelta=lambda d: d["DeltaNES"].abs())
                    .sort_values("absDelta", ascending=False)
                    .head(top_n)["pathway"].tolist())

        return (df_rank.assign(absDelta=lambda d: d["DeltaNES"].abs())
                .sort_values("absDelta", ascending=False)
                .head(top_n)["pathway"].tolist())

    top_paths = _select_paths(D_rank)
    if not top_paths and D_rank is not D:
        top_paths = _select_paths(D)

    if max_total and len(top_paths) < max_total:
        score_map = (D.assign(score=D["DeltaNES"].abs())
                     .groupby("pathway")["score"].max()
                     .sort_values(ascending=False))
        for p in score_map.index:
            if p not in top_paths:
                top_paths.append(p)
            if len(top_paths) >= max_total:
                break

    if not top_paths:
        return

    D = D[D["pathway"].isin(top_paths)].copy()
    if D.empty:
        return

    vmax = max(float(D["DeltaNES"].abs().max()), 1e-6)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    order_y = []
    if "Active" in D["class"].unique():
        order_y = (D[D["class"] == "Active"]
                   .assign(absDelta=lambda d: d["DeltaNES"].abs())
                   .sort_values("absDelta", ascending=False)["pathway"].unique().tolist())
    for p in D["pathway"].unique():
        if p not in order_y:
            order_y.append(p)

    y_map = {p: i for i, p in enumerate(order_y)}
    D["y_num"] = D["pathway"].map(y_map)

    x_gap_use = max(0.55, float(x_gap))
    x_pos_map = {cls: i * x_gap_use for i, cls in enumerate(classes)}
    D["x_num"] = D["class"].map(x_pos_map)

    max_size_metric = max(1e-6, float(D["-log10FDR"].max()))
    def size_scale(v):
        return 20.0 + 140.0 * (v / max_size_metric)

    if mark_single_hit:
        D["multi_hit"] = D["k_overlap"].fillna(0) >= 2
    else:
        D["multi_hit"] = False

    D_multi = D[D["multi_hit"]].copy()
    D_single = D[~D["multi_hit"]].copy()

    fig_w = max(5.0, 2.8 + (len(classes) - 1) * x_gap_use * 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))

    if not D_single.empty:
        ax.scatter(x=D_single["x_num"], y=D_single["y_num"],
                   c=D_single["DeltaNES"], cmap=palette, norm=norm,
                   s=D_single["-log10FDR"].map(size_scale),
                   edgecolors="black", linewidths=0.5, alpha=0.88)
    if not D_multi.empty:
        ax.scatter(x=D_multi["x_num"], y=D_multi["y_num"],
                   c=D_multi["DeltaNES"], cmap=palette, norm=norm,
                   s=(D_multi["-log10FDR"].map(size_scale) * 1.1),
                   edgecolors="black", linewidths=1.4, alpha=0.98)

    ax.set_yticks(range(len(order_y)))
    ax.set_yticklabels([_cleanup_pathway_name(p, 30) for p in order_y], fontsize=8)
    ax.set_ylim(-0.6, len(order_y) - 0.4)

    xticks = [x_pos_map[c] for c in classes]
    ax.set_xticks(xticks)
    ax.set_xticklabels(classes, fontsize=x_labelsize, fontweight='bold', rotation=0, ha='center')
    ax.set_xlim(min(xticks) - x_gap_use * 0.55, max(xticks) + x_gap_use * 0.55)

    # ---- ΔNES colorbar: move left to align with FDR legend box centerline ----
    cax = fig.add_axes([0.70, 0.25, 0.028, 0.45])  # x moved left (0.88 -> 0.845)
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=palette), cax=cax)
    cb.set_label("ΔNES", rotation=270, labelpad=10, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ---- 2-point FDR legend: small on top, big on bottom; top aligned ----
    fdr_vals = D["FDR"].astype(float).values
    fdr_vals = fdr_vals[np.isfinite(fdr_vals)]
    if fdr_vals.size > 0:
        fdr_min = float(np.min(fdr_vals))            # strongest -> biggest marker
        fdr_weak = float(np.quantile(fdr_vals, 0.75)) # weaker -> smaller marker

        if not (fdr_weak > fdr_min * 1.10):
            fdr_weak = min(0.95, max(fdr_min * 3.0, fdr_weak))

        v_strong = float(-np.log10(np.clip(fdr_min, 1e-300, 1.0)))
        v_weak = float(-np.log10(np.clip(fdr_weak, 1e-300, 1.0)))

        v_strong = min(max(v_strong, 0.0), max_size_metric)
        v_weak = min(max(v_weak, 0.0), max_size_metric)

        s_strong = float(size_scale(v_strong))  # big
        s_weak = float(size_scale(v_weak))      # small

        legend_scale = 0.75

        # order: small (weak) on top, big (strong) on bottom
        handles = [
            plt.scatter([], [], s=s_weak * legend_scale, color="#DDDDDD", edgecolors="black", linewidths=0.8),
            plt.scatter([], [], s=s_strong * legend_scale, color="#DDDDDD", edgecolors="black", linewidths=0.8),
        ]
        labels = [f"≤ {fdr_weak:.2g}", f"≤ {fdr_min:.2g}"]

        ax.legend(
            handles,
            labels,
            title="FDR",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),   # align top edge with main box top
            bbox_transform=ax.transAxes,
            borderaxespad=0.0,
            frameon=True,
            fontsize=7,
            title_fontsize=8,
            borderpad=0.6,
            labelspacing=0.9,
            handletextpad=0.8,
            scatterpoints=1,
        )

    ax.set_title("Mass KEGG (ΔNES)", fontsize=10, fontweight="bold", pad=6)
    plt.tight_layout(rect=[0, 0, 0.81, 0.97])  # right margin reduced for new colorbar/legend alignment
    save_publication_figure(
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
    print(f"[Mass] Saved with width={fig_w:.1f}")

def plot_rna_pathway_importance_heatmap(
    importance_df: pd.DataFrame,
    save_path: Path,
    top_n_pathways: int = 15,
    figsize: Tuple[float, float] = (6, 10)
):
    """
    Compact heatmap of RNA pathway importance.
    """
    set_publication_style()

    if importance_df.empty:
        warnings.warn("Empty importance DataFrame, skipping plot.")
        return

    pivot = importance_df.pivot(index='pathway', columns='class', values='importance')
    pivot = pivot.fillna(0)

    class_order = ["Active", "Stable", "Control"]
    pivot = pivot[[c for c in class_order if c in pivot.columns]]

    if "Active" in pivot.columns:
        top_pathways = pivot.nlargest(top_n_pathways, "Active").index.tolist()
    else:
        top_pathways = pivot.sum(axis=1).nlargest(top_n_pathways).index.tolist()

    pivot = pivot.loc[top_pathways]

    def clean_name(n):
        n = re.sub(r"^(HALLMARK_|REACTOME_|GOBP_)", "", n)
        return n.replace("_", " ").title()[:45]

    pivot.index = [clean_name(p) for p in pivot.index]

    if "Active" in pivot.columns:
        pivot = pivot.sort_values("Active", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        "importance_cmap",
        ["#F7F7F7", "#D4E6F1", "#85C1E9", "#2E86AB"]
    )

    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 9},
        cbar_kws={'label': 'Mean |SHAP|', 'shrink': 0.6},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("A) RNA Pathway Importance", fontsize=14, fontweight='bold', pad=10)

    plt.xticks(rotation=0, fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(
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

    print(f"[OK] RNA importance heatmap saved: {save_path}")
