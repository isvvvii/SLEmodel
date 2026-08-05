#!/usr/bin/env python
from __future__ import annotations

import re
import textwrap
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


EXP = Path(os.environ.get("SLEMODEL_SCRNA_OUTPUT_DIR", Path.cwd() / "outputs" / "single_cell")).expanduser().resolve()
SOURCE = Path(os.environ.get("SLEMODEL_DEG_PATHWAY_SOURCE_DIR", EXP / "source_tables")).expanduser().resolve()
OUT = EXP / "figures" / "deg_pathway_summary"
OUT.mkdir(parents=True, exist_ok=True)

SCRIPT_PY = Path(__file__).resolve()

MYELOID_DEG_SOURCE = SOURCE / "myeloid_platelet_deg_source.tsv"
CYTOTOXIC_DEG_SOURCE = SOURCE / "cd8_nk_deg_source.tsv"
MYELOID_PATHWAY_SOURCE = SOURCE / "myeloid_platelet_pathway_source.tsv"
CYTOTOXIC_PATHWAY_SOURCE = SOURCE / "cytotoxic_migration_pathway_source.tsv"

STEM = "deg_pathway_summary"
DPI = 600
WIDTH_MM = 180
HEIGHT_MM = 210
FIGSIZE = (WIDTH_MM / 25.4, HEIGHT_MM / 25.4)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def nfloat(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def short_scope(s: str) -> str:
    repl = {
        "Myeloid cells": "Myeloid cells",
        "CD14 classical monocytes": "CD14 mono.",
        "Inflammatory monocytes": "Inflam. mono.",
        "TREM1-high CD14 monocytes": "TREM1-hi mono.",
        "Platelet/MK-like cells": "Platelet/MK",
        "CD8 T cells": "CD8 T cells",
        "Naive CD8 T cells": "Naive CD8",
        "Cytotoxic CD8 T cells": "Cyto CD8",
        "NK/TNK cells": "NK/TNK cells",
        "Cytotoxic NK cells": "Cyto NK",
        "GZMK+ NK cells": "GZMK+ NK",
    }
    return repl.get(str(s), str(s))


def clean_pathway(s: str) -> str:
    s = str(s)
    s = re.sub(r"_", " ", s)
    s = re.sub(r"\s*\(GO:\d+\)", "", s)
    s = re.sub(r"\s*R-HSA-\d+", "", s)
    s = re.sub(r"^(GOBP|GO BP|REACTOME|KEGG)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    aliases = {
        "Platelet Activation, Signaling And Aggregation": "Platelet activation/signaling",
        "Platelet Activation, Signaling and Aggregation": "Platelet activation/signaling",
        "Platelet Degranulation": "Platelet degranulation",
        "Neutrophil Degranulation": "Neutrophil degranulation",
        "Regulation Of Platelet Aggregation": "Regulation of platelet aggregation",
        "Positive Regulation Of Platelet Aggregation": "Positive regulation of platelet aggregation",
        "Platelet Aggregation (Plug Formation)": "Platelet aggregation",
        "Response To Elevated Platelet Cytosolic Ca2+": "Elevated platelet cytosolic Ca2+",
        "Complement and coagulation cascades": "Complement/coagulation cascades",
        "Cytokine Signaling In Immune System": "Cytokine signaling in immune system",
        "Natural killer cell mediated cytotoxicity": "NK cell-mediated cytotoxicity",
        "Positive Regulation Of Chemokine Production": "Positive regulation of chemokine production",
        "Positive Regulation of Chemokine Production": "Positive regulation of chemokine production",
        "Regulation of Lymphocyte Activation": "Regulation of lymphocyte activation",
        "Regulation Of Lymphocyte Activation": "Regulation of lymphocyte activation",
        "Negative Regulation Of Chemokine Production": "Negative regulation of chemokine production",
        "Negative Regulation of Chemokine Production": "Negative regulation of chemokine production",
        "Regulation of Chemokine Production": "Regulation of chemokine production",
        "Positive Regulation of Natural Killer Cell Mediated Cytotoxicity": "Positive regulation of NK cytotoxicity",
        "Regulation of Natural Killer Cell Mediated Cytotoxicity": "Regulation of NK cytotoxicity",
    }
    return aliases.get(s, s)


def wrap_label(s: str, width: int = 27) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


DEG_A_GENES = [
    "S100A8",
    "S100A9",
    "S100A12",
    "LYZ",
    "FCN1",
    "LST1",
    "TREM1",
    "IL1B",
    "CXCL8",
    "PF4",
    "PPBP",
    "ITGA2B",
    "ITGB3",
    "TUBB1",
]
DEG_A_SCOPES = [
    "full|Myeloid",
    "myeloid|CD14_classical_monocyte",
    "myeloid|inflammatory_monocyte",
    "myeloid|TREM1high_like_CD14_monocyte",
    "full|Platelet_MK_like",
]
DEG_B_GENES = [
    "TNF",
    "TBX21",
    "PRF1",
    "NKG7",
    "GZMB",
    "GZMH",
    "GZMK",
    "CCL4",
    "CCL5",
    "CX3CR1",
    "KLRD1",
    "KLRK1",
]
DEG_B_SCOPES = ["full|CD8_T", "CD8|naive_CD8", "CD8|cytotoxic_CD8", "full|NK_TNK", "NK|NK_Cytotoxic", "NK|NK_GZMK"]

MYELOID_PATHWAY_TARGETS = [
    ("Reactome", "Hemostasis R-HSA-109582", "myeloid|CD14_classical_monocyte", "Hemostasis"),
    ("Reactome", "Platelet Activation, Signaling And Aggregation R-HSA-76002", "full|Myeloid", "Platelet activation/signaling"),
    ("Reactome", "Neutrophil Degranulation R-HSA-6798695", "myeloid|inflammatory_monocyte", "Neutrophil degranulation"),
    ("Reactome", "Platelet Degranulation R-HSA-114608", "full|Myeloid", "Platelet degranulation"),
    ("KEGG", "Leukocyte transendothelial migration", "full|Myeloid", "Leukocyte transendothelial migration"),
    ("KEGG", "Complement and coagulation cascades", "full|Myeloid", "Complement/coagulation cascades"),
    ("GO_BP", "Regulation Of Platelet Aggregation (GO:0090330)", "full|Myeloid", "Regulation of platelet aggregation"),
]

CYTOTOXIC_PATHWAY_TARGETS = [
    ("Reactome", "Cytokine Signaling In Immune System R-HSA-1280215", "full|CD8_T", "Cytokine signaling in immune system"),
    ("KEGG", "Natural killer cell mediated cytotoxicity", "full|NK_TNK", "NK cell-mediated cytotoxicity"),
    ("GO_BP", "Positive Regulation Of Chemokine Production (GO:0032722)", "CD8|naive_CD8", "Positive regulation of chemokine production"),
    ("GO_BP", "Regulation of Lymphocyte Activation (GO:0051249)", "full|NK_TNK", "Regulation of lymphocyte activation"),
]


def prepare_deg(path: Path, panel: str, genes: list[str], scopes: list[str], expected_sign: int):
    raw = read_tsv(path)
    raw["log2FC"] = nfloat(raw["log2FC"])
    raw["p_value"] = nfloat(raw["p_value"])
    raw["FDR"] = nfloat(raw["FDR"])
    rows = []
    excluded_rows = []
    for gene in genes:
        for scope in scopes:
            hit = raw[(raw["gene"] == gene) & (raw["cell_scope"] == scope)]
            if hit.empty:
                rows.append(
                    {
                        "panel": panel,
                        "gene": gene,
                        "cell_scope": scope,
                        "display_cell_scope": scope,
                        "effect_log2FC": np.nan,
                        "pvalue": np.nan,
                        "FDR_q": np.nan,
                        "direction": "missing",
                        "value_plot": np.nan,
                        "reason_selected": "missing from formal DEG heatmap source",
                        "source_file": str(path),
                    }
                )
                excluded_rows.append(
                    {
                        "item_type": "gene",
                        "item_name": f"{gene} | {scope}",
                        "panel": panel,
                        "reason_excluded": "missing DEG value in formal source table; tile left blank",
                        "source_file": str(path),
                    }
                )
                continue
            r = hit.iloc[0].copy()
            value = float(r["log2FC"]) if pd.notna(r["log2FC"]) else np.nan
            direction_ok = value > 0 if expected_sign > 0 else value < 0
            direction = "APO-up" if value > 0 else "APO-down" if value < 0 else "near-zero"
            if direction_ok:
                if pd.notna(r["FDR"]) and r["FDR"] < 0.05:
                    reason = "FDR<0.05 and direction-consistent in formal DEG heatmap source"
                elif pd.notna(r["p_value"]) and r["p_value"] < 0.05:
                    reason = "nominal p<0.05 and direction-consistent in formal DEG heatmap source"
                else:
                    reason = "direction-consistent curated gene from formal DEG heatmap source; retained as contextual tile"
            else:
                reason = "wrong direction or missing value for this comparison; tile left blank"
                excluded_rows.append(
                    {
                        "item_type": "gene",
                        "item_name": f"{gene} | {scope}",
                        "panel": panel,
                        "reason_excluded": "wrong direction for this comparison; tile left blank",
                        "source_file": str(path),
                    }
                )
            rows.append(
                {
                    "panel": panel,
                    "gene": gene,
                    "cell_scope": scope,
                    "display_cell_scope": short_scope(r.get("display_cell_scope", scope)),
                    "effect_log2FC": value,
                    "pvalue": r["p_value"],
                    "FDR_q": r["FDR"],
                    "direction": direction,
                    "value_plot": value if direction_ok else np.nan,
                    "reason_selected": reason,
                    "source_file": str(path),
                }
            )
    plot_df = pd.DataFrame(rows)
    selected = plot_df[plot_df["value_plot"].notna()].copy()
    selected = selected[
        [
            "panel",
            "gene",
            "cell_scope",
            "display_cell_scope",
            "effect_log2FC",
            "pvalue",
            "FDR_q",
            "direction",
            "reason_selected",
            "source_file",
        ]
    ]
    excluded = pd.DataFrame(excluded_rows)
    return plot_df, selected, excluded


def with_row_fdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_value"] = nfloat(df["p_value"])
    df["FDR"] = nfloat(df["FDR"])
    df["best_fdr"] = nfloat(df["best_fdr"]) if "best_fdr" in df.columns else np.nan
    df["row_FDR"] = pd.concat([df["FDR"], df["best_fdr"]], axis=1).min(axis=1, skipna=True)
    df["gene_count"] = nfloat(df["overlap_count"])
    df["gene_ratio"] = df["gene_count"] / nfloat(df["query_gene_count"])
    df["display_cell_scope"] = df["display_cell_scope"].map(short_scope)
    return df


def pick_pathways(path: Path, panel: str, targets: list[tuple[str, str, str, str]], direction: str):
    raw = with_row_fdr(read_tsv(path))
    selected_rows = []
    selected_keys = set()
    for database, term_name, scope, clean_name in targets:
        hit = raw[(raw["database"] == database) & (raw["term_name"] == term_name) & (raw["cell_scope"] == scope) & (raw["direction"] == direction)]
        if hit.empty:
            raise RuntimeError(f"Missing pathway target: {database} | {term_name} | {scope}")
        row = hit.iloc[0].copy()
        row["panel"] = panel
        row["pathway_clean"] = clean_name
        row["plot_direction"] = "APO-up" if direction == "up" else "APO-down"
        row["neg_log10_fdr_plot"] = -np.log10(max(float(row["row_FDR"]), 1e-300))
        row["reason_selected"] = (
            "FDR/best-FDR<0.05 and direction-consistent"
            if row["row_FDR"] < 0.05
            else "FDR/best-FDR<=0.1 and direction-consistent"
        )
        row["source_file"] = str(path)
        selected_rows.append(row)
        selected_keys.add((database, term_name, scope))
    selected = pd.DataFrame(selected_rows)

    raw["pathway_clean"] = raw["term_name"].map(clean_pathway)
    raw["_key"] = list(zip(raw["database"], raw["term_name"], raw["cell_scope"]))
    relevant_patterns = (
        "platelet|hemostasis|neutrophil|leukocyte|complement|coagulation"
        if direction == "up"
        else "cytokine|cytotoxic|chemokine|lymphocyte|natural killer|nk"
    )
    cand = raw[
        (raw["direction"] == direction)
        & raw["term_name"].str.contains(relevant_patterns, case=False, regex=True, na=False)
        & ~raw["_key"].isin(selected_keys)
    ].copy()

    selected_clean = set(selected["pathway_clean"])
    reasons = []
    for _, r in cand.iterrows():
        clean = clean_pathway(r["term_name"])
        reason = "not selected after relevance/FDR ranking"
        if clean in selected_clean:
            reason = "duplicate pathway term or duplicate scope; not plotted"
        if direction == "up" and clean in {
            "Platelet activation",
            "Platelet aggregation",
            "Positive regulation of platelet aggregation",
            "Elevated platelet cytosolic Ca2+",
        }:
            reason = "redundant platelet-related term removed from the summary"
        if direction == "down" and (
            clean in {"Regulation of chemokine production", "Positive regulation of NK cytotoxicity", "Regulation of NK cytotoxicity"}
            or pd.notna(r["row_FDR"])
            and r["row_FDR"] > 0.1
        ):
            reason = "weak FDR>0.1 or redundant APO-down pathway term removed from the summary"
        if direction == "down" and clean == "Negative regulation of chemokine production":
            reason = "not plotted because CD8/NK rows were weak despite a borderline best-FDR from non-CD8/NK context"
        reasons.append(reason)
    excluded = pd.DataFrame(
        {
            "item_type": "pathway",
            "item_name": cand["pathway_clean"],
            "panel": panel,
            "reason_excluded": reasons,
            "source_file": str(path),
        }
    )
    return selected, excluded


def plot_heatmap(ax, df: pd.DataFrame, genes: list[str], scopes: list[str], title: str, cmap, norm):
    matrix = df.pivot(index="gene", columns="cell_scope", values="value_plot").reindex(index=genes, columns=scopes)
    values = matrix.to_numpy(dtype=float)
    for y in range(len(genes)):
        for x in range(len(scopes)):
            value = values[y, x]
            fill = "#eeeeee" if np.isnan(value) else cmap(norm(value))
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=fill, edgecolor="white", linewidth=0.45))
    ax.set_xlim(-0.5, len(scopes) - 0.5)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(len(scopes)))
    xlabels = [short_scope(df.loc[df["cell_scope"] == s, "display_cell_scope"].dropna().iloc[0]) for s in scopes]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)
    ax.tick_params(length=0, pad=1.5)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_vector_colorbar(cax, cmap, norm):
    cax.set_xlim(0, 1)
    cax.set_ylim(norm.vmin, norm.vmax)
    edges = np.linspace(norm.vmin, norm.vmax, 96)
    for y0, y1 in zip(edges[:-1], edges[1:]):
        mid = (y0 + y1) / 2
        cax.add_patch(Rectangle((0, y0), 1, y1 - y0, facecolor=cmap(norm(mid)), edgecolor="none"))
    cax.set_xticks([])
    cax.set_yticks([norm.vmin, 0, norm.vmax])
    cax.set_ylabel("APO - non-APO log2FC", labelpad=3)
    cax.tick_params(axis="y", labelsize=6, length=2, width=0.5)
    for spine in cax.spines.values():
        spine.set_linewidth(0.4)


def plot_dotplot(ax, df: pd.DataFrame, title: str, color: str):
    ordered = df.sort_values("neg_log10_fdr_plot", ascending=True).copy()
    y = np.arange(len(ordered))
    sizes = 16 + ordered["gene_count"].to_numpy(float) * 5.2
    ax.scatter(ordered["neg_log10_fdr_plot"], y, s=sizes, c=color, edgecolor="#303030", linewidth=0.35, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(v, 27) for v in ordered["pathway_clean"]])
    ax.set_xlabel("-log10(FDR)", labelpad=3)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, width=0.5)
    xmax = max(ordered["neg_log10_fdr_plot"].max() * 1.10, 2.0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)


def write_tables(gene_selected, pathway_selected, excluded):
    gene_selected.to_csv(OUT / "selected_genes.tsv", sep="\t", index=False)
    pathway_cols = [
        "panel",
        "database",
        "term_name",
        "pathway_clean",
        "cell_scope",
        "display_cell_scope",
        "plot_direction",
        "p_value",
        "FDR",
        "best_fdr",
        "row_FDR",
        "gene_count",
        "gene_ratio",
        "reason_selected",
        "source_file",
    ]
    pathway_selected[pathway_cols].rename(
        columns={
            "term_name": "pathway_original",
            "plot_direction": "direction",
            "p_value": "pvalue",
            "FDR": "FDR_q",
        }
    ).to_csv(OUT / "selected_pathways.tsv", sep="\t", index=False)
    excluded.to_csv(OUT / "excluded_candidates.tsv", sep="\t", index=False)

    gene_source = pd.DataFrame(
        {
            "panel": gene_selected["panel"],
            "item_name": gene_selected["gene"],
            "cell_scope": gene_selected["cell_scope"],
            "value": gene_selected["effect_log2FC"],
            "pvalue": gene_selected["pvalue"],
            "FDR_q": gene_selected["FDR_q"],
            "direction": gene_selected["direction"],
            "reason_selected": gene_selected["reason_selected"],
            "source_file": gene_selected["source_file"],
        }
    )
    gene_source.insert(1, "item_type", "gene")
    pathway_source = pd.DataFrame(
        {
            "panel": pathway_selected["panel"],
            "item_name": pathway_selected["pathway_clean"],
            "cell_scope": pathway_selected["cell_scope"],
            "value": pathway_selected["neg_log10_fdr_plot"],
            "pvalue": pathway_selected["p_value"],
            "FDR_q": pathway_selected["FDR"],
            "direction": pathway_selected["plot_direction"],
            "reason_selected": pathway_selected["reason_selected"],
            "source_file": pathway_selected["source_file"],
        }
    )
    pathway_source.insert(1, "item_type", "pathway")
    pd.concat([gene_source, pathway_source], ignore_index=True).to_csv(
        OUT / "deg_pathway_summary_source_used.tsv", sep="\t", index=False
    )


def write_readme(gene_selected, pathway_selected, excluded):
    myeloid_genes = ", ".join(DEG_A_GENES)
    cytotoxic_genes = ", ".join(DEG_B_GENES)
    myeloid_paths = "; ".join(
        pathway_selected[pathway_selected["panel"].eq("myeloid_platelet_pathway")]["pathway_clean"].tolist()
    )
    cytotoxic_paths = "; ".join(
        pathway_selected[pathway_selected["panel"].eq("cytotoxic_migration_pathway")]["pathway_clean"].tolist()
    )
    n_excluded = len(excluded)
    readme = f"""DEG and pathway summary
=======================

Output stem: {STEM}
Output directory: {OUT}

Purpose
-------
This plot summarizes APO versus non-APO single-cell results in a 2x2 layout:
- APO-up myeloid/platelet DEGs
- APO-down CD8/NK DEGs
- APO-up myeloid/platelet pathways
- APO-down cytotoxic/migration pathways

Source tables
-------------
Heatmap genes are selected from formal DEG source tables:
- {MYELOID_DEG_SOURCE}
- {CYTOTOXIC_DEG_SOURCE}

Pathways are selected from formal pathway source tables:
- {MYELOID_PATHWAY_SOURCE}
- {CYTOTOXIC_PATHWAY_SOURCE}

No h5ad was read. No DEG/enrichment rerun was performed.

Statistical display notes
-------------------------
Heatmap values are existing edgeR APO - non-APO log2FC values from the formal DEG source tables. Gene-level FDR is sparse, so the heatmap uses nominal p<0.05 and direction-consistent curated genes for visualization; direction-inconsistent or missing tiles are rendered as very light grey and listed in the excluded table.

Pathway plots prioritize FDR/best_FDR < 0.05. The cytotoxicity and migration summary allows FDR/best_FDR <= 0.1 terms when they are direction-consistent and belong to the prespecified categories. The dotplot x-axis is -log10(FDR). Where best_fdr is available in the source table, row_FDR is the minimum of FDR and best_fdr and is recorded in the selected pathway table.

Selected content
----------------
Myeloid/platelet genes ({len(DEG_A_GENES)}): {myeloid_genes}
CD8/NK genes ({len(DEG_B_GENES)}): {cytotoxic_genes}
Myeloid/platelet pathways ({sum(pathway_selected['panel'].eq('myeloid_platelet_pathway'))}): {myeloid_paths}
Cytotoxic/migration pathways ({sum(pathway_selected['panel'].eq('cytotoxic_migration_pathway'))}): {cytotoxic_paths}

Excluded candidates
-------------------
Excluded candidates include duplicate pathway terms, FDR>0.1 APO-down terms, wrong-direction genes, missing tiles, or weakly relevant terms. Total excluded/context rows written: {n_excluded}.

Generated files
---------------
- {STEM}.png
- {STEM}.pdf
- {STEM}.svg
- {STEM}_source_used.tsv
- selected_genes.tsv
- selected_pathways.tsv
- excluded_candidates.tsv
- README_deg_pathway_summary.txt
"""
    (OUT / "README_deg_pathway_summary.txt").write_text(readme)


def main():
    a_plot, a_sel, a_exc = prepare_deg(MYELOID_DEG_SOURCE, "myeloid_platelet_deg", DEG_A_GENES, DEG_A_SCOPES, 1)
    b_plot, b_sel, b_exc = prepare_deg(CYTOTOXIC_DEG_SOURCE, "cd8_nk_deg", DEG_B_GENES, DEG_B_SCOPES, -1)
    c_sel, c_exc = pick_pathways(MYELOID_PATHWAY_SOURCE, "myeloid_platelet_pathway", MYELOID_PATHWAY_TARGETS, "up")
    d_sel, d_exc = pick_pathways(CYTOTOXIC_PATHWAY_SOURCE, "cytotoxic_migration_pathway", CYTOTOXIC_PATHWAY_TARGETS, "down")

    gene_selected = pd.concat([a_sel, b_sel], ignore_index=True)
    pathway_selected = pd.concat([c_sel, d_sel], ignore_index=True)
    excluded = pd.concat([a_exc, b_exc, c_exc, d_exc], ignore_index=True).drop_duplicates()

    write_tables(gene_selected, pathway_selected, excluded)
    write_readme(gene_selected, pathway_selected, excluded)

    cmap = colors.LinearSegmentedColormap.from_list("apo_logfc", ["#2166ac", "#f7f7f7", "#b2182b"])
    cmap.set_bad("#eeeeee")
    norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"height_ratios": [0.98, 1.20], "hspace": 0.48, "wspace": 0.96},
    )
    fig.subplots_adjust(left=0.185, right=0.81, top=0.955, bottom=0.075)

    plot_heatmap(axes[0, 0], a_plot, DEG_A_GENES, DEG_A_SCOPES, "APO-up myeloid/platelet DEGs", cmap, norm)
    plot_heatmap(axes[0, 1], b_plot, DEG_B_GENES, DEG_B_SCOPES, "APO-down CD8/NK DEGs", cmap, norm)
    plot_dotplot(axes[1, 0], c_sel, "APO-up myeloid/platelet pathways", "#c23b31")
    plot_dotplot(axes[1, 1], d_sel, "APO-down cytotoxic/migration\npathways", "#2f70b7")

    cax = fig.add_axes([0.855, 0.58, 0.022, 0.26])
    draw_vector_colorbar(cax, cmap, norm)

    legend_counts = [5, 15, 30]
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#707070", markeredgecolor="#303030", markersize=np.sqrt(16 + count * 5.2))
        for count in legend_counts
    ]
    fig.legend(
        handles,
        [str(x) for x in legend_counts],
        title="Gene count",
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.14),
        borderaxespad=0,
        labelspacing=0.9,
        handletextpad=0.9,
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Wrote DEG and pathway summary outputs to {OUT}")


if __name__ == "__main__":
    main()
