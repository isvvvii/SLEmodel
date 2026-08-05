#!/usr/bin/env python3
"""Plot the annotated PBMC atlas and sample-level cell composition."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import scanpy as sc


LINEAGE_ORDER = [
    "CD8_Tnaive", "CD8_Tcm", "CD8_Tem", "CD8_Teff", "CD8_Tcycling",
    "CD4_Tnaive", "CD4_Treg", "CD4_T_ISG",
    "NK_Cytotoxic", "NK_GZMK", "NK_Prolif",
    "Mono_Classical", "Mono_NonClassical", "cDC2", "pDC",
    "B_Naive", "B_Memory", "B_ABC", "B_Plasmablast",
]

PALETTE = {
    "CD8_Tnaive": "#4C78A8", "CD8_Tcm": "#72A6C8", "CD8_Tem": "#E17C54",
    "CD8_Teff": "#D64F45", "CD8_Tcycling": "#9C6ADE",
    "CD4_Tnaive": "#7A6EAF", "CD4_Treg": "#8C77B8", "CD4_T_ISG": "#6A5A9E",
    "NK_Cytotoxic": "#48A868", "NK_GZMK": "#8FCB72", "NK_Prolif": "#91A4D2",
    "Mono_Classical": "#7B6BAA", "Mono_NonClassical": "#9D91C4",
    "cDC2": "#E48DB1", "pDC": "#ECA8C7",
    "B_Naive": "#D95F59", "B_Memory": "#E38A83", "B_ABC": "#C84E67",
    "B_Plasmablast": "#F0A49B",
}

SUBSETS = [
    ("PBMC", LINEAGE_ORDER),
    ("CD8 T cells", [x for x in LINEAGE_ORDER if x.startswith("CD8_")]),
    ("CD4 T cells", [x for x in LINEAGE_ORDER if x.startswith("CD4_")]),
    ("NK cells", [x for x in LINEAGE_ORDER if x.startswith("NK_")]),
    ("Myeloid cells", ["Mono_Classical", "Mono_NonClassical", "cDC2", "pDC"]),
    ("B cells", [x for x in LINEAGE_ORDER if x.startswith("B_")]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True, help="Annotated adata_full_atlas_final.h5ad")
    parser.add_argument("--metadata", type=Path, required=True, help="Controlled sample metadata TSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate(adata, metadata: pd.DataFrame) -> None:
    required_obs = {"sample_id", "lvl2_label"}
    missing = required_obs.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"AnnData obs is missing columns: {sorted(missing)}")
    if "X_umap" not in adata.obsm:
        raise ValueError("AnnData object does not contain X_umap")
    required_meta = {"sample_id", "apo_group"}
    missing = required_meta.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")


def plot_umap_panels(adata, output_dir: Path) -> None:
    use = adata[adata.obs.get("stage3_include", pd.Series(True, index=adata.obs_names)).astype(bool)].copy()
    use.obs["lvl2_label"] = pd.Categorical(use.obs["lvl2_label"], categories=LINEAGE_ORDER)

    fig = plt.figure(figsize=(10.2, 5.3))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.05, 1, 1, 1], hspace=0.25, wspace=0.2)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
    ]

    for ax, (title, labels) in zip(axes, SUBSETS):
        subset = use[use.obs["lvl2_label"].isin(labels)].copy()
        subset.obs["lvl2_label"] = subset.obs["lvl2_label"].cat.remove_unused_categories()
        present = list(subset.obs["lvl2_label"].cat.categories)
        colors = [PALETTE[label] for label in present]
        sc.pl.umap(
            subset,
            color="lvl2_label",
            palette=colors,
            title=f"{title} (n={subset.n_obs:,})",
            frameon=False,
            legend_loc="on data",
            legend_fontsize=6,
            size=3,
            ax=ax,
            show=False,
        )
        ax.set_xlabel("UMAP1", fontsize=7)
        ax.set_ylabel("UMAP2", fontsize=7)

    fig.savefig(output_dir / "pbmc_umap_atlas.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pbmc_umap_atlas.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_composition(adata, metadata: pd.DataFrame, output_dir: Path) -> None:
    use = adata[adata.obs.get("stage3_include", pd.Series(True, index=adata.obs_names)).astype(bool)]
    table = pd.crosstab(use.obs["sample_id"].astype(str), use.obs["lvl2_label"].astype(str))
    table = table.reindex(columns=LINEAGE_ORDER, fill_value=0)
    proportions = table.div(table.sum(axis=1), axis=0)

    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata = metadata.set_index("sample_id").reindex(proportions.index)
    order_fields = [c for c in ["trimester_or_stage", "activity_group", "apo_group"] if c in metadata]
    sample_order = metadata.sort_values(order_fields, kind="stable").index if order_fields else metadata.index
    proportions = proportions.loc[sample_order]

    fig, (ax, ax_meta) = plt.subplots(
        2, 1, figsize=(8.0, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [8, 1.1], "hspace": 0.03},
    )
    bottom = pd.Series(0.0, index=proportions.index)
    x = range(len(proportions))
    for label in LINEAGE_ORDER:
        values = proportions[label]
        ax.bar(x, values, bottom=bottom, width=0.9, color=PALETTE[label], label=label, linewidth=0)
        bottom += values
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=6, ncol=2)

    strip_fields = [c for c in ["trimester_or_stage", "activity_group", "apo_group"] if c in metadata]
    strip_palette = {
        "early": "#53A8D3", "mid": "#78D98B", "late": "#E87A43",
        "Active": "#33B7C5", "Stable": "#D56AC2",
        "APO": "#D95F5F", "non-APO": "#5B8DB8",
    }
    for row, field in enumerate(strip_fields):
        values = metadata.loc[sample_order, field].fillna("unknown").astype(str)
        categories = list(dict.fromkeys(values))
        code = values.map({value: i for i, value in enumerate(categories)}).to_numpy()[None, :]
        colors = [strip_palette.get(value, "#BDBDBD") for value in categories]
        ax_meta.imshow(code, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors),
                       extent=(-0.5, len(sample_order) - 0.5, row + 0.05, row + 0.95))
    ax_meta.set_yticks([i + 0.5 for i in range(len(strip_fields))], strip_fields, fontsize=7)
    ax_meta.set_xticks(list(x), sample_order, rotation=45, ha="right", fontsize=7)
    ax_meta.set_ylim(len(strip_fields), 0)
    for spine in ax_meta.spines.values():
        spine.set_visible(False)

    proportions.to_csv(output_dir / "celltype_proportions.tsv", sep="\t")
    fig.savefig(output_dir / "celltype_composition.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "celltype_composition.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.atlas)
    metadata = pd.read_csv(args.metadata, sep=None, engine="python")
    validate(adata, metadata)
    plot_umap_panels(adata, args.output_dir)
    plot_composition(adata, metadata, args.output_dir)


if __name__ == "__main__":
    main()
