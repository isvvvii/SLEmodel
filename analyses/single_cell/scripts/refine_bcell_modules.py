#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(os.environ.get("SLEMODEL_SCRNA_ROOT", Path.cwd())).expanduser().resolve()
BASE = Path(os.environ.get("SLEMODEL_SCRNA_OUTPUT_DIR", ROOT / "outputs" / "single_cell")).expanduser().resolve()
SCRIPT36 = Path(__file__).with_name("run_apo_analysis.py")
OUT = BASE / "bcell_module_refinement"


TARGET_MODULES = [
    {
        "module_name": "BAFF_APRIL_receptor_response_B",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["TNFRSF13B", "TNFRSF13C", "TNFRSF17"],
        "interpretation": "B/plasmablast response-side receptor program for BAFF/APRIL survival and maintenance.",
        "notes": "",
    },
    {
        "module_name": "B_survival_maintenance_receptor_program",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["TNFRSF13B", "TNFRSF13C", "TNFRSF17", "CD40", "CXCR4", "CXCR5"],
        "interpretation": "B/plasmablast survival, maintenance and trafficking receptor program.",
        "notes": "",
    },
    {
        "module_name": "B_cell_activation_BCR_CD40",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["CD79A", "CD79B", "MS4A1", "CD40", "CD74", "TLR7"],
        "interpretation": "BCR/CD40/TLR7-related B-cell activation and antigen-linked activation program.",
        "notes": "",
    },
    {
        "module_name": "B_antigen_presentation_MHCII",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1", "CD74"],
        "interpretation": "B-cell antigen presentation / MHC-II program.",
        "notes": "",
    },
    {
        "module_name": "ABC_like_autoreactive_B_program",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["TBX21", "ITGAX", "FCRL5", "FCRL3", "TLR7", "ZEB2"],
        "interpretation": "ABC-like / atypical autoreactive B-cell program relevant to SLE.",
        "notes": "",
    },
    {
        "module_name": "Naive_B_program",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["TCL1A", "IGHD", "FCER2", "IL4R", "CCR7"],
        "interpretation": "Naive B-cell state program.",
        "notes": "",
    },
    {
        "module_name": "Memory_B_program",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["CD27", "TNFRSF13B", "BANK1", "CD83", "AIM2"],
        "interpretation": "Memory B-cell program.",
        "notes": "",
    },
    {
        "module_name": "Plasmablast_ASC_core_nonIG",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["PRDM1", "IRF4", "XBP1", "MZB1", "JCHAIN", "SDC1"],
        "interpretation": "Plasmablast / antibody-secreting-cell differentiation core program without immunoglobulin heavy/light chain genes.",
        "notes": "non-IG ASC core",
    },
    {
        "module_name": "Antibody_secretion_ASC_IG_allowed",
        "module_class": "B_intrinsic",
        "intended_cell_scope": "B-lineage scopes only",
        "genes": ["XBP1", "MZB1", "JCHAIN", "SDC1", "IGKC", "IGHG1", "IGHG2", "IGHG3", "IGHA1", "IGLC2"],
        "interpretation": "Antibody-secreting-cell program including immunoglobulin genes.",
        "notes": "IG-gene-sensitive; interpret cautiously",
    },
    {
        "module_name": "BAFF_APRIL_ligand_source",
        "module_class": "source_side",
        "intended_cell_scope": "non-B source-side scopes only",
        "genes": ["TNFSF13B", "TNFSF13"],
        "interpretation": "Myeloid/APC-derived BAFF/APRIL ligand source signal for B/plasmablast survival.",
        "notes": "source-side or microenvironmental signal, not direct B-cell-intrinsic evidence",
    },
    {
        "module_name": "CD40LG_Tcell_help_source",
        "module_class": "source_side",
        "intended_cell_scope": "non-B source-side scopes only",
        "genes": ["CD40LG", "ICOS", "CD28", "IL21"],
        "interpretation": "T-cell help / CD40LG-related source signal.",
        "notes": "IL21 may be sparse in PBMC; report gene coverage; source-side only",
    },
    {
        "module_name": "CXCL12_CXCL13_B_homing_source",
        "module_class": "source_side",
        "intended_cell_scope": "non-B source-side scopes only",
        "genes": ["CXCL12", "CXCL13"],
        "interpretation": "B-cell homing / follicular-like chemokine source signal.",
        "notes": "May be sparse in PBMC; report gene coverage; source-side only",
    },
    {
        "module_name": "MIF_CD74_CXCR4_CD44_axis",
        "module_class": "source_side",
        "intended_cell_scope": "non-B source-side scopes only",
        "genes": ["MIF", "CD74", "CXCR4", "CD44"],
        "interpretation": "B-cell survival / migration-related MIF-CD74-CXCR4-CD44 axis.",
        "notes": "mixed ligand-receptor context; compare with CellPhoneDB if relevant; source-side only",
    },
]

B_OBJECTS = {"B_cell"}
SOURCE_OBJECTS = {"full_atlas", "myeloid", "CD4", "CD8", "NK"}
B_SCOPES = [
    "B|B_total",
    "B|naive_B",
    "B|memory_B",
    "B|atypical_B_ABC_like",
    "B|IFN_high_B",
    "B|plasmablast_differentiation_high_B",
    "B|true_annotation_plasmablast_B",
]
SOURCE_SCOPES = [
    "full|Myeloid",
    "full|pDC",
    "full|Platelet_MK_like",
    "myeloid|myeloid_total",
    "myeloid|inflammatory_monocyte",
    "myeloid|CD14_classical_monocyte",
    "myeloid|FCGR3A_CD16_monocyte",
    "myeloid|cDC",
    "myeloid|pDC",
    "CD4|CD4_total",
    "CD4|CD4_naive",
    "CD4|CD4_Treg",
    "CD4|CD4_ISG",
    "CD8|CD8_total",
    "CD8|naive_CD8",
    "CD8|ISGhigh_naive_CD8_like",
    "CD8|GZMKpos_GZMBlow_effmem_like_CD8",
    "CD8|cytotoxic_CD8",
    "NK|NK_total",
    "NK|NK_Cytotoxic",
    "NK|NK_GZMK",
    "full|NK_TNK",
]


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("single_cell_analysis", SCRIPT36)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def gene_coverage_flag(n_found: int, n_requested: int) -> str:
    if n_requested <= 2:
        return "ok" if n_found >= 2 else "low_gene_coverage"
    if n_requested == 3:
        return "ok" if n_found >= 2 else "low_gene_coverage"
    return "ok" if n_found >= 3 else "low_gene_coverage"


def bh_fdr(p: pd.Series) -> np.ndarray:
    p = pd.to_numeric(p, errors="coerce")
    out = np.full(len(p), np.nan)
    valid = p.notna()
    if valid.any():
        out[valid.to_numpy()] = multipletests(p[valid], method="fdr_bh")[1]
    return out


def sample_grouping(analysis) -> pd.DataFrame:
    group_map = (
        analysis.build_metadata()
        .reset_index(drop=True)[["sample_id", "apo_group"]]
        .rename(columns={"apo_group": "group"})
    )
    group_map["group"] = group_map["group"].replace({"no_APO": "non-APO"})
    counts = group_map.groupby("group")["sample_id"].nunique().to_dict()
    if counts.get("APO") != 8 or counts.get("non-APO") != 9:
        raise RuntimeError(f"Expected 8 APO and 9 non-APO samples; found {counts}")
    return group_map.sort_values("sample_id")


def mann_stats(sample_scores: pd.DataFrame, group_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_samples = sorted(group_map["sample_id"].tolist())
    for key, sub in sample_scores.groupby(["object_name", "cell_scope", "module_name"], sort=False):
        object_name, cell_scope, module_name = key
        if "group" not in sub.columns:
            sub = sub.merge(group_map, on="sample_id", how="left")
        apo = sub.loc[sub["group"].eq("APO"), "module_score_mean"].dropna().astype(float)
        non = sub.loc[sub["group"].eq("non-APO"), "module_score_mean"].dropna().astype(float)
        p = np.nan
        if len(apo) > 0 and len(non) > 0:
            p = float(mannwhitneyu(apo, non, alternative="two-sided", method="exact").pvalue)
        apo_mean = float(np.mean(apo)) if len(apo) else np.nan
        non_mean = float(np.mean(non)) if len(non) else np.nan
        apo_med = float(np.median(apo)) if len(apo) else np.nan
        non_med = float(np.median(non)) if len(non) else np.nan
        mean_diff = apo_mean - non_mean if np.isfinite(apo_mean) and np.isfinite(non_mean) else np.nan
        med_diff = apo_med - non_med if np.isfinite(apo_med) and np.isfinite(non_med) else np.nan
        direction = "APO_high" if mean_diff > 0 else "APO_low" if mean_diff < 0 else "no_difference"
        first = sub.iloc[0]
        available = set(sub.loc[sub["module_score_mean"].notna(), "sample_id"])
        missing = [s for s in all_samples if s not in available]
        cell_counts = dict(zip(sub["sample_id"], sub["n_cells"]))
        rows.append(
            {
                "object_name": object_name,
                "cell_scope": cell_scope,
                "module_name": module_name,
                "module_class": first["module_class"],
                "APO_median": apo_med,
                "nonAPO_median": non_med,
                "APO_mean": apo_mean,
                "nonAPO_mean": non_mean,
                "mean_difference": mean_diff,
                "median_difference": med_diff,
                "direction": direction,
                "p_value": p,
                "n_APO": int(len(apo)),
                "n_nonAPO": int(len(non)),
                "missing_samples": ";".join(missing),
                "cell_count_per_sample": ";".join(f"{s}:{int(cell_counts.get(s, 0))}" for s in all_samples),
                "genes_found": first["genes_found"],
                "n_genes_found": int(first["n_genes_found"]),
                "n_genes_requested": int(first["n_genes_requested"]),
                "gene_coverage_flag": first["gene_coverage_flag"],
                "interpretability_flag": first["interpretability_flag"],
                "low_count_any_group": len(apo) < 3 or len(non) < 3,
            }
        )
    stats = pd.DataFrame(rows)
    stats["fdr_bh"] = bh_fdr(stats["p_value"])
    cols = [
        "object_name",
        "cell_scope",
        "module_name",
        "module_class",
        "APO_median",
        "nonAPO_median",
        "APO_mean",
        "nonAPO_mean",
        "mean_difference",
        "median_difference",
        "direction",
        "p_value",
        "fdr_bh",
        "n_APO",
        "n_nonAPO",
        "missing_samples",
        "cell_count_per_sample",
        "genes_found",
        "n_genes_found",
        "n_genes_requested",
        "gene_coverage_flag",
        "interpretability_flag",
        "low_count_any_group",
    ]
    return stats[cols].sort_values(["p_value", "cell_scope", "module_name"], na_position="last")


def module_definitions_df() -> pd.DataFrame:
    df = pd.DataFrame(TARGET_MODULES).copy()
    df["genes"] = df["genes"].map(lambda x: ";".join(x))
    return df


def score_modules(analysis, group_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, str]]:
    target_gene_union = []
    for rec in TARGET_MODULES:
        target_gene_union.extend(rec["genes"])
    mask_gene_union = list(analysis.TARGETED_GENES_FOR_MODULES)
    gene_union = list(dict.fromkeys(target_gene_union + mask_gene_union))

    score_rows = []
    state_rows = []
    matrix_keys = {}
    h5ad_read = []
    for object_name, path in analysis.H5ADS.items():
        if object_name not in B_OBJECTS | SOURCE_OBJECTS:
            continue
        target = analysis.H5Target(object_name, path, gene_union)
        h5ad_read.append(str(path))
        matrix_keys[object_name] = target.matrix_key
        try:
            masks, defs = analysis.masks_for_object(target)
            for scope, mask in masks.items():
                if object_name == "B_cell" and scope in B_SCOPES:
                    state_rows.append(
                        {
                            "object_name": object_name,
                            "cell_scope": scope,
                            "total_cells": int(mask.sum()),
                            "samples_with_cells": int(pd.Series(target.samples[mask]).nunique()) if int(mask.sum()) else 0,
                            "definition_method": next((d["definition_method"] for d in defs if d["cell_scope"] == scope), ""),
                        }
                    )
                if object_name == "B_cell":
                    modules = [m for m in TARGET_MODULES if m["module_class"] == "B_intrinsic" and scope in B_SCOPES]
                else:
                    modules = [m for m in TARGET_MODULES if m["module_class"] == "source_side" and scope in SOURCE_SCOPES]
                if not modules:
                    continue
                idx = np.where(mask)[0]
                for mod in modules:
                    found = [g for g in mod["genes"] if g in target.gene_to_idx]
                    n_found = len(found)
                    coverage = gene_coverage_flag(n_found, len(mod["genes"]))
                    cell_score = target.score(found)
                    if mod["module_class"] == "source_side":
                        interp = "source-side or microenvironmental signal, not direct B-cell-intrinsic evidence"
                    elif "IG_allowed" in mod["module_name"]:
                        interp = "B-lineage intrinsic but IG-gene-sensitive; interpret cautiously"
                    else:
                        interp = "B-lineage intrinsic module"
                    if coverage != "ok":
                        interp = interp + "; low gene coverage"
                    for sid in group_map["sample_id"].tolist():
                        rows = idx[target.samples[idx] == sid] if len(idx) else np.array([], dtype=int)
                        vals = cell_score[rows] if len(rows) else np.array([])
                        score_rows.append(
                            {
                                "sample_id": sid,
                                "group": group_map.set_index("sample_id").loc[sid, "group"],
                                "object_name": object_name,
                                "cell_scope": scope,
                                "module_name": mod["module_name"],
                                "module_class": mod["module_class"],
                                "module_score_mean": float(np.nanmean(vals)) if len(vals) and np.isfinite(vals).any() else np.nan,
                                "module_score_median": float(np.nanmedian(vals)) if len(vals) and np.isfinite(vals).any() else np.nan,
                                "n_cells": int(len(rows)),
                                "low_count": int(len(rows)) < 20,
                                "genes_requested": ";".join(mod["genes"]),
                                "genes_found": ";".join(found),
                                "n_genes_requested": len(mod["genes"]),
                                "n_genes_found": n_found,
                                "gene_coverage_flag": coverage,
                                "interpretability_flag": interp,
                                "h5ad_path": str(path),
                                "expression_layer": target.matrix_key,
                            }
                        )
        finally:
            target.close()
    return pd.DataFrame(score_rows), pd.DataFrame(state_rows), h5ad_read, sorted(set(matrix_keys.values())), matrix_keys


def annotate_candidates(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.copy()
    out["recommend_for_display"] = False
    out["recommend_reason"] = ""
    for i, r in out.iterrows():
        ok_basic = (
            pd.notna(r["p_value"])
            and r["p_value"] < 0.05
            and r["direction"] == "APO_low"
            and r["gene_coverage_flag"] == "ok"
            and not bool(r["low_count_any_group"])
        )
        ig_sensitive = "IG_allowed" in str(r["module_name"])
        if r["module_class"] == "B_intrinsic":
            rec = ok_basic and not ig_sensitive
            reason = "direct B-lineage APO_low nominal p<0.05 with acceptable coverage" if rec else "not recommended: not APO_low p<0.05 with acceptable direct B-lineage interpretability"
            if ig_sensitive:
                reason = "not recommended for the summary by default: IG-gene-sensitive and ambient/doublet-sensitive"
        else:
            rec = ok_basic and r["module_name"] in {"BAFF_APRIL_ligand_source", "CD40LG_Tcell_help_source", "MIF_CD74_CXCR4_CD44_axis"}
            reason = "source-side APO_low nominal p<0.05 with B communication relevance; not direct B-cell-intrinsic evidence" if rec else "not recommended: source-side result is not APO_low p<0.05 with acceptable coverage/interpretability"
        out.loc[i, "recommend_for_display"] = bool(rec)
        out.loc[i, "recommend_reason"] = reason
    return out


def display_label(scope: str, module: str) -> str:
    s = scope.replace("B|true_annotation_plasmablast_B", "Plasmablast B").replace("myeloid|FCGR3A_CD16_monocyte", "FCGR3A/CD16 mono.")
    s = s.replace("myeloid|cDC", "cDC").replace("myeloid|CD14_classical_monocyte", "CD14 mono.")
    s = s.replace("myeloid|myeloid_total", "Myeloid total").replace("full|Platelet_MK_like", "Platelet/MK-like")
    m = module.replace("_", " ").replace("BAFF APRIL", "BAFF/APRIL").replace("MHCII", "MHC-II")
    return f"{s}\n{m}"


def plot_combo(scores: pd.DataFrame, stats: pd.DataFrame, modules: list[tuple[str, str]], title: str, outstem: str):
    modules = [m for m in modules if m[0] and m[1]]
    if not modules:
        modules = [("", "")]
    n = len(modules)
    fig, axes = plt.subplots(1, n, figsize=(180 / 25.4, 72 / 25.4), sharey=False)
    if n == 1:
        axes = [axes]
    colors = {"non-APO": "#5B8DB8", "APO": "#D95F5F"}
    rng = np.random.default_rng(7)
    for ax, (scope, module) in zip(axes, modules):
        if not scope:
            ax.axis("off")
            continue
        sub = scores[(scores["cell_scope"] == scope) & (scores["module_name"] == module)].copy()
        st = stats[(stats["cell_scope"] == scope) & (stats["module_name"] == module)]
        for xpos, group in enumerate(["non-APO", "APO"]):
            vals = sub.loc[sub["group"] == group, "module_score_mean"].dropna().astype(float).to_numpy()
            if len(vals):
                ax.boxplot(vals, positions=[xpos], widths=0.42, patch_artist=True, showfliers=False, boxprops={"facecolor": "white", "edgecolor": colors[group], "linewidth": 1}, medianprops={"color": colors[group], "linewidth": 1.2}, whiskerprops={"color": colors[group]}, capprops={"color": colors[group]})
                jitter = rng.normal(0, 0.035, len(vals))
                ax.scatter(np.full(len(vals), xpos) + jitter, vals, s=18, color=colors[group], alpha=0.9, edgecolor="white", linewidth=0.3)
        ptxt = "p=NA"
        if not st.empty and pd.notna(st.iloc[0]["p_value"]):
            ptxt = f"p={st.iloc[0]['p_value']:.3g}"
        ax.set_title(display_label(scope, module), fontsize=7.2, pad=5)
        ax.text(0.5, 0.98, ptxt, transform=ax.transAxes, ha="center", va="top", fontsize=6.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["non-APO", "APO"], rotation=25, ha="right", fontsize=6.3)
        ax.tick_params(axis="y", labelsize=6.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    axes[0].set_ylabel("Mean module score", fontsize=7)
    fig.suptitle(title, fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{outstem}.{ext}", dpi=600)
    plt.close(fig)


def select_combinations(intrinsic: pd.DataFrame, source: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    direct_rec = intrinsic[intrinsic["recommend_for_display"]].sort_values("p_value")
    direct_pool = intrinsic[(intrinsic["direction"] == "APO_low") & (intrinsic["gene_coverage_flag"] == "ok")].sort_values("p_value")
    source_rec = source[source["recommend_for_display"]].sort_values("p_value")

    def modstr(row):
        if row is None:
            return "insufficient_candidate"
        return f"{row.cell_scope} | {row.module_name}"

    best_direct_rows = [r for r in direct_rec.itertuples()][:3]
    while len(best_direct_rows) < 3:
        extras = [r for r in direct_pool.itertuples() if f"{r.cell_scope}|{r.module_name}" not in {f"{x.cell_scope}|{x.module_name}" for x in best_direct_rows}]
        if extras:
            best_direct_rows.append(extras[0])
        else:
            best_direct_rows.append(None)
            break
    mixed = []
    pb = intrinsic[(intrinsic["cell_scope"] == "B|true_annotation_plasmablast_B") & (intrinsic["direction"] == "APO_low")].sort_values("p_value")
    mixed.append(next(pb.itertuples(), None))
    mixed.extend([r for r in source_rec.itertuples()][:2])
    while len(mixed) < 3:
        mixed.append(None)

    def get(scope, module):
        x = stats[(stats["cell_scope"] == scope) & (stats["module_name"] == module)]
        return next(x.itertuples(), None) if not x.empty else None

    baff_april_set = [
        get("B|true_annotation_plasmablast_B", "BAFF_APRIL_receptor_response_B"),
        get("myeloid|FCGR3A_CD16_monocyte", "BAFF_APRIL_ligand_source"),
        get("myeloid|cDC", "BAFF_APRIL_ligand_source"),
    ]
    excluded = [
        "myeloid|pDC | antibody_secretion (gene-set/cell-scope mismatch)",
        "myeloid|pDC | plasma_cell_differentiation (gene-set/cell-scope mismatch)",
        "pDC modules do not provide B-cell-intrinsic evidence",
    ]
    rows = [
        {
            "combination_name": "Best direct B-lineage combination",
            "module_1": modstr(best_direct_rows[0]) if len(best_direct_rows) > 0 else "insufficient_candidate",
            "module_2": modstr(best_direct_rows[1]) if len(best_direct_rows) > 1 else "insufficient_candidate",
            "module_3": modstr(best_direct_rows[2]) if len(best_direct_rows) > 2 else "insufficient_candidate",
            "rationale": "Prioritize B-intrinsic modules in B/plasmablast scopes; do not force three if support is weak.",
            "statistical_strength": "based on nominal p ranking among APO_low B_intrinsic results",
            "biological_coherence": "direct B-lineage evidence when modules are interpretable and non-IG-sensitive",
            "caveats": "A mixed or source-side summary is used when fewer than three direct B-cell candidates meet the selection criteria.",
        },
        {
            "combination_name": "Mixed B-intrinsic and source-side combination",
            "module_1": modstr(mixed[0]),
            "module_2": modstr(mixed[1]),
            "module_3": modstr(mixed[2]),
            "rationale": "Combine plasmablast/B-intrinsic evidence with source-side B survival or communication signals.",
            "statistical_strength": "ranked by nominal p value and direction",
            "biological_coherence": "links module scores to the cell-communication analysis",
            "caveats": "Source-side modules are microenvironmental signals, not direct B-cell-intrinsic evidence.",
        },
        {
            "combination_name": "BAFF/APRIL-focused combination",
            "module_1": modstr(baff_april_set[0]),
            "module_2": modstr(baff_april_set[1]),
            "module_3": modstr(baff_april_set[2]),
            "rationale": "Plasmablast BAFF/APRIL receptor response with myeloid/APC ligand-source modules.",
            "statistical_strength": "use if no more coherent mixed B-lineage combination is stronger",
            "biological_coherence": "consistent B/plasmablast survival-maintenance axis",
            "caveats": "This set is less diverse because all three modules focus on BAFF/APRIL.",
        },
        {
            "combination_name": "Excluded cross-lineage combination",
            "module_1": excluded[0],
            "module_2": excluded[1],
            "module_3": excluded[2],
            "rationale": "Documents cross-lineage gene-set and cell-scope mismatch.",
            "statistical_strength": "nominal significance does not resolve the cell-scope mismatch",
            "biological_coherence": "poor for B-cell-intrinsic claim",
            "caveats": "pDC antibody/plasma gene sets are cross-lineage mismatch; not pDC to plasma-cell differentiation.",
        },
    ]
    return pd.DataFrame(rows)


def tuple_from_modstr(s: str) -> tuple[str, str] | None:
    if "|" not in s or s == "insufficient_candidate" or s.startswith("do not"):
        return None
    parts = [x.strip() for x in s.split("|")]
    if len(parts) < 2:
        return None
    scope = "|".join(parts[:-1]).strip()
    module = parts[-1].strip()
    return scope, module


def write_readme(
    stats: pd.DataFrame,
    scores: pd.DataFrame,
    b_states: pd.DataFrame,
    intrinsic: pd.DataFrame,
    source: pd.DataFrame,
    combos: pd.DataFrame,
    h5ad_read: list[str],
    layer_summary: list[str],
):
    sig = stats[stats["p_value"] < 0.05]
    borderline = stats[(stats["p_value"] >= 0.05) & (stats["p_value"] < 0.10)]
    direct_rec = intrinsic[intrinsic["recommend_for_display"]]
    source_rec = source[source["recommend_for_display"]]
    mixed_set_meets_criteria = "yes" if len(direct_rec) >= 2 and len(source_rec) >= 1 else "no"
    if mixed_set_meets_criteria == "no":
        recommendation = "The BAFF/APRIL-focused module set is retained."
    else:
        recommendation = "The mixed B-intrinsic and source-side module set meets the selection criteria."
    readme = f"""# B-lineage and humoral module analysis

## Inputs

- APO grouping source: `{os.environ.get("SLEMODEL_SCRNA_METADATA", str(ROOT / "data" / "single_cell" / "sample_metadata.tsv"))}`
- Annotated h5ad paths from `{SCRIPT36}`

## h5ad / Expression Layer

Read h5ad: YES, read-only with h5py. No h5ad file was modified.

Expression layer used: `{'; '.join(layer_summary)}`. Matrix priority: `layers/log1p_norm`, `layers/log1p_renorm`, `layers/lognorm`, `layers/log1p`, then `X`.

Cell annotation fields and scope definitions follow `masks_for_object()` in the main single-cell analysis.

## B-cell states found

{b_states.to_string(index=False)}

## Method

Targeted B-lineage and source-side modules were scored per sample and cell scope using the mean log-normalized expression of module genes. APO versus non-APO comparisons included 8 and 9 participants, respectively. Two-sided Mann-Whitney U tests were followed by BH correction across targeted tests.

Glycosylation modules are analyzed separately in the B-cell glycosylation analysis.

## Counts

- Tested targeted module tests: {len(stats)}
- nominal p<0.05: {len(sig)}
- borderline 0.05<=p<0.10: {len(borderline)}

## Direct B-lineage candidates recommended

{direct_rec[['cell_scope','module_name','direction','p_value','recommend_reason']].to_string(index=False) if len(direct_rec) else 'None'}

## Source-side candidates recommended

{source_rec[['cell_scope','module_name','direction','p_value','recommend_reason']].to_string(index=False) if len(source_rec) else 'None'}

Source-side candidates are source-side or microenvironmental signal, not direct B-cell-intrinsic evidence.

## Recommended module combinations

{combos.to_string(index=False)}

## Excluded cross-lineage modules

The pDC antibody-secretion and plasma-cell-differentiation gene sets are excluded from B-cell-intrinsic interpretation because their gene-set and cell-scope definitions do not match.

## Selected module set

{recommendation}

## Output files

- `targeted_Blineage_module_definitions.tsv`
- `targeted_Blineage_module_scores_by_sample.tsv`
- `targeted_Blineage_module_stats_all.tsv`
- `targeted_Blineage_module_nominal_p005.tsv`
- `targeted_Blineage_module_borderline_p005_to_p010.tsv`
- `B_intrinsic_candidates_for_display.tsv`
- `source_side_candidates_for_display.tsv`
- `ranked_module_combinations.tsv`
- `weak_or_not_recommended_modules.tsv`
- `preview_best_Blineage_combination.pdf/png/svg`
- `preview_mixed_B_source_combination.pdf/png/svg`
- `preview_BAFF_APRIL_combination.pdf/png/svg`
"""
    (OUT / "README_Blineage_humoral_modules.md").write_text(readme)
    return mixed_set_meets_criteria, recommendation


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_module()
    group_map = sample_grouping(analysis)
    defs = module_definitions_df()
    defs.to_csv(OUT / "targeted_Blineage_module_definitions.tsv", sep="\t", index=False)

    scores, b_states, h5ad_read, layer_summary, matrix_keys = score_modules(analysis, group_map)
    scores.to_csv(OUT / "targeted_Blineage_module_scores_by_sample.tsv", sep="\t", index=False)
    b_states.to_csv(OUT / "B_cell_states_found.tsv", sep="\t", index=False)
    stats = mann_stats(scores, group_map)
    stats.to_csv(OUT / "targeted_Blineage_module_stats_all.tsv", sep="\t", index=False)
    sig = stats[stats["p_value"] < 0.05].sort_values("p_value")
    borderline = stats[(stats["p_value"] >= 0.05) & (stats["p_value"] < 0.10)].sort_values("p_value")
    sig.to_csv(OUT / "targeted_Blineage_module_nominal_p005.tsv", sep="\t", index=False)
    borderline.to_csv(OUT / "targeted_Blineage_module_borderline_p005_to_p010.tsv", sep="\t", index=False)

    annotated = annotate_candidates(stats)
    intrinsic = annotated[annotated["module_class"] == "B_intrinsic"].sort_values("p_value")
    source = annotated[annotated["module_class"] == "source_side"].sort_values("p_value")
    intrinsic.to_csv(OUT / "B_intrinsic_candidates_for_display.tsv", sep="\t", index=False)
    source.to_csv(OUT / "source_side_candidates_for_display.tsv", sep="\t", index=False)
    weak = annotated[(annotated["p_value"] < 0.10) & (~annotated["recommend_for_display"])].copy()
    weak["not_recommended_reason"] = weak["recommend_reason"]
    weak.to_csv(OUT / "weak_or_not_recommended_modules.tsv", sep="\t", index=False)

    combos = select_combinations(intrinsic, source, annotated)
    combos.to_csv(OUT / "ranked_module_combinations.tsv", sep="\t", index=False)

    for combo_name, outstem in [
        ("Best direct B-lineage combination", "preview_best_Blineage_combination"),
        ("Mixed B-intrinsic and source-side combination", "preview_mixed_B_source_combination"),
        ("BAFF/APRIL-focused combination", "preview_BAFF_APRIL_combination"),
    ]:
        row = combos[combos["combination_name"] == combo_name].iloc[0]
        mods = [tuple_from_modstr(row[f"module_{i}"]) for i in [1, 2, 3]]
        plot_combo(scores, annotated, [m for m in mods if m is not None], combo_name, outstem)

    mixed_set_meets_criteria, recommendation = write_readme(
        stats, scores, b_states, intrinsic, source, combos, h5ad_read, layer_summary
    )

    h5ad_audit = pd.DataFrame(
        [{"object_name": k, "h5ad_path": str(analysis.H5ADS[k]), "expression_layer": v, "read_only": True} for k, v in matrix_keys.items()]
    )
    h5ad_audit.to_csv(OUT / "h5ad_readonly_input_audit.tsv", sep="\t", index=False)
    group_map.to_csv(OUT / "analysis_grouping.tsv", sep="\t", index=False)

    direct_rec = intrinsic[intrinsic["recommend_for_display"]]
    source_rec = source[source["recommend_for_display"]]
    print(f"OUTPUT_DIR={OUT}")
    print("H5AD_READ=YES_READ_ONLY")
    print("EXPRESSION_LAYER=" + ";".join(layer_summary))
    print("B_CELL_STATES_FOUND=" + "; ".join(b_states["cell_scope"].tolist()))
    print(f"TESTED_TARGETED_MODULE_TESTS={len(stats)}")
    print(f"NOMINAL_P005={len(sig)}")
    print(f"BORDERLINE_P005_TO_P010={len(borderline)}")
    print("DIRECT_B_RECOMMENDED=" + ("; ".join(f"{r.cell_scope} | {r.module_name}" for r in direct_rec.itertuples()) if len(direct_rec) else "None"))
    print("SOURCE_SIDE_RECOMMENDED=" + ("; ".join(f"{r.cell_scope} | {r.module_name}" for r in source_rec.itertuples()) if len(source_rec) else "None"))
    print("MIXED_SET_MEETS_CRITERIA=" + mixed_set_meets_criteria.upper())
    print("SELECTED_MODULE_SET=" + recommendation)


if __name__ == "__main__":
    main()
