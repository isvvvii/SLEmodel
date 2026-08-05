# aggregate_interpretability_results.py

import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR_DEFAULT = "interpretability_results"
OUT_DIR_DEFAULT = "interpretability_aggregate"

SEED_RE = re.compile(r"seed_(\d+)")
FOLD_RE = re.compile(r"(Repeat_\d+-Fold_\d+)")

def _find_seed_fold(path: Path):
    """Extract the seed and fold name from a result path."""
    seed, fold = None, None
    for p in path.parts:
        m = SEED_RE.search(p)
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None
        m = FOLD_RE.search(p)
        if m:
            fold = m.group(1)
    return seed, fold

def _safe_read_csv(fp: Path, **kwargs):
    try:
        return pd.read_csv(fp, **kwargs)
    except Exception as e:
        print(f"[WARN] Failed to read {fp}: {e}")
        return None

def collect_files(base_dir: Path):
    """List result CSV files using the analysis output naming convention."""
    return {
        "smd": list(base_dir.glob("**/smd_summary.csv")),
        "smd_negcontrol": list(base_dir.glob("**/smd_summary_negcontrol.csv")),
        "ot_repair": list(base_dir.glob("**/ot_repair_stats.csv")),

        "global": list(base_dir.glob("**/global_alignment_metrics.csv")),
        "locality_mass": list(base_dir.glob("**/locality_mass.csv")) + list(base_dir.glob("**/locality_mass_negcontrol*.csv")),
        "locality_rna": list(base_dir.glob("**/locality_rna.csv")) + list(base_dir.glob("**/locality_rna_negcontrol*.csv")),
        "class_flow_mass": list(base_dir.glob("**/class_flow_matrix_mass.csv")),
        "class_flow_rna": list(base_dir.glob("**/class_flow_matrix_rna.csv")),
        "hit_at_k": list(base_dir.glob("**/hit_at_k.csv")),
        "cka": list(base_dir.glob("**/cka_consistency.csv")),
        "cost_latent": list(base_dir.glob("**/cost_latent_consistency.csv")),

        "within_pair": list(base_dir.glob("**/within_pair_summary.csv")),
        "entropy_corr": list(base_dir.glob("**/entropy_confidence_corr_seed_*.csv")),
    }

def process_smd_negcontrol(files):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def process_ot_repair_stats(files):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        if "seed" not in df.columns:
            df["seed"] = seed
        if "fold" not in df.columns:
            df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def process_smd(files):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def compute_smd_gain_vs_negcontrol(smd_true: pd.DataFrame, smd_neg: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the gain of observed OT over negative controls for each
    seed, fold, group, modality and control type. Positive values indicate
    lower post-match imbalance or a larger reduction in absolute SMD.

    Return columns: seed, fold, modality, group, type, metric and value.
    """
    if smd_true is None or smd_true.empty or smd_neg is None or smd_neg.empty:
        return pd.DataFrame()

    key_cols = ["seed", "fold", "modality", "group"]
    need_true = key_cols + ["post_match_smd", "delta_smd"]
    need_neg = key_cols + ["type", "post_match_smd", "delta_smd"]

    if not set(need_true).issubset(set(smd_true.columns)):
        return pd.DataFrame()
    if not set(need_neg).issubset(set(smd_neg.columns)):
        return pd.DataFrame()

    t = smd_true[need_true].copy()
    n = smd_neg[need_neg].copy()

    merged = n.merge(
        t,
        on=key_cols,
        suffixes=("_neg", "_true"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["gain_post_match_smd"] = merged["post_match_smd_neg"] - merged["post_match_smd_true"]
    merged["gain_delta_smd"] = merged["delta_smd_neg"] - merged["delta_smd_true"]

    out = []
    for metric in ["gain_post_match_smd", "gain_delta_smd"]:
        tmp = merged[key_cols + ["type", metric]].copy()
        tmp["metric"] = metric
        tmp.rename(columns={metric: "value"}, inplace=True)
        out.append(tmp)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def process_global(files):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _tag_locality(fp: Path, modality_guess: str):
    """Infer the modality and control type from a result filename."""
    name = fp.name.lower()
    ctrl = "none"
    if "negcontrol" in name and "random" in name:
        ctrl = "negcontrol_random"
    elif "negcontrol" in name and "shuffled" in name:
        ctrl = "negcontrol_shuffled"
    modality = "Mass" if "mass" in name else ("RNA" if "rna" in name else modality_guess)
    return modality, ctrl

def process_locality(files, default_modality):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        modality, ctrl = _tag_locality(fp, default_modality)
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        df["modality"] = modality
        df["control"] = ctrl
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def process_class_flow(files, modality):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp, index_col=0)
        if df is None or df.empty:
            continue
        long = df.stack().reset_index()
        long.columns = ["gly_label", "modality_label", "flow"]
        long["modality"] = modality
        long["seed"] = seed
        long["fold"] = fold
        rows.append(long)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def process_hit_at_k(files):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def process_cka(files):
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def process_cost_latent(files):
    wide_rows = []
    long_rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["fold"] = fold
        wide_rows.append(df)

        cols = {c.lower(): c for c in df.columns}
        mcol = cols.get("spearman_cost_zs_mass") or cols.get("spearman_cost_zs_m")
        rcol = cols.get("spearman_cost_zs_rna") or cols.get("spearman_cost_zs_r")
        base_cols = ["seed", "fold"]
        if mcol and mcol in df.columns:
            long_rows.append(
                df[base_cols + [mcol]].rename(columns={mcol: "rho"}).assign(modality="Mass")
            )
        if rcol and rcol in df.columns:
            long_rows.append(
                df[base_cols + [rcol]].rename(columns={rcol: "rho"}).assign(modality="RNA")
            )
    wide = pd.concat(wide_rows, ignore_index=True) if wide_rows else pd.DataFrame()
    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    return wide, long

def process_within_pair(files):
    """
    Combine ``within_pair_summary.csv`` files generated by ``interpretability.py``.
      seed, fold,
      matched_mean_mass, rand_mean_mass, delta_mean_mass, perm_p_mass,
      matched_mean_rna,  rand_mean_rna,  delta_mean_rna,  perm_p_rna
    """
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue

        df = df.copy()
        if "seed" not in df.columns:
            df["seed"] = seed
        if "fold" not in df.columns:
            df["fold"] = fold
        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def process_entropy_confidence_corr(files):
    """
    Combine per-seed entropy-confidence correlation files.
      seed, fold,
      rho_entropy_mass_conf, rho_entropy_rna_conf,
      rho_entropy_mass_acc,  rho_entropy_rna_acc
    """
    rows = []
    for fp in files:
        seed, fold = _find_seed_fold(fp)
        df = _safe_read_csv(fp)
        if df is None or df.empty:
            continue

        df = df.copy()
        if "seed" not in df.columns:
            df["seed"] = seed
        if "fold" not in df.columns:
            df["fold"] = fold

        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_within_pair(within_pair: pd.DataFrame):
    """
    Summarize the mean cosine-similarity difference and the proportion of
    permutation tests with P < 0.05 by modality.
    """
    if within_pair is None or within_pair.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = within_pair.copy()

    # ---- Δcos summary ----
    long_rows = []
    mapping = [
        ("Mass", "delta_mean_mass"),
        ("RNA", "delta_mean_rna"),
    ]
    for modality, col in mapping:
        if col not in df.columns:
            continue
        t = df[["seed", "fold", col]].copy()
        t[col] = pd.to_numeric(t[col], errors="coerce")
        t = t.rename(columns={col: "value"})
        t["modality"] = modality
        t["metric"] = "delta_cosine_matched_minus_permuted"
        long_rows.append(t)

    delta_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    summary_delta = add_basic_summary(delta_long, ["modality", "metric"], "value")

    # ---- significance summary (p<0.05) ----
    sig_rows = []
    p_mapping = [
        ("Mass", "perm_p_mass"),
        ("RNA", "perm_p_rna"),
    ]
    for modality, pcol in p_mapping:
        if pcol not in df.columns:
            continue
        pvals = pd.to_numeric(df[pcol], errors="coerce")
        pvals = pvals[np.isfinite(pvals)]
        if pvals.empty:
            continue
        n = int(pvals.shape[0])
        n_sig = int((pvals < 0.05).sum())
        sig_rows.append({
            "modality": modality,
            "n_runs": n,
            "n_sig_p_lt_0_05": n_sig,
            "frac_sig_p_lt_0_05": float(n_sig / max(n, 1)),
        })

    summary_sig = pd.DataFrame(sig_rows)
    return summary_delta, summary_sig

def summarize_entropy_confidence_corr(entropy_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize Spearman correlations between entropy and prediction confidence
    or correctness.
    """
    if entropy_corr is None or entropy_corr.empty:
        return pd.DataFrame()

    df = entropy_corr.copy()

    pairs = [
        ("Mass", "rho_entropy_mass_conf", "rho_entropy_vs_confidence"),
        ("RNA",  "rho_entropy_rna_conf",  "rho_entropy_vs_confidence"),
        ("Mass", "rho_entropy_mass_acc",  "rho_entropy_vs_accuracy"),
        ("RNA",  "rho_entropy_rna_acc",   "rho_entropy_vs_accuracy"),
    ]

    rows = []
    for modality, col, metric_name in pairs:
        if col not in df.columns:
            continue
        t = df[["seed", "fold", col]].copy()
        t[col] = pd.to_numeric(t[col], errors="coerce")
        t = t.rename(columns={col: "value"})
        t["modality"] = modality
        t["metric"] = metric_name
        rows.append(t)

    long_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_basic_summary(long_df, ["modality", "metric"], "value")


def add_basic_summary(df, group_cols, value_col, keep_cols=None):
    """Summarize a long-form value column by the requested grouping columns."""
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    if keep_cols:
        keep = [c for c in keep_cols if c in tmp.columns]
        tmp = tmp[keep + group_cols + [value_col]].copy()
    g = tmp.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(['mean', 'std', 'count']).reset_index()
    out['sem'] = out['std'] / np.sqrt(out['count'].clip(lower=1))
    out['ci95_low'] = out['mean'] - 1.96 * out['sem']
    out['ci95_high'] = out['mean'] + 1.96 * out['sem']
    return out

def build_all_in_one_long(smd, glob, loc_m, loc_r, flow_m, flow_r, hitk, cka, cost_long,
                          within_pair, entropy_corr):
    """Construct a single long-form table with a consistent schema."""
    tables = []

    # SMD
    if not smd.empty:
        for metric in ["pre_match_smd", "post_match_smd", "delta_smd"]:
            if metric in smd.columns:
                t = smd[["seed", "fold", "modality", "group", metric]].copy()
                t["table"] = "smd"
                t["metric"] = metric
                t.rename(columns={metric: "value", "group": "key1"}, inplace=True)
                tables.append(t)

    # Global distances
    if not glob.empty:
        for metric in ["emd_pre", "emd_post", "mmd2_pre", "mmd2_post", "sink_pre", "sink_post", "mean_transport_cost"]:
            if metric in glob.columns:
                t = glob[["seed", "fold", "modality", metric]].copy()
                t["table"] = "global"
                t["metric"] = metric
                t.rename(columns={metric: "value"}, inplace=True)
                tables.append(t)

    # Locality (Mass/RNA)
    def _loc_to_long(df):
        if df.empty:
            return []
        keep = ["row_entropy", "hhi", "effective_donors", "top1", "top3", "top5", "top10"]
        exist = [c for c in keep if c in df.columns]
        out = []
        for metric in exist:
            t = df[["seed", "fold", "modality", "control", metric]].copy()
            t["table"] = "locality"
            t["metric"] = metric
            t.rename(columns={metric: "value", "control": "key1"}, inplace=True)
            out.append(t)
        return out

    if not loc_m.empty:
        tables += _loc_to_long(loc_m)
    if not loc_r.empty:
        tables += _loc_to_long(loc_r)

    # Class flow matrices
    def _flow_to_long(df, modality):
        if df.empty:
            return []
        t = df[["seed", "fold", "flow", "gly_label", "modality_label"]].copy()
        t["table"] = "class_flow"
        t["metric"] = "flow"
        t["modality"] = modality
        t.rename(columns={"gly_label": "key1", "modality_label": "key2", "flow": "value"}, inplace=True)
        return [t]

    tables += _flow_to_long(flow_m, "Mass")
    tables += _flow_to_long(flow_r, "RNA")

    # Hit@k
    if not hitk.empty:
        for metric in ["mean_rank", "hit@1", "hit@5", "hit@10"]:
            if metric in hitk.columns:
                t = hitk[["seed", "fold", "modality", metric]].copy()
                t["table"] = "hit_at_k"
                t["metric"] = metric
                t.rename(columns={metric: "value"}, inplace=True)
                tables.append(t)

    # CKA / ΔCKA (prefer delta_vs_random)
    if not cka.empty:
        preferred = []
        if "delta_vs_random" in cka.columns:
            preferred.append("delta_vs_random")
        if "delta_vs_colperm" in cka.columns:
            preferred.append("delta_vs_colperm")
        if "cka_ot" in cka.columns:
            preferred.append("cka_ot")
        if "cka_random" in cka.columns:
            preferred.append("cka_random")
        if "cka_colperm" in cka.columns:
            preferred.append("cka_colperm")
        if "cka_linear" in cka.columns:
            preferred.append("cka_linear")  # backward compatible

        for col in preferred:
            t = cka[["seed", "fold", "modality", col]].copy()
            t["table"] = "cka"
            t["metric"] = col
            t.rename(columns={col: "value"}, inplace=True)
            tables.append(t)

    # Cost–latent (long)
    if not cost_long.empty:
        t = cost_long[["seed", "fold", "modality", "rho"]].copy()
        t["table"] = "cost_latent"
        t["metric"] = "spearman_rho"
        t.rename(columns={"rho": "value"}, inplace=True)
        tables.append(t)

    # ===== NEW: within-pair (Δcos) =====
    if within_pair is not None and not within_pair.empty:
        for modality, col in [("Mass", "delta_mean_mass"), ("RNA", "delta_mean_rna")]:
            if col not in within_pair.columns:
                continue
            t = within_pair[["seed", "fold", col]].copy()
            t[col] = pd.to_numeric(t[col], errors="coerce")
            t["table"] = "within_pair"
            t["metric"] = "delta_cosine_matched_minus_permuted"
            t["modality"] = modality
            t.rename(columns={col: "value"}, inplace=True)
            tables.append(t)

    # ===== NEW: entropy ↔ confidence/accuracy correlations =====
    if entropy_corr is not None and not entropy_corr.empty:
        pairs = [
            ("Mass", "rho_entropy_mass_conf", "rho_entropy_vs_confidence"),
            ("RNA",  "rho_entropy_rna_conf",  "rho_entropy_vs_confidence"),
            ("Mass", "rho_entropy_mass_acc",  "rho_entropy_vs_accuracy"),
            ("RNA",  "rho_entropy_rna_acc",   "rho_entropy_vs_accuracy"),
        ]
        for modality, col, metric_name in pairs:
            if col not in entropy_corr.columns:
                continue
            t = entropy_corr[["seed", "fold", col]].copy()
            t[col] = pd.to_numeric(t[col], errors="coerce")
            t["table"] = "entropy_corr"
            t["metric"] = metric_name
            t["modality"] = modality
            t.rename(columns={col: "value"}, inplace=True)
            tables.append(t)

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

def summarize_transport_cost(glob: pd.DataFrame, include_total: bool = False) -> pd.DataFrame:
    """
    Build a separate summary table for OT transport-cost statistics.

    Expected columns in `glob` (per-fold raw):
      - mean_transport_cost
      - median_transport_cost
      - (optional) total_transport_cost

    Output schema (mean/std/sem/95%CI) matches other summary_* files:
      modality, metric, mean, std, count, sem, ci95_low, ci95_high
    """
    if glob is None or glob.empty:
        return pd.DataFrame()

    metrics = ["mean_transport_cost", "median_transport_cost"]
    if include_total:
        metrics.append("total_transport_cost")

    rows = []
    for metric in metrics:
        if metric not in glob.columns:
            continue
        t = glob[["modality", metric]].copy()
        t[metric] = pd.to_numeric(t[metric], errors="coerce")
        t.rename(columns={metric: "value"}, inplace=True)
        t["metric"] = metric
        rows.append(t)

    if not rows:
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    return add_basic_summary(long_df, ["modality", "metric"], "value")


def main(base_dir=BASE_DIR_DEFAULT, out_dir=OUT_DIR_DEFAULT):
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Input directory: {base.resolve()}")
    print(f"[INFO] Output directory: {out.resolve()}")

    files = collect_files(base)

    smd = process_smd(files["smd"])
    glob = process_global(files["global"])
    loc_m = process_locality(files["locality_mass"], "Mass")
    loc_r = process_locality(files["locality_rna"],  "RNA")
    flow_m = process_class_flow(files["class_flow_mass"], "Mass")
    flow_r = process_class_flow(files["class_flow_rna"],  "RNA")
    hitk = process_hit_at_k(files["hit_at_k"])
    cka  = process_cka(files["cka"])
    cost_wide, cost_long = process_cost_latent(files["cost_latent"])
    within_pair = process_within_pair(files["within_pair"])
    entropy_corr = process_entropy_confidence_corr(files["entropy_corr"])


    def _save(df, name):
        if df is not None and not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"[OK] Wrote {name}.csv ({len(df)} rows)")
        else:
            print(f"[WARN] No data for {name}")

    _save(smd, "combined_smd_summary_raw")
    _save(glob, "combined_global_alignment_metrics_raw")
    _save(loc_m, "combined_locality_mass_raw")
    _save(loc_r, "combined_locality_rna_raw")
    _save(flow_m, "combined_class_flow_mass_long")
    _save(flow_r, "combined_class_flow_rna_long")
    _save(hitk, "combined_hit_at_k_raw")
    _save(cka,  "combined_cka_consistency_raw")
    _save(cost_wide, "combined_cost_latent_consistency_wide")
    _save(cost_long, "combined_cost_latent_consistency_long")

    smd_neg = process_smd_negcontrol(files["smd_negcontrol"])
    repair = process_ot_repair_stats(files["ot_repair"])

    _save(smd_neg, "combined_smd_summary_negcontrol_raw")
    _save(repair, "combined_ot_repair_stats_raw")
    _save(within_pair, "combined_within_pair_summary_raw")
    _save(entropy_corr, "combined_entropy_confidence_corr_raw")


    if smd_neg is not None and not smd_neg.empty:
        smd_neg_long = pd.DataFrame()
        for metric in ["post_match_smd", "delta_smd"]:
            if metric in smd_neg.columns:
                t = smd_neg[["modality", "group", "type", metric]].rename(columns={metric: "value"}).assign(metric=metric)
                smd_neg_long = pd.concat([smd_neg_long, t], ignore_index=True)
        summary_smd_neg = add_basic_summary(smd_neg_long, ["modality", "group", "type", "metric"], "value")
        _save(summary_smd_neg, "summary_smd_negcontrol_mean_ci")

    gain_long = compute_smd_gain_vs_negcontrol(smd, smd_neg)
    _save(gain_long, "combined_smd_gain_vs_negcontrol_long")
    if gain_long is not None and not gain_long.empty:
        summary_gain = add_basic_summary(gain_long, ["modality", "group", "type", "metric"], "value")
        _save(summary_gain, "summary_smd_gain_vs_negcontrol_mean_ci")

    if repair is not None and not repair.empty:
        repair_long = pd.DataFrame()
        for metric in ["n_repaired_rows", "frac_repaired_rows"]:
            if metric in repair.columns:
                t = repair[["modality", metric]].rename(columns={metric: "value"}).assign(metric=metric)
                repair_long = pd.concat([repair_long, t], ignore_index=True)
        summary_repair = add_basic_summary(repair_long, ["modality", "metric"], "value")
        _save(summary_repair, "summary_ot_repair_mean_ci")



    all_long = build_all_in_one_long(smd, glob, loc_m, loc_r, flow_m, flow_r, hitk, cka, cost_long,
                                     within_pair, entropy_corr)
    _save(all_long, "all_interpretability_long")

    smd_summary = pd.DataFrame()
    if not smd.empty:
        smd_long = pd.DataFrame()
        for metric in ["pre_match_smd", "post_match_smd", "delta_smd"]:
            if metric in smd.columns:
                t = smd[["modality", "group", metric]].rename(columns={metric: "value"}).assign(metric=metric)
                smd_long = pd.concat([smd_long, t], ignore_index=True)
        smd_summary = add_basic_summary(smd_long, ["modality", "group", "metric"], "value")
        _save(smd_summary, "summary_smd_mean_ci")

    # 2) Global distances
    glob_summary = pd.DataFrame()
    if not glob.empty:
        glob_long = pd.DataFrame()
        for metric in ["emd_pre", "emd_post", "mmd2_pre", "mmd2_post", "sink_pre", "sink_post", "mean_transport_cost"]:
            if metric in glob.columns:
                t = glob[["modality", metric]].rename(columns={metric: "value"}).assign(metric=metric)
                glob_long = pd.concat([glob_long, t], ignore_index=True)
        glob_summary = add_basic_summary(glob_long, ["modality", "metric"], "value")
        _save(glob_summary, "summary_global_distances_mean_ci")

    # 2b) Transport cost (separate summary; optional QC metric)
    summary_transport_cost = summarize_transport_cost(glob, include_total=False)
    _save(summary_transport_cost, "summary_transport_cost_mean_ci")

    # 3) Locality
    def _summ_loc(df):
        if df is None or df.empty:
            return pd.DataFrame()
        melted = df.melt(id_vars=["seed", "fold", "modality", "control"],
                         var_name="metric", value_name="value")
        melted = melted[~melted["metric"].isin(["seed", "fold", "modality", "control"])]
        return add_basic_summary(melted, ["modality", "control", "metric"], "value")

    summary_locality_mass = _summ_loc(loc_m)
    summary_locality_rna  = _summ_loc(loc_r)
    _save(summary_locality_mass, "summary_locality_mass_mean_ci")
    _save(summary_locality_rna,  "summary_locality_rna_mean_ci")

    # 4) Class flow
    def _summ_flow(df, modality):
        if df is None or df.empty:
            return pd.DataFrame()
        return add_basic_summary(df.rename(columns={"flow": "value"}),
                                 ["gly_label", "modality_label"], "value").assign(modality=modality)

    summary_class_flow_mass = _summ_flow(flow_m, "Mass")
    summary_class_flow_rna  = _summ_flow(flow_r, "RNA")
    _save(summary_class_flow_mass, "summary_class_flow_mass_mean_ci")
    _save(summary_class_flow_rna,  "summary_class_flow_rna_mean_ci")

    # 5) Hit@k
    summary_hit_at_k_df = pd.DataFrame()
    if not hitk.empty:
        hk_long = pd.DataFrame()
        for metric in ["mean_rank", "hit@1", "hit@5", "hit@10"]:
            if metric in hitk.columns:
                hk_long = pd.concat(
                    [hk_long, hitk[["modality", metric]].rename(columns={metric: "value"}).assign(metric=metric)],
                    ignore_index=True
                )
        summary_hit_at_k_df = add_basic_summary(hk_long, ["modality", "metric"], "value")
        _save(summary_hit_at_k_df, "summary_hit_at_k_mean_ci")

    # 6) CKA / ΔCKA (prefer delta_vs_random)
    summary_cka_df = pd.DataFrame()
    if not cka.empty:
        if "delta_vs_random" in cka.columns:
            summary_cka_df = add_basic_summary(
                cka.rename(columns={"delta_vs_random": "value"}), ["modality"], "value"
            )
            _save(summary_cka_df, "summary_cka_delta_vs_random_mean_ci")
        elif "cka_ot" in cka.columns:
            summary_cka_df = add_basic_summary(
                cka.rename(columns={"cka_ot": "value"}), ["modality"], "value"
            )
            _save(summary_cka_df, "summary_cka_ot_mean_ci")
        elif "cka_linear" in cka.columns:
            summary_cka_df = add_basic_summary(
                cka.rename(columns={"cka_linear": "value"}), ["modality"], "value"
            )
            _save(summary_cka_df, "summary_cka_linear_mean_ci")

    # 7) Cost–latent
    summary_cost_latent_df = pd.DataFrame()
    if not cost_long.empty:
        summary_cost_latent_df = add_basic_summary(
            cost_long.rename(columns={"rho": "value"}), ["modality"], "value"
        )
        _save(summary_cost_latent_df, "summary_cost_latent_rho_mean_ci")

    summary_within_delta, summary_within_sig = summarize_within_pair(within_pair)
    _save(summary_within_delta, "summary_within_pair_delta_cos_mean_ci")
    _save(summary_within_sig, "summary_within_pair_significance")

    # 9) Entropy ↔ confidence/accuracy correlations (mean±CI)
    summary_entropy_corr = summarize_entropy_confidence_corr(entropy_corr)
    _save(summary_entropy_corr, "summary_entropy_confidence_corr_mean_ci")

    xlsx_path = out / "interpretability_summary.xlsx"
    wrote_excel = False
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            for name, df in [
                ("all_interpretability_long", all_long),
                ("smd_raw", smd),
                ("global_raw", glob),
                ("locality_mass_raw", loc_m),
                ("locality_rna_raw", loc_r),
                ("class_flow_mass_long", flow_m),
                ("class_flow_rna_long", flow_r),
                ("hit_at_k_raw", hitk),
                ("cka_raw", cka),
                ("cost_latent_wide", cost_wide),
                ("cost_latent_long", cost_long),
                ("within_pair_raw", within_pair),
                ("entropy_corr_raw", entropy_corr),
                ("summary_smd", smd_summary),
                ("summary_global", glob_summary),
                ("summary_locality_mass", summary_locality_mass),
                ("summary_locality_rna", summary_locality_rna),
                ("summary_class_flow_mass", summary_class_flow_mass),
                ("summary_class_flow_rna", summary_class_flow_rna),
                ("summary_hit_at_k", summary_hit_at_k_df),
                ("summary_cka", summary_cka_df),
                ("summary_transport_cost", summary_transport_cost),
                ("summary_cost_latent", summary_cost_latent_df),
                ("summary_within_pair_delta", summary_within_delta),
                ("summary_within_pair_sig", summary_within_sig),
                ("summary_entropy_corr", summary_entropy_corr)
            ]:
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=name[:31], index=False)
        wrote_excel = True
    except Exception as e:
        print(f"[WARN] xlsxwriter failed; trying the default engine: {e}")

    if not wrote_excel:
        with pd.ExcelWriter(xlsx_path) as writer:
            for name, df in [
                ("all_interpretability_long", all_long),
                ("smd_raw", smd),
                ("global_raw", glob),
                ("locality_mass_raw", loc_m),
                ("locality_rna_raw", loc_r),
                ("class_flow_mass_long", flow_m),
                ("class_flow_rna_long", flow_r),
                ("hit_at_k_raw", hitk),
                ("cka_raw", cka),
                ("cost_latent_wide", cost_wide),
                ("cost_latent_long", cost_long),
                ("within_pair_raw", within_pair),
                ("entropy_corr_raw", entropy_corr),
                ("summary_smd", smd_summary),
                ("summary_global", glob_summary),
                ("summary_locality_mass", summary_locality_mass),
                ("summary_locality_rna", summary_locality_rna),
                ("summary_class_flow_mass", summary_class_flow_mass),
                ("summary_class_flow_rna", summary_class_flow_rna),
                ("summary_hit_at_k", summary_hit_at_k_df),
                ("summary_cka", summary_cka_df),
                ("summary_cost_latent", summary_cost_latent_df),
                ("summary_within_pair_delta", summary_within_delta),
                ("summary_within_pair_sig", summary_within_sig),
                ("summary_entropy_corr", summary_entropy_corr),
            ]:
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"[OK] Wrote Excel workbook: {xlsx_path}")

    index_rows = []
    for csv_path in out.glob("*.csv"):
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                n = sum(1 for _ in f) - 1
        except Exception:
            n = None
        index_rows.append({"file": csv_path.name, "rows": n})
    pd.DataFrame(index_rows).to_csv(out / "INDEX_files_counts.csv", index=False)
    print("[DONE] Aggregation complete.")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR_DEFAULT
    out  = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR_DEFAULT
    main(base, out)
