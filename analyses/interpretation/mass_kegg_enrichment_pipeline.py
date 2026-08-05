#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KEGG enrichment pipeline for mass-spectrometry features."""

from __future__ import annotations
import argparse
import subprocess
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import warnings

from slemodel import config as cfg
from .feature_annotations import build_feature_label_maps, annotate_mass_features_for_enrichment

OUT_ROOT = Path("clinical_explain_results") / "enrichment_results"

# Name normalization (for KEGG matching)
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega"
}
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+")


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    for g, eng in _GREEK.items():
        s = s.replace(g, eng)
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = s.replace("_", " ")
    s = _WS.sub(" ", s).strip()
    return s


def _modality_of(name: str) -> str:
    return name.split("::", 1)[0] if "::" in name else "?"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_shap_npz(npz_path: Path):
    dat = np.load(npz_path, allow_pickle=True)
    phi_stack = dat["phi_stack"]  # (N, F, C)
    X_total = dat["X_total"]
    feature_names = dat["feature_names"].tolist()
    return phi_stack, X_total, feature_names


def prepare_mass_audit(
    feature_names: List[str],
    out_csv: Path,
    match_ppm: float = 50.0,
    allowed_adducts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate mass_name_mapping_audit.csv using serum_metabolite.xlsx exact mass matching.

    This is the simplified version that directly uses feature_annotations module.
    """
    if allowed_adducts is None:
        allowed_adducts = ["[M+Na]+", "[M+2Na-H]+", "[M+K]+", "[M+H]+", "[M+NH4]+"]

    # Use the new batch annotation function
    df = annotate_mass_features_for_enrichment(
        feature_names=feature_names,
        match_ppm=match_ppm,
        allowed_adducts=allowed_adducts
    )

    # Add normalized name column for KEGG matching
    df['candidate_norm'] = df['candidate'].apply(normalize_name)

    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_annotated = df['is_annotated'].sum()
    n_total = len(df['feature'].unique())
    print(f"[OK] Prepared audit file: {out_csv}")
    print(f"     Features: {n_total}, Annotated: {n_annotated}")

    return df


def run_subprocess(argv: List[str]):
    print("[run]", " ".join(argv))
    ret = subprocess.run(argv, stdout=sys.stdout, stderr=sys.stderr)
    if ret.returncode != 0:
        raise SystemExit(f"Subprocess failed: {' '.join(argv)}")


def recommend_aliases(
    unmatched_tsv: Path,
    kegg_compound_list_tsv: Path,
    out_csv: Path,
    min_score: int = 90,
    topk: int = 5
) -> pd.DataFrame:
    """
    Generate alias suggestions for unmatched metabolite names using fuzzy matching.
    """
    cols_out = ["query_raw", "query_norm", "suggestion_norm", "score", "cpd_ids"]

    try:
        from rapidfuzz import process as rf_process, fuzz as rf_fuzz
        have_rf = True
    except ImportError:
        import difflib
        have_rf = False

    def _empty_df():
        df0 = pd.DataFrame(columns=cols_out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df0.to_csv(out_csv, index=False)
        return df0

    if not unmatched_tsv.exists() or unmatched_tsv.stat().st_size == 0:
        print(f"[alias] No unmatched file or empty: {unmatched_tsv}")
        return _empty_df()

    if not kegg_compound_list_tsv.exists():
        print(f"[alias] No KEGG compound list: {kegg_compound_list_tsv}")
        return _empty_df()

    um = pd.read_csv(unmatched_tsv, sep="\t")
    queries_raw = set()
    for _, r in um.iterrows():
        for col in ["candidate", "matched_synonym"]:
            val = r.get(col, "")
            val = "" if pd.isna(val) else str(val).strip()
            if not val or val.lower() in {"nan", "none", "null"}:
                continue
            queries_raw.add(val)

    queries = [(q, normalize_name(q)) for q in sorted(queries_raw)]
    queries = [(q, qn) for (q, qn) in queries if qn]

    if not queries:
        print("[alias] No valid unmatched names to suggest.")
        return _empty_df()

    df_k = pd.read_csv(kegg_compound_list_tsv, sep="\t")
    if not {"cpd_id", "names_raw"}.issubset(set(df_k.columns)):
        print("[alias] KEGG compound list missing columns.")
        return _empty_df()

    # Build normalized synonym -> CID mapping
    norm_to_cids: Dict[str, set] = {}
    for _, r in df_k.iterrows():
        cid = str(r["cpd_id"]).strip()
        names = "" if pd.isna(r.get("names_raw")) else str(r["names_raw"])
        parts = [p.strip() for p in names.split(";") if p.strip()]
        for p in parts:
            pn = normalize_name(p)
            if pn:
                norm_to_cids.setdefault(pn, set()).add(cid)

    all_norm_syns = list(norm_to_cids.keys())

    out_rows = []
    for q_raw, qn in queries:
        if have_rf:
            hits = rf_process.extract(qn, all_norm_syns, scorer=rf_fuzz.token_set_ratio, limit=topk)
            for cand, score, _ in hits:
                if int(score) >= min_score:
                    out_rows.append({
                        "query_raw": q_raw,
                        "query_norm": qn,
                        "suggestion_norm": cand,
                        "score": int(score),
                        "cpd_ids": ",".join(sorted(norm_to_cids.get(cand, [])))
                    })
        else:
            cands = difflib.get_close_matches(qn, all_norm_syns, n=topk, cutoff=min_score/100.0)
            for cand in cands:
                score = int(100 * difflib.SequenceMatcher(None, qn, cand).ratio())
                out_rows.append({
                    "query_raw": q_raw,
                    "query_norm": qn,
                    "suggestion_norm": cand,
                    "score": score,
                    "cpd_ids": ",".join(sorted(norm_to_cids.get(cand, [])))
                })

    if not out_rows:
        print(f"[alias] No suggestions >= min_score={min_score}.")
        return _empty_df()

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values(["query_norm", "score"], ascending=[True, False])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[alias] Suggestions written to: {out_csv} (rows={len(out_df)})")
    return out_df


def main():
    ap = argparse.ArgumentParser(description="Mass KEGG enrichment pipeline from SHAP outputs.")
    ap.add_argument("--shap_npz", type=str,
                    default="clinical_explain_results/shap_outputs/mor_shap_outputs.npz")
    ap.add_argument("--kegg_dir", type=str, required=True,
                    help="Directory containing kegg_name_to_cpd.tsv and kegg_hsa_pathway_to_cids.json")
    ap.add_argument("--outdir", type=str, default=str(OUT_ROOT / "mass_kegg"))

    # Matching parameters (aligned with platform characteristics)
    ap.add_argument("--match_ppm", type=float, default=50.0,
                    help="PPM tolerance for exact mass matching")
    ap.add_argument("--allowed_adducts", type=str,
                    default="[M+Na]+,[M+2Na-H]+,[M+K]+,[M+H]+",
                    help="Allowed adduct forms (comma-separated)")

    # Alias/blacklist
    ap.add_argument("--alias_csv", type=str, default=None)
    ap.add_argument("--blacklist", type=str, default=None)
    ap.add_argument("--blacklist_mode", choices=["hard", "soft", "off"], default="hard")
    ap.add_argument("--blacklist_weight", type=float, default=0.2)

    # Enrichment options
    ap.add_argument("--background", choices=["observed", "kegg"], default="kegg")
    ap.add_argument("--min_size", type=int, default=2)
    ap.add_argument("--min_overlap", type=int, default=2)
    ap.add_argument("--fdr", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=20)

    # Alias suggestion
    ap.add_argument("--suggest-alias", action="store_true",
                    help="Generate alias suggestions for unmatched names")
    ap.add_argument("--suggest-min-score", type=int, default=90)

    args = ap.parse_args()

    shap_npz = Path(args.shap_npz)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)
    audit_dir = outdir / "audit"
    _ensure_dir(audit_dir)

    # Auto-inject alias/blacklist if not specified
    if args.alias_csv is None and Path("ref/mass_alias.tsv").exists():
        args.alias_csv = "ref/mass_alias.tsv"
    if args.blacklist is None and Path("ref/mass_exogenous_blacklist.txt").exists():
        args.blacklist = "ref/mass_exogenous_blacklist.txt"

    # Load SHAP outputs
    phi_stack, X_total, feature_names = load_shap_npz(shap_npz)

    # Prepare audit table using new annotation
    audit_csv = audit_dir / "mass_name_mapping_audit.csv"
    allowed_adducts_list = [x.strip() for x in args.allowed_adducts.split(",") if x.strip()]

    prepare_mass_audit(
        feature_names=feature_names,
        out_csv=audit_csv,
        match_ppm=args.match_ppm,
        allowed_adducts=allowed_adducts_list
    )

    # Step 1: Map names -> KEGG cpd
    kegg_dir = Path(args.kegg_dir)
    name_dict = kegg_dir / "kegg_name_to_cpd.tsv"
    if not name_dict.exists():
        raise FileNotFoundError(f"Missing KEGG name dictionary: {name_dict}. Run kegg_hsa_mapping.py first.")

    map_outdir = audit_dir / "mapping"
    argv_map = [
        sys.executable, "mass_to_kegg_mapper.py",
        "--audit", str(audit_csv),
        "--kegg-dir", str(kegg_dir),
        "--outdir", str(map_outdir),
        "--match-ppm", str(args.match_ppm),
        "--allowed-adducts", args.allowed_adducts,
    ]
    if args.alias_csv:
        argv_map += ["--alias-csv", args.alias_csv]
    if args.blacklist:
        argv_map += ["--blacklist", args.blacklist]
    if args.blacklist_mode:
        argv_map += ["--blacklist-mode", args.blacklist_mode]
    if args.blacklist_weight is not None:
        argv_map += ["--blacklist-weight", str(args.blacklist_weight)]
    argv_map += ["--force"]
    run_subprocess(argv_map)

    feature_to_cpd = map_outdir / "feature_to_cpd.json"
    if not feature_to_cpd.exists():
        raise FileNotFoundError(f"feature_to_cpd.json not found at {feature_to_cpd}")

    # Step 1.5: Alias suggestion (optional)
    if args.suggest_alias:
        unmatched_tsv = map_outdir / "unmatched.tsv"
        kegg_compounds = kegg_dir / "kegg_compound_list.tsv"
        suggest_csv = map_outdir / "alias_suggestions.csv"
        recommend_aliases(unmatched_tsv, kegg_compounds, suggest_csv,
                         min_score=args.suggest_min_score, topk=5)
        print("[alias] Review alias_suggestions.csv and add to ref/mass_alias.tsv if needed.")

    print(f"[Done] Mass KEGG enrichment pipeline completed. Output: {outdir}")


if __name__ == "__main__":
    main()
