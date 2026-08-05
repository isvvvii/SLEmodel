#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map mass-spectrometry feature annotations to KEGG compounds.
------------------------------------------------
Map mass feature candidates to KEGG CIDs using normalized name matching.
The input table contains the model features and annotation evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional

import pandas as pd
import numpy as np

# ---------- Name Normalization ----------
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


# ---------- Column Name Helpers ----------
def pick_col(df: pd.DataFrame, cands: List[str], default=None):
    for c in cands:
        if c in df.columns:
            return c
    return default


# ---------- Load Alias Map ----------
def load_alias_map(pth: Path) -> Dict[str, Set[str]]:
    """Load alias TSV: alias -> set of cpd_ids"""
    alias_map: Dict[str, Set[str]] = {}
    if not pth or not pth.exists():
        return alias_map

    df = pd.read_csv(pth, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}

    if "alias" not in cols or "cpd_id" not in cols:
        raise ValueError(f"Alias table needs 'alias' and 'cpd_id' columns. Found: {list(df.columns)}")

    for alias_raw, cids_raw in df[[cols["alias"], cols["cpd_id"]]].values:
        if not isinstance(alias_raw, str) or not isinstance(cids_raw, str):
            continue
        alias_candidates = re.split(r"[;,]", alias_raw)
        cid_candidates = [x.strip() for x in re.split(r"[;,]", cids_raw) if x.strip()]
        for a in alias_candidates:
            a_norm = normalize_name(a)
            if a_norm:
                alias_map.setdefault(a_norm, set()).update(cid_candidates)

    return alias_map


# ---------- Load Blacklist ----------
def load_blacklist(pth: Optional[Path]) -> Set[str]:
    """Load blacklist file (one name per line)"""
    bl = set()
    default_black = {
        "fosfomycin", "ethambutol", "fluorouracil", "m xylene",
        "isometheptene", "bovinocidin", "allyl isothiocyanate"
    }
    bl |= {normalize_name(x) for x in default_black}

    if pth and pth.exists():
        with open(pth, "r", encoding="utf-8") as f:
            for line in f:
                nm = line.strip()
                if nm:
                    bl.add(normalize_name(nm))
    return bl

def main():
    ap = argparse.ArgumentParser(description="Map mass candidates to KEGG CIDs")
    ap.add_argument("--audit", required=True, help="mass_name_mapping_audit.csv path")
    ap.add_argument("--kegg-dir", required=True, help="KEGG cache directory")
    ap.add_argument("--outdir", required=True, help="Output directory")

    ap.add_argument("--match-ppm", type=float, default=60.0, help="PPM threshold (for filtering)")
    ap.add_argument("--allowed-adducts", type=str,
                    default="[M+Na]+,[M+2Na-H]+,[M+K]+,[M+H]+",
                    help="Allowed adduct forms")

    ap.add_argument("--alias-csv", type=str, default=None)
    ap.add_argument("--blacklist", type=str, default=None)
    ap.add_argument("--blacklist-mode", choices=["hard", "soft", "off"], default="hard")
    ap.add_argument("--blacklist-weight", type=float, default=0.2)
    ap.add_argument("--exogenous-weight", type=float, default=0.1,
                    help="Weight multiplier for exogenous compounds (default 0.1)")
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load KEGG name dictionary
    name2cpd_path = Path(args.kegg_dir) / "kegg_name_to_cpd.tsv"
    if not name2cpd_path.exists():
        raise FileNotFoundError(f"Missing {name2cpd_path}. Run kegg_hsa_mapping.py first.")

    name2cpd: Dict[str, List[str]] = {}
    with name2cpd_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            key = row["name_norm"].strip()
            ids = [x.strip() for x in row["cpd_ids"].split(",") if x.strip()]
            if key:
                name2cpd[key] = ids

    # Load alias and blacklist
    alias_map = load_alias_map(Path(args.alias_csv)) if args.alias_csv else {}
    blacklist = load_blacklist(Path(args.blacklist) if args.blacklist else None)

    allowed_adducts = {a.strip().upper() for a in args.allowed_adducts.split(",") if a.strip()}

    # Load audit table
    audit = pd.read_csv(args.audit)

    # Required columns
    required = ["feature", "candidate", "adduct", "ppm", "is_annotated"]
    for col in required:
        if col not in audit.columns:
            raise ValueError(f"Audit table missing required column: {col}")

    # Check for optional columns
    has_exo_col = 'is_exogenous' in audit.columns
    has_weight_col = 'weight_norm' in audit.columns

    rows_mapped = []
    rows_unmatched = []
    feat_to_cpd: Dict[str, Dict[str, float]] = {}

    # Statistics
    n_total = 0
    n_mapped = 0
    n_unmatched = 0
    n_blacklist_block = 0
    n_adduct_block = 0
    n_exogenous_downweighted = 0

    for _, r in audit.iterrows():
        n_total += 1

        feat = str(r.get("feature", "")).strip()
        cand = str(r.get("candidate", "")).strip()
        adduct = str(r.get("adduct", "")).strip().upper()
        ppm_val = r.get("ppm", float('nan'))
        is_ann = r.get("is_annotated", False)

        # Get weight from audit or default to 1.0
        weight = float(r.get("weight_norm", 1.0)) if has_weight_col else 1.0

        # Get exogenous flag
        is_exogenous = bool(r.get("is_exogenous", False)) if has_exo_col else False

        # Skip unannotated
        if not is_ann:
            continue

        # Adduct filter
        if adduct and adduct not in allowed_adducts:
            n_adduct_block += 1
            continue

        # Normalize name
        cand_norm = normalize_name(cand)

        # Blacklist check
        if args.blacklist_mode == "hard" and cand_norm in blacklist:
            n_blacklist_block += 1
            continue
        elif args.blacklist_mode == "soft" and cand_norm in blacklist:
            weight *= args.blacklist_weight

        # PPM filter
        try:
            if float(ppm_val) > args.match_ppm:
                continue
        except (ValueError, TypeError):
            pass

        # Apply exogenous weight penalty
        if is_exogenous:
            weight *= args.exogenous_weight
            n_exogenous_downweighted += 1

        # Map to KEGG CIDs
        cid_set: Set[str] = set()
        src_flags = []

        # Try alias first
        if cand_norm in alias_map:
            cid_set.update(alias_map[cand_norm])
            src_flags.append("alias")

        # Try KEGG name dictionary
        if cand_norm in name2cpd:
            cid_set.update(name2cpd[cand_norm])
            src_flags.append("kegg_name")

        map_source = ",".join(sorted(set(src_flags))) if src_flags else ""

        if cid_set:
            n_mapped += 1
            for cid in sorted(cid_set):
                rows_mapped.append({
                    "feature": feat,
                    "candidate": cand,
                    "candidate_norm": cand_norm,
                    "cpd_id": cid,
                    "ppm": ppm_val,
                    "adduct": adduct,
                    "weight": weight,
                    "is_exogenous": is_exogenous,
                    "map_source": map_source
                })
                feat_to_cpd.setdefault(feat, {})
                feat_to_cpd[feat][cid] = feat_to_cpd[feat].get(cid, 0.0) + weight
        else:
            n_unmatched += 1
            rows_unmatched.append({
                "feature": feat,
                "candidate": cand,
                "candidate_norm": cand_norm,
                "ppm": ppm_val,
                "adduct": adduct,
                "weight": weight,
                "is_exogenous": is_exogenous
            })

    # Save outputs
    pd.DataFrame(rows_mapped).to_csv(outdir / "mapped_long.tsv", sep="\t", index=False)
    pd.DataFrame(rows_unmatched).to_csv(outdir / "unmatched.tsv", sep="\t", index=False)

    with (outdir / "feature_to_cpd.json").open("w", encoding="utf-8") as f:
        json.dump(
            {k: [{"cid": cid, "weight": float(w)} for cid, w in sorted(v.items())]
             for k, v in feat_to_cpd.items()},
            f, ensure_ascii=False, indent=2
        )

    # Summary
    lines = [
        f"Total annotated rows: {n_total}",
        f"Mapped to KEGG: {n_mapped}",
        f"Unmatched: {n_unmatched}",
        f"Blocked (adduct): {n_adduct_block}",
        f"Blocked (blacklist): {n_blacklist_block}",
        f"Exogenous downweighted: {n_exogenous_downweighted}",
        f"Exogenous weight factor: {args.exogenous_weight}",
        f"Features with KEGG mapping: {len(feat_to_cpd)}",
    ]
    (outdir / "mapping_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
