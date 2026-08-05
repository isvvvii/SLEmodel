#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kegg_hsa_mapping.py
--------------------------------
Retrieve mappings between human KEGG pathways and KEGG compounds for the
mass-spectrometry enrichment analysis.

Outputs in ``--outdir``:
1) kegg_hsa_pathways.tsv
   - hsa_id, map_id, pathway_name
2) kegg_compound_list.tsv
   - cpd_id, names_raw
3) kegg_hsa_cpd_pairs.tsv
   - hsa_id, map_id, cpd_id
4) kegg_hsa_compound_pathway_map.csv
   - hsa_id, map_id, pathway_name, cpd_id, cpd_name
5) kegg_name_to_cpd.tsv
   - name_norm, comma-separated cpd_ids
6) kegg_hsa_pathway_to_mets.json
   - pathway name to normalized metabolite names
7) kegg_hsa_pathway_to_cids.json
   - pathway name to KEGG compound identifiers

Usage:
    python kegg_hsa_mapping.py --outdir ./kegg_cache
Optional:
    --timeout 10         # request timeout in seconds
    --sleep 0.2          # interval between REST calls
    --max-pathways 50    # limit the number of pathways for testing
    --force              # overwrite existing outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd

BASE = "https://rest.kegg.jp"

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

@dataclass
class KeggClient:
    base: str = BASE
    timeout: int = 10
    sleep: float = 0.2
    retries: int = 3
    backoff: float = 1.5

    def _get(self, path: str) -> str:
        url = f"{self.base}{path}"
        last_exc = None
        for k in range(self.retries):
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    time.sleep(self.sleep)
                    return r.text
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
            except Exception as e:
                last_exc = e
            time.sleep(self.backoff ** k)
        raise RuntimeError(f"GET failed after {self.retries} tries: {url} :: {last_exc}")

    def list_hsa_pathways(self) -> List[Tuple[str, str, str]]:
        """
        Return tuples of ``(hsa_id, map_id, cleaned_name)``.
        KEGG: /list/pathway/hsa   ->  path:hsa00010<TAB>Glycolysis / Gluconeogenesis - Homo sapiens (human)
        """
        txt = self._get("/list/pathway/hsa")
        rows = []
        for line in txt.strip().splitlines():
            if not line.strip():
                continue
            pid, name = line.split("\t", 1)
            hsa_id = pid.replace("path:", "").strip()              # hsa00010
            map_id = "map" + hsa_id[3:]                            # map00010
            name_clean = re.sub(r"\s*-\s*Homo sapiens.*$", "", name).strip()
            rows.append((hsa_id, map_id, name_clean))
        return rows

    def compounds_in_map(self, map_id: str) -> List[str]:
        """
        KEGG: /link/cpd/map00010  ->  path:map00010<TAB>cpd:C00022
        Return compound identifiers without the KEGG prefix.
        """
        txt = self._get(f"/link/cpd/{map_id}")
        cpds = []
        for line in txt.strip().splitlines():
            if not line.strip():
                continue
            left, right = line.split("\t", 1)
            cpd = right.replace("cpd:", "").strip()
            if cpd:
                cpds.append(cpd)
        return sorted(set(cpds))

    def list_compounds(self) -> List[Tuple[str, str]]:
        """
        KEGG: /list/compound  ->  cpd:C00022<TAB>Name; Synonym; Synonym...
        Return tuples of ``(compound_id, raw_names)``.
        """
        txt = self._get("/list/compound")
        rows = []
        for line in txt.strip().splitlines():
            if not line.strip():
                continue
            cid, names = line.split("\t", 1)
            cpd_id = cid.replace("cpd:", "").strip()
            rows.append((cpd_id, names.strip()))
        return rows

def build_and_save(outdir: Path, timeout: int, sleep: float, retries: int,
                   max_pathways: Optional[int], force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cli = KeggClient(timeout=timeout, sleep=sleep, retries=retries)

    print("[1/5] Fetching human pathways (hsa) ...")
    hsa_rows = cli.list_hsa_pathways()
    if max_pathways:
        hsa_rows = hsa_rows[:max_pathways]
    df_hsa = pd.DataFrame(hsa_rows, columns=["hsa_id", "map_id", "pathway_name"])
    p1 = outdir / "kegg_hsa_pathways.tsv"
    if p1.exists() and not force:
        print(f"  - Exists: {p1} (use --force to overwrite)")
    else:
        df_hsa.to_csv(p1, sep="\t", index=False)
        print(f"  - Saved: {p1}  ({len(df_hsa)} pathways)")

    print("[2/5] Linking compounds for each pathway (mapXXXXX) ...")
    pair_rows: List[Tuple[str, str, str]] = []  # (hsa_id, map_id, cpd_id)
    for i, (hsa_id, map_id, _) in enumerate(hsa_rows, 1):
        cpds = cli.compounds_in_map(map_id)
        for c in cpds:
            pair_rows.append((hsa_id, map_id, c))
        if i % 25 == 0 or i == len(hsa_rows):
            print(f"  - {i}/{len(hsa_rows)} pathways processed")
    df_pairs = pd.DataFrame(pair_rows, columns=["hsa_id", "map_id", "cpd_id"])
    p2 = outdir / "kegg_hsa_cpd_pairs.tsv"
    if p2.exists() and not force:
        print(f"  - Exists: {p2}")
    else:
        df_pairs.to_csv(p2, sep="\t", index=False)
        print(f"  - Saved: {p2}  ({len(df_pairs)} pairs)")

    print("[3/5] Fetching compound names & synonyms ...")
    comp_rows = cli.list_compounds()
    df_comp = pd.DataFrame(comp_rows, columns=["cpd_id", "names_raw"])
    p3 = outdir / "kegg_compound_list.tsv"
    if p3.exists() and not force:
        print(f"  - Exists: {p3}")
    else:
        df_comp.to_csv(p3, sep="\t", index=False)
        print(f"  - Saved: {p3}  ({len(df_comp)} compounds)")

    print("[4/5] Merging to long table ...")
    df = df_pairs.merge(df_hsa, on=["hsa_id", "map_id"], how="left") \
                 .merge(df_comp, on="cpd_id", how="left")

    def _main_name(s: str) -> str:
        if not isinstance(s, str) or not s.strip():
            return ""
        return s.split(";")[0].strip()
    df["cpd_name"] = df["names_raw"].map(_main_name)

    p4 = outdir / "kegg_hsa_compound_pathway_map.csv"
    if p4.exists() and not force:
        print(f"  - Exists: {p4}")
    else:
        df[["hsa_id", "map_id", "pathway_name", "cpd_id", "cpd_name"]].to_csv(p4, index=False)
        print(f"  - Saved: {p4}  ({len(df)} rows)")

    print("[5/5] Building name→cpd dictionary (normalized) ...")
    name2cids: Dict[str, set] = {}
    for _, row in df_comp.iterrows():
        cid = row["cpd_id"]
        raw = row["names_raw"] or ""
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        for p in parts:
            key = normalize_name(p)
            if not key:
                continue
            name2cids.setdefault(key, set()).add(cid)

    p5 = outdir / "kegg_name_to_cpd.tsv"
    if p5.exists() and not force:
        print(f"  - Exists: {p5}")
    else:
        with p5.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["name_norm", "cpd_ids"])
            for k, vs in sorted(name2cids.items()):
                w.writerow([k, ",".join(sorted(vs))])
        print(f"  - Saved: {p5}  ({len(name2cids)} keys)")

    map_to_name = dict(df_hsa[["map_id", "pathway_name"]].values)
    path_to_cids: Dict[str, List[str]] = {}
    for (mid, _hid), sub in df_pairs.groupby(["map_id", "hsa_id"]):
        pname = map_to_name.get(mid, mid)
        cids = sorted(set(sub["cpd_id"].astype(str)))
        path_to_cids[pname] = cids

    cpd_to_norms: Dict[str, set] = {}
    for _, row in df_comp.iterrows():
        cid = row["cpd_id"]
        raw = row["names_raw"] or ""
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        for p in parts:
            key = normalize_name(p)
            if key:
                cpd_to_norms.setdefault(cid, set()).add(key)

    path_to_norm_mets: Dict[str, List[str]] = {}
    for pname, cids in path_to_cids.items():
        norms = set()
        for c in cids:
            norms |= cpd_to_norms.get(c, set())
        path_to_norm_mets[pname] = sorted(norms)

    pjson_a = outdir / "kegg_hsa_pathway_to_mets.json"
    pjson_b = outdir / "kegg_hsa_pathway_to_cids.json"
    if (pjson_a.exists() or pjson_b.exists()) and not force:
        print(f"  - Exists: {pjson_a.name} / {pjson_b.name}")
    else:
        pjson_a.write_text(json.dumps(path_to_norm_mets, ensure_ascii=False, indent=2), encoding="utf-8")
        pjson_b.write_text(json.dumps(path_to_cids, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  - Saved: {pjson_a}")
        print(f"  - Saved: {pjson_b}")

    print("\n[Done] KEGG human pathway mapping complete.")
    print(f"      Output directory: {outdir.resolve()}")
    print("      Map normalized metabolite names or KEGG compound identifiers")
    print("      to the generated JSON or TSV files before enrichment analysis.")

def parse_args():
    ap = argparse.ArgumentParser(description="Build KEGG hsa pathway ⇄ compound mapping (for mass enrichment)")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    ap.add_argument("--sleep", type=float, default=0.2, help="Interval between REST calls in seconds")
    ap.add_argument("--retries", type=int, default=3, help="Number of retries after a failed request")
    ap.add_argument("--max-pathways", type=int, default=None, help="Optional pathway limit for testing")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output files")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir)
    try:
        build_and_save(outdir=outdir,
                       timeout=args.timeout,
                       sleep=args.sleep,
                       retries=args.retries,
                       max_pathways=args.max_pathways,
                       force=args.force)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
