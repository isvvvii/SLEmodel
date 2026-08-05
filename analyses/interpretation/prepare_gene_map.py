# prepare_gene_map.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an ENSG-to-HGNC map from a GENCODE GRCh38 GTF and optionally check
coverage of an expression matrix with genes stored as columns.

Usage examples:
  # Build the mapping only
  python prepare_gene_map.py --gtf gencode.v46.annotation.gtf.gz \
      --out ref/ensembl_to_hgnc_GRCh38.csv

  # Build the mapping and check matrix coverage
  python prepare_gene_map.py --gtf gencode.v46.annotation.gtf.gz \
      --out ref/ensembl_to_hgnc_GRCh38.csv \
      --matrix data/rna/rna_enhanced_features.csv
"""
import re, csv, gzip, argparse
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", required=True, help="GENCODE GRCh38 annotation GTF (.gtf or .gtf.gz)")
    ap.add_argument("--out", default="ref/ensembl_to_hgnc_GRCh38.csv", help="Output CSV path")
    ap.add_argument("--matrix", default=None, help="Optional RNA matrix with genes stored as columns")
    return ap.parse_args()

def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8", errors="ignore")

def build_map(gtf_path, out_csv):
    pat = re.compile(r'(gene_id|gene_name) "([^"]+)"')
    seen = set()
    n_gene, n_written = 0, 0
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open_text(gtf_path) as fin, open(out_csv, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["ensembl_gene_id", "hgnc_symbol"])
        for line in fin:
            if not line or line[0] == '#':
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            n_gene += 1
            attrs = dict(pat.findall(parts[8]))
            gid = attrs.get("gene_id","")
            sym = attrs.get("gene_name","")
            gid = re.sub(r"\.\d+$","", gid)
            if gid and sym and (gid, sym) not in seen:
                w.writerow([gid, sym])
                seen.add((gid, sym)); n_written += 1
    return n_gene, n_written

def extract_ensg_from_header(matrix_path):
    """
    Extract Ensembl gene identifiers from a comma- or tab-delimited header and
    remove version suffixes.
    """
    path = Path(matrix_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
    sep = "," if header.count(",") >= header.count("\t") else "\t"
    cols = [c.strip().strip('"') for c in header.split(sep)]
    genes = []
    for c in cols:
        if re.match(r"^ENSG\d+(?:\.\d+)?$", c):
            genes.append(re.sub(r"\.\d+$", "", c))
    return genes

def check_coverage(map_csv, genes):
    mp = {}
    with open(map_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mp[row["ensembl_gene_id"]] = row["hgnc_symbol"]
    total = len(genes)
    hit = sum(1 for g in genes if g in mp)
    miss = total - hit
    rate = (hit / total * 100) if total else 0.0
    misses = [g for g in genes if g not in mp][:10]
    return total, hit, miss, rate, misses

def main():
    args = parse_args()
    print(f"[info] Building map from: {args.gtf}")
    n_gene, n_written = build_map(args.gtf, args.out)
    print(f"[done] GTF parsed. gene rows (feature=gene): {n_gene}; unique ENSG→symbol written: {n_written}")
    print(f"[save] Mapping saved to: {args.out}")

    if args.matrix:
        print(f"[info] Checking coverage against matrix header: {args.matrix}")
        genes = extract_ensg_from_header(args.matrix)
        print(f"[info] Found {len(genes)} ENSG-like gene columns in header.")
        total, hit, miss, rate, misses = check_coverage(args.out, genes)
        print(f"[check] total genes: {total}, mapped: {hit}, missing: {miss}, coverage: {rate:.2f}%")
        if misses:
            print(f"[check] first few missing IDs: {misses}")

if __name__ == "__main__":
    main()
