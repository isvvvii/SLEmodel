#!/usr/bin/env python3
"""Reference implementation of the single-cell QC and PCA workflow.

This workflow applies per-sample Scrublet filtering, recorded quality-control
thresholds, normalization, highly variable gene selection and PCA to approved
10x-format count matrices. Outputs provide the preprocessing boundary required
by the downstream single-cell workflow without embedding participant identifiers
or private paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply per-sample Scrublet and QC to 10x-format count matrices, "
            "merge retained cells, normalize counts, select HVGs, and compute PCA."
        )
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path(os.environ.get("SLEMODEL_SCRNA_MATRIX_ROOT", "data/single_cell/raw")),
        help="Directory containing one 10x-format matrix directory per sample.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            os.environ.get(
                "SLEMODEL_SCRNA_PREPROCESS_METADATA",
                "analyses/single_cell/config/sample_metadata.tsv",
            )
        ),
        help=(
            "CSV/TSV metadata with sample_id and optional matrix_dir. "
            "All additional columns are copied to AnnData.obs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "SLEMODEL_SCRNA_PREPROCESS_OUTPUT_DIR",
                "outputs/single_cell/preprocessing",
            )
        ),
    )
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--max-counts", type=float, default=40000)
    parser.add_argument("--max-pct-mito", type=float, default=20.0)
    parser.add_argument("--max-doublet-score", type=float, default=0.25)
    parser.add_argument(
        "--expected-doublet-rate",
        type=float,
        default=0.05,
        help="Scrublet expected doublet rate (default: 0.05).",
    )
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the metadata and matrix-directory layout without reading matrices.",
    )
    return parser


def read_metadata(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = pd.read_csv(path, sep=None, engine="python", dtype={"sample_id": str})
    if "sample_id" not in metadata.columns:
        raise ValueError("Metadata must contain a sample_id column")
    if metadata["sample_id"].isna().any() or metadata["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be present and unique")
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    return metadata


def resolve_matrix_directories(metadata, matrix_root: Path) -> list[tuple[object, Path]]:
    resolved: list[tuple[object, Path]] = []
    for row in metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        matrix_dir_value = getattr(row, "matrix_dir", None)
        if matrix_dir_value is None or str(matrix_dir_value).strip() in {"", "nan", "NA"}:
            matrix_dir = matrix_root / f"{sample_id}.matrix"
        else:
            matrix_dir = Path(str(matrix_dir_value))
            if not matrix_dir.is_absolute():
                matrix_dir = matrix_root / matrix_dir
        resolved.append((row, matrix_dir.expanduser().resolve()))
    return resolved


def validate_matrix_directory(matrix_dir: Path) -> None:
    required_stems = ("matrix.mtx", "features.tsv", "barcodes.tsv")
    missing = [
        stem
        for stem in required_stems
        if not (matrix_dir / stem).exists() and not (matrix_dir / f"{stem}.gz").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{matrix_dir} is not a complete 10x-format directory; missing {missing}"
        )


def run(args: argparse.Namespace) -> None:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc

    matrix_root = args.matrix_root.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = read_metadata(metadata_path)
    resolved = resolve_matrix_directories(metadata, matrix_root)
    for _, matrix_dir in resolved:
        validate_matrix_directory(matrix_dir)

    if args.dry_run:
        print(f"Validated {len(resolved)} sample matrix directories.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    retained_objects = []
    qc_rows: list[dict[str, object]] = []
    metadata_columns = [column for column in metadata.columns if column != "matrix_dir"]

    for row, matrix_dir in resolved:
        sample_id = str(row.sample_id)
        print(f"[QC] {sample_id}: {matrix_dir}")
        adata = sc.read_10x_mtx(
            matrix_dir,
            var_names="gene_symbols",
            make_unique=True,
            cache=False,
            gex_only=True,
        )
        adata.obs_names = [f"{sample_id}:{barcode}" for barcode in adata.obs_names]
        adata.var_names_make_unique()
        upper_names = adata.var_names.str.upper()
        adata.var["mt"] = upper_names.str.startswith("MT-")
        adata.var["ribo"] = upper_names.str.startswith(("RPS", "RPL"))
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        sc.pp.scrublet(
            adata,
            expected_doublet_rate=args.expected_doublet_rate,
            random_state=args.random_seed,
        )
        if "predicted_doublet" in adata.obs:
            adata.obs["doublet_predicted"] = adata.obs["predicted_doublet"].astype(bool)
            del adata.obs["predicted_doublet"]
        else:
            adata.obs["doublet_predicted"] = (
                adata.obs["doublet_score"] >= args.max_doublet_score
            )

        pass_low_genes = adata.obs["n_genes_by_counts"] >= args.min_genes
        pass_high_genes = adata.obs["n_genes_by_counts"] <= args.max_genes
        pass_counts = adata.obs["total_counts"] < args.max_counts
        pass_mito = adata.obs["pct_counts_mt"] < args.max_pct_mito
        pass_doublet = adata.obs["doublet_score"] < args.max_doublet_score
        keep = (
            pass_low_genes
            & pass_high_genes
            & pass_counts
            & pass_mito
            & pass_doublet
        )

        qc_rows.append(
            {
                "sample_id": sample_id,
                "n_raw": int(adata.n_obs),
                "n_lost_low_genes": int((~pass_low_genes).sum()),
                "n_lost_high_genes": int((~pass_high_genes).sum()),
                "n_lost_high_umi": int((~pass_counts).sum()),
                "n_lost_high_mt": int((~pass_mito).sum()),
                "n_lost_doublet": int((~pass_doublet).sum()),
                "n_after_qc": int(keep.sum()),
                "pct_kept": round(float(keep.mean() * 100), 1),
            }
        )

        adata = adata[keep].copy()
        row_values = row._asdict()
        for column in metadata_columns:
            adata.obs[column] = row_values[column]
        retained_objects.append(adata)

    merged = ad.concat(
        retained_objects,
        join="outer",
        merge="same",
        uns_merge="first",
        index_unique=None,
    )
    merged.obs_names_make_unique()
    # pandas may return nullable StringDtype indexes for 10x inputs. anndata
    # 0.11.4 cannot write these unless an experimental setting is enabled.
    merged.obs_names = pd.Index([str(value) for value in merged.obs_names], dtype=object)
    merged.var_names = pd.Index([str(value) for value in merged.var_names], dtype=object)
    for frame in (merged.obs, merged.var):
        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.StringDtype):
                frame[column] = frame[column].astype(object)
    upper_names = merged.var_names.str.upper()
    merged.var["mt"] = upper_names.str.startswith("MT-")
    merged.var["ribo"] = upper_names.str.startswith(("RPS", "RPL"))

    merged.layers["counts"] = merged.X.copy()
    merged.raw = merged.copy()
    sc.pp.normalize_total(merged, target_sum=args.target_sum)
    sc.pp.log1p(merged)
    merged.layers["log1p_norm"] = merged.X.copy()

    sc.pp.highly_variable_genes(
        merged,
        layer="counts",
        flavor="seurat_v3",
        n_top_genes=args.n_hvg,
        batch_key="sample_id",
        subset=False,
    )
    merged.var["highly_variable"] = (
        merged.var["highly_variable"].astype(bool)
        & ~merged.var["mt"].astype(bool)
        & ~merged.var["ribo"].astype(bool)
    )

    hvg_mask = merged.var["highly_variable"].to_numpy()
    pca_input = merged[:, hvg_mask].copy()
    sc.pp.scale(pca_input, max_value=10)
    sc.tl.pca(
        pca_input,
        n_comps=args.n_pcs,
        mask_var=None,
        random_state=args.random_seed,
    )
    merged.obsm["X_pca"] = pca_input.obsm["X_pca"].copy()
    merged.uns["pca"] = pca_input.uns["pca"].copy()
    loadings = np.zeros((merged.n_vars, args.n_pcs), dtype=np.float32)
    loadings[hvg_mask] = pca_input.varm["PCs"]
    merged.varm["PCs"] = loadings

    qc_thresholds = {
        "min_genes": args.min_genes,
        "max_genes": args.max_genes,
        "max_counts": args.max_counts,
        "max_pct_mito": args.max_pct_mito,
        "doublet_score": args.max_doublet_score,
    }
    merged.uns["qc_thresholds"] = qc_thresholds
    merged.uns["preprocessing_reference"] = {
        "original_script_available": False,
        "scanpy_scrublet_expected_doublet_rate": args.expected_doublet_rate,
        "target_sum": args.target_sum,
        "hvg_flavor": "seurat_v3",
        "requested_hvg": args.n_hvg,
        "n_pcs": args.n_pcs,
        "random_seed": args.random_seed,
    }

    qc_summary = pd.DataFrame(qc_rows)
    qc_summary.to_csv(output_dir / "qc_summary.csv", index=False)
    with (output_dir / "preprocessing_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "qc_thresholds": qc_thresholds,
                "expected_doublet_rate": args.expected_doublet_rate,
                "target_sum": args.target_sum,
                "hvg_flavor": "seurat_v3",
                "requested_hvg": args.n_hvg,
                "retained_hvg_after_mt_ribo_exclusion": int(hvg_mask.sum()),
                "n_pcs": args.n_pcs,
                "random_seed": args.random_seed,
                "n_samples": int(len(resolved)),
                "n_cells_post_qc": int(merged.n_obs),
                "n_genes": int(merged.n_vars),
            },
            handle,
            indent=2,
        )
    merged.write_h5ad(output_dir / "adata_pca.h5ad")
    print(
        f"Wrote {merged.n_obs:,} cells x {merged.n_vars:,} genes "
        f"to {output_dir / 'adata_pca.h5ad'}"
    )


def main() -> None:
    args = build_parser().parse_args()
    metadata = read_metadata(args.metadata.expanduser().resolve())
    resolved = resolve_matrix_directories(metadata, args.matrix_root.expanduser().resolve())
    for _, matrix_dir in resolved:
        validate_matrix_directory(matrix_dir)
    if args.dry_run:
        print(f"Validated {len(resolved)} sample matrix directories.")
        return
    run(args)


if __name__ == "__main__":
    main()
