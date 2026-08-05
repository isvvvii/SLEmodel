# Single-cell analysis

This directory contains the APO versus non-APO analyses. It includes sample-level module scoring,
raw-count pseudobulk differential expression, pathway over-representation
analysis, CellPhoneDB analysis, the B-lineage module refinement, and the
plotting scripts.

## Scope of reproducibility

The released single-cell workflow reproduces the reported APO versus non-APO analyses 
from processed and annotated AnnData objects. 
Raw sequencing files, count matrices, processed AnnData objects and participant metadata 
are not publicly distributed because they contain controlled human data.

A reference preprocessing workflow is provided in scripts/preprocess_raw_counts_reference.py. 
It implements the recorded per-sample Scrublet filtering, quality-control thresholds, 
normalization, highly variable gene selection and principal component analysis settings. 
The processed checkpoint used in the study contained 191,563 cells, 37,487 genes, 2,996 highly 
variable genes after exclusion of mitochondrial and ribosomal genes, and 50 principal components.

The six annotated AnnData objects listed below are required to reproduce the downstream statistical analyses.

## Required inputs

Place the following files in `data/single_cell/`, or set
`SLEMODEL_SCRNA_INPUT_DIR` to their directory:

- `adata_full_atlas_final.h5ad`
- `adata_bcell_annotated.h5ad`
- `adata_cd4_annotated.h5ad`
- `adata_cd8_annotated.h5ad`
- `adata_nk_annotated.h5ad`
- `adata_myeloid_annotated.h5ad`
- `sample_metadata.tsv`

The metadata table must contain one row per participant and the columns
`sample_id` and `apo_group`; `apo_group` must be either `APO` or `non-APO`.
Optional columns are shown in `config/sample_metadata.example.tsv`. 
For reference preprocessing, `matrix_dir` gives the sample's 10x-format
matrix directory relative to `SLEMODEL_SCRNA_MATRIX_ROOT`; when omitted, the
script uses `<sample_id>.matrix`.

The AnnData objects must contain a sample identifier in `obs` and raw counts
in a count layer usable for pseudobulk analysis. Normalized log-expression is
used for module scoring. The script records the selected layers in its audit
tables.

## Environment

The reported single-cell analysis used Scanpy 1.11.5, anndata 0.11.4, edgeR
4.4.0 and CellPhoneDB 5.0.1. Create the separate environment with:

```bash
conda env create -f analyses/single_cell/environment.yml
conda activate slemodel-single-cell
```

MSigDB/GO, Reactome and KEGG GMT files and the CellPhoneDB database must be
obtained from their respective providers and are not redistributed here.

## Preprocessing from count matrices

The recorded workflow applied Scrublet separately to each sample and retained cells
with 200–6,000 detected genes, fewer than 40,000 UMIs, less than 20%
mitochondrial transcripts and a Scrublet score below 0.25. Counts were
normalized to 10,000 per cell and log1p-transformed. Three thousand HVGs were
requested with the Seurat v3 method; mitochondrial and ribosomal genes were
removed from the HVG set before 50-component PCA.

The reference script sets the Scrublet expected doublet rate to 0.05 by default 
and allows this value to be changed through a command-line argument:

```bash
export SLEMODEL_SCRNA_MATRIX_ROOT=/approved/path/to/10x_matrices
export SLEMODEL_SCRNA_PREPROCESS_METADATA=/approved/path/to/sample_metadata.tsv
python analyses/single_cell/scripts/preprocess_raw_counts_reference.py \
  --output-dir outputs/single_cell/preprocessing
```

The script writes `qc_summary.csv`, `preprocessing_parameters.json` and `adata_pca.h5ad`. 

## Run the analysis

From the repository root:

```bash
export SLEMODEL_SCRNA_ROOT="$PWD"
export SLEMODEL_SCRNA_INPUT_DIR=/approved/path/to/processed_h5ad
export SLEMODEL_SCRNA_METADATA=/approved/path/to/sample_metadata.tsv
export SLEMODEL_SCRNA_OUTPUT_DIR="$PWD/outputs/single_cell"
export SLEMODEL_CELLPHONEDB_DATABASE_DIR=/path/to/cellphonedb_database

python analyses/single_cell/scripts/run_apo_analysis.py
python analyses/single_cell/scripts/refine_bcell_modules.py
```

CellPhoneDB is run separately in the APO and non-APO groups with 1,000
permutations. Between-group communication differences are descriptive, as
stated in the Methods.

## Visualization

The plotting scripts encode the displayed
feature and pathway selections:

```bash
python analyses/single_cell/figures/plot_atlas_and_composition.py \
  --atlas /approved/path/to/adata_full_atlas_final.h5ad \
  --metadata /approved/path/to/sample_metadata.tsv \
  --output-dir outputs/single_cell/figures/atlas
Rscript analyses/single_cell/figures/plot_module_scores.R
Rscript analyses/single_cell/figures/plot_communication_glycosylation.R
python analyses/single_cell/figures/plot_deg_pathway_summary.py
```

These scripts use statistical source tables produced by the controlled-data
analysis. The expected source-table locations can be overridden with
`SLEMODEL_MODULE_SCORE_SOURCE`,
`SLEMODEL_COMMUNICATION_SOURCE_DIR`,
`SLEMODEL_SCRNA_GLYCO_SOURCE_DIR`, and
`SLEMODEL_DEG_PATHWAY_SOURCE_DIR`.
