# SLEmodel

SLEmodel is a multitask deep-learning framework for integrating unpaired clinical multi-omics cohorts. It uses propensity-score-guided, entropy-regularized optimal transport to construct pseudo-paired samples, learns shared and modality-specific representations, and jointly performs three-class disease-activity classification and cross-modal reconstruction.

This repository contains analysis code for SLEmodel training, alignment diagnostics, ablation studies, traditional machine-learning comparisons, SHAP analysis, and multi-omics biological interpretation reported in:

> *SLEmodel: A Multimodal Deep Learning Framework for Pregnancy-Associated SLE Activity Stratification in Small Cohorts*

## Repository contents

- `src/slemodel/`: final model, training workflow, evaluation, and fold-wise result aggregation
- `analyses/alignment/`: propensity-score and optimal-transport alignment diagnostics
- `analyses/ablation/`: component, architecture, and single-modality ablations
- `analyses/benchmarks/`: traditional machine-learning classification and reconstruction comparisons
- `analyses/interpretation/`: SHAP, glycan-trait, RNA GSEA, mass-spectrometry KEGG, and cross-modal network analyses
- `analyses/single_cell/`: APO versus non-APO PBMC single-cell analysis
- `data/README.md`: required input layout and access information

## Installation

The reported model analyses used Ubuntu 22.04, Python 3.10, PyTorch 2.5.1,
CUDA 12.1, one NVIDIA RTX 4090 GPU (24 GB), and 16 Intel Xeon Gold 6430
vCPUs.

```bash
conda env create -f environment.yml
conda activate slemodel
pip install -e .
```

For a CPU-only installation, install the appropriate PyTorch build for the target system and then run:

```bash
pip install -e ".[analysis]"
```

## Input data

Place the three model-ready modality tables under `data/` as described in [data/README.md](data/README.md). Patient-level glycomics and mass-spectrometry data are not distributed in this repository. The bulk RNA-seq cohort is available from GEO under accession GSE235508, and the ordered RNA feature list is provided in `data/features/rna_features_1124.csv`.

## Reproducing the reported model evaluation

Run all five master seeds, each with fivefold cross-validation:

```bash
slemodel-train
```

Aggregate classification, calibration, and reconstruction metrics:

```bash
slemodel-summarize
```

The reported reconstruction values are calculated separately in every validation fold and summarized across the 25 seed-fold evaluations. Detailed commands for the other analyses are provided in [docs/workflows.md](docs/workflows.md).

The single-cell analysis uses a separate environment because it also requires
Scanpy, edgeR and CellPhoneDB. See
[`analyses/single_cell/README.md`](analyses/single_cell/README.md).

## External resources

Some interpretation workflows require separately obtained resources: MSigDB v2025.1 gene sets, a GRCh38 Ensembl-to-HGNC mapping, KEGG pathway mappings, and the study-specific serum metabolite reference database. Their expected locations and licensing considerations are listed in [docs/workflows.md](docs/workflows.md).

## Data and code scope

The generated glycomics and mass-spectrometry data are available from the corresponding author upon reasonable request, subject to institutional approval and applicable data-use agreements.

Please cite the associated article when using this code.
