# Input data layout

SLEmodel expects one comma-separated table per modality:

```text
data/
├── gly/
│   └── glycomics_model_input.csv
├── mass/
│   └── mass_model_input.csv
└── rna/
    └── rna_model_input.csv
```

Each table has the same structure:

1. `ID`: de-identified participant identifier
2. `Group`: one of `Active`, `Stable`, or `Control`
3. remaining columns: numeric modality features

The reported inputs contained:

| Modality | Participants | Features | Active | Stable | Control |
|---|---:|---:|---:|---:|---:|
| Serum IgG glycomics | 377 | 51 | 75 | 122 | 180 |
| Serum mass spectrometry | 105 | 60 | 34 | 42 | 29 |
| Bulk RNA-seq | 194 | 1,124 | 22 | 79 | 93 |

The cohorts comprise non-overlapping participants; identifiers are used only to track samples within each modality.

The bulk RNA-seq cohort is publicly available from GEO under accession GSE235508. The locally generated glycomics and mass-spectrometry data are available from the corresponding author upon reasonable request, subject to institutional approval and applicable data-use agreements.

The reported model uses a fixed set of 1,124 Ensembl genes. Their order in the RNA input table is recorded in `features/rna_features_1124.csv`. 