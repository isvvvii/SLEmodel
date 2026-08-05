# Analysis workflows

Run commands from the repository root after installing the package.

## 1. SLEmodel training and summary

```bash
slemodel-train
slemodel-summarize
```

Training products are written beneath `experiments/`. Aggregate summaries are written to `analysis_results_foldmean/`.

## 2. Alignment diagnostics

Generate per-fold diagnostics, aggregate them:

```bash
python -m analyses.alignment.interpretability
python -m analyses.alignment.aggregate_interpretability_results \
  interpretability_results interpretability_aggregate
python -m analyses.alignment.plot_alignment_diagnostics
python -m analyses.alignment.plot_alignment_additional_diagnostics
```

The sensitivity analysis for modality-specific OT regularization is run with:

```bash
python -m analyses.alignment.run_sensitivity_analysis
```

## 3. Ablation studies

```bash
python -m analyses.ablation.ablation_runner
python -m analyses.ablation.ablation_analysis_results
python -m analyses.ablation.plot_ablation_results
```

## 4. Traditional machine-learning comparisons

```bash
python -m analyses.benchmarks.ml_classification
python -m analyses.benchmarks.ml_crossmodal_reconstruction
python -m analyses.benchmarks.train_slemodel_crossmodal_regressors
python -m analyses.benchmarks.evaluate_slemodel_crossmodal_reconstruction
```

The classification workflow evaluates the eight single-modality models reported in the Methods: LDA, logistic regression, linear SVM, decision tree, k-nearest neighbours, random forest, elastic-net logistic regression, and XGBoost. Early fusion uses elastic-net logistic regression and XGBoost; late fusion uses random-forest base learners with a logistic-regression meta-learner. The reconstruction workflow compares PLS regression and random-forest regression with cross-modal regressors trained on frozen SLEmodel representations.

## 5. Model interpretation and multi-omics analysis

```bash
python -m analyses.interpretation.shap_analysis
python -m analyses.interpretation.plot_signed_shap_summary
python -m analyses.interpretation.plot_representative_cases
```

For the biological panels, first obtain the required external resources and then run:

```bash
python -m analyses.interpretation.plot_multimodal_interpretation \
  --msigdb_h ref/msigdb/h.all.v2025.1.Hs.symbols.gmt \
  --msigdb_reactome ref/msigdb/c2.cp.reactome.v2025.1.Hs.symbols.gmt \
  --msigdb_gobp ref/msigdb/c5.go.bp.v2025.1.Hs.symbols.gmt
```

Required resources are not committed when redistribution is restricted:

- MSigDB v2025.1 Hallmark, Reactome, and GO Biological Process GMT files
- GRCh38 Ensembl-to-HGNC mapping used for RNA feature annotation
- KEGG pathway mappings obtained through the KEGG REST service
- the study-specific serum metabolite database used for exact-mass annotation

The scripts record statistical and display audit tables alongside each generated panel.

## 6. Pregnancy-associated SLE single-cell analysis

The PBMC single-cell workflow is kept separate from SLEmodel training. A
reference raw-count QC/PCA implementation is available for approved 10x-format
matrices:

```bash
python analyses/single_cell/scripts/preprocess_raw_counts_reference.py \
  --matrix-root /approved/path/to/10x_matrices \
  --metadata /approved/path/to/sample_metadata.tsv \
  --output-dir outputs/single_cell/preprocessing
```

Reproduction of the reported APO versus non-APO
statistics starts from controlled, processed AnnData objects and performs
sample-level module testing, raw-count pseudobulk differential expression,
pathway analysis and CellPhoneDB analysis:

```bash
conda env create -f analyses/single_cell/environment.yml
conda activate slemodel-single-cell
python analyses/single_cell/scripts/run_apo_analysis.py
python analyses/single_cell/scripts/refine_bcell_modules.py
```

Input layout, environment variables, visualization scripts and the limits of
the reference preprocessing code are documented in
[`analyses/single_cell/README.md`](../analyses/single_cell/README.md).
