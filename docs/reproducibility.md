# Reproducibility record

## Reported configuration

- master seeds: 42, 100, 2025, 7, 123
- cross-validation: five stratified folds for each master seed
- batch size: 32
- propensity-score classifier training: 100 epochs
- downstream maximum: 500 epochs
- optimizer: Adam
- learning rate: 1 × 10⁻⁴
- weight decay: 5 × 10⁻³
- latent dimension: 64
- attention heads: 4
- dropout: 0.6
- reconstruction-loss weight: 0.8
- semi-reconstruction-loss weight: 0.6
- shared-private orthogonality-loss weight: 0.1
- early stopping: minimum 25 epochs, patience 40
- mass OT regularization: 2 × 10⁻⁴
- RNA OT regularization: 2 × 10⁻³
- mass OT cost: squared Euclidean distance between square-root-transformed propensity vectors
- RNA OT cost: squared Euclidean distance between propensity vectors

Standardization, propensity-score classifiers, and OT couplings are fitted within the training partition. Validation glycomics anchors are coupled only to the mass and RNA training pools.

The reported runs used fixed focal-loss class weights derived once from the complete glycomics anchor cohort. The same weights were applied to every cross-validation fold.

## Result aggregation

Classification means and standard deviations use the 25 seed-fold evaluations. Reconstruction R², MAE, RMSE, and Pearson correlation are calculated within each validation fold before aggregation. `slemodel.summarize` implements this fold-wise procedure.

## Model used for interpretation

The model-of-record is selected in two steps:

1. choose the master seed with the median mean validation macro-AUROC;
2. within that seed, choose the fold whose macro-AUROC is closest to the seed mean.

For the reported runs, this procedure selected seed 123, fold 5. SHAP used all 75 pseudo-paired samples in that validation loader as the background distribution and 200 samples for the expected-gradients approximation.

## Software and hardware

The reported model runs used an Ubuntu 22.04 image with Python 3.10, PyTorch
2.5.1 and CUDA 12.1. The compute node contained one NVIDIA RTX 4090 GPU
(24 GB) and 16 Intel Xeon Gold 6430 vCPUs. `environment.yml` records the core
software versions used by the released workflow, including NumPy 1.26.4,
pandas 2.2.3, scikit-learn 1.5.2 and SciPy 1.14.1.
