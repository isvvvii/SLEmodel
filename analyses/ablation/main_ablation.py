# main_ablation.py

import argparse
import json
import torch
import numpy as np
import sys
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from .dataset import (StaticMatchedDataset, DynamicHybridDataset,
                      SingleModalityDataset, RandomMatchedDataset)
from .utils import (
    get_ot_coupling,
    get_ot_coupling_modality_specific,
    get_propensity_scores_calibrated,
    FocalLoss
)
from .models import DecoupledModel, PropensityScoreClassifier
from .evaluation import evaluate_and_visualize

from . import config as cfg
from slemodel.train import load_data, get_all_metrics_from_loader
from slemodel.utils import train_classifier, get_propensity_scores

logger = cfg.logging.getLogger(__name__)


def load_experiment_config(config_file):
    """Load one ablation configuration into the config module."""
    with open(config_file, 'r') as f:
        exp_config = json.load(f)

    for key, value in exp_config.items():
        setattr(cfg, key, value)

    return exp_config

def save_val_ps_ot(
    fold_dir: Path,
    ps_val_gly: torch.Tensor,
    ps_pool_mass: torch.Tensor,
    ps_pool_rna: torch.Tensor,
    G_mass: torch.Tensor,
    G_rna: torch.Tensor
):
    """Save validation propensity scores and transport plans for interpretation."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        fold_dir / "val_ps_ot.npz",
        ps_val_gly=ps_val_gly.detach().cpu().numpy(),
        ps_pool_mass=ps_pool_mass.detach().cpu().numpy(),
        ps_pool_rna=ps_pool_rna.detach().cpu().numpy(),
        G_mass=G_mass.detach().cpu().numpy(),
        G_rna=G_rna.detach().cpu().numpy(),
    )

def create_modified_dataset(
    train_gly_x, train_gly_y, val_gly_x, val_gly_y,
    pool_mass_x, pool_rna_x,
    ps_train_gly, ps_val_gly, ps_pool_mass, ps_pool_rna,
    gly_train_idx, gly_val_idx, gly_ids_raw, config,
    gly_x_raw, mass_x_raw, rna_x_raw,
    mass_train_idx, rna_train_idx,
    mass_scaler, rna_scaler, mass_ids_raw, rna_ids_raw,
    mass_splits, rna_splits, global_fold_idx,
    mass_y_encoded, rna_y_encoded,
    fold_dir: Path = None,
):
    """Construct the training and validation datasets for one ablation."""

    single_modality = config.get('SINGLE_MODALITY_MODE', None)
    use_random_matching = config.get('USE_RANDOM_MATCHING', False)
    dynamic_resample = config.get('DYNAMIC_OT_RESAMPLE', True)

    all_dims = {
        'gly': gly_x_raw.shape[1],
        'mass': mass_x_raw.shape[1],
        'rna': rna_x_raw.shape[1]
    }

    _, mass_val_idx = mass_splits[global_fold_idx]
    _, rna_val_idx = rna_splits[global_fold_idx]

    # -------------------------
    # -------------------------
    if single_modality:
        logger.info(f"Creating dataset for SINGLE MODALITY mode: {single_modality.upper()}")
        if single_modality == 'gly':
            train_dataset = SingleModalityDataset(train_gly_x, train_gly_y, gly_ids_raw[gly_train_idx], 'gly', all_dims)
            val_dataset = SingleModalityDataset(val_gly_x, val_gly_y, gly_ids_raw[gly_val_idx], 'gly', all_dims)

        elif single_modality == 'mass':
            mass_x_train_scaled = mass_scaler.transform(mass_x_raw[mass_train_idx])
            mass_x_val_scaled = mass_scaler.transform(mass_x_raw[mass_val_idx])
            train_mass_x = torch.tensor(mass_x_train_scaled, dtype=torch.float32)
            val_mass_x = torch.tensor(mass_x_val_scaled, dtype=torch.float32)

            train_dataset = SingleModalityDataset(
                train_mass_x, torch.tensor(mass_y_encoded[mass_train_idx], dtype=torch.long),
                mass_ids_raw[mass_train_idx], 'mass', all_dims
            )
            val_dataset = SingleModalityDataset(
                val_mass_x, torch.tensor(mass_y_encoded[mass_val_idx], dtype=torch.long),
                mass_ids_raw[mass_val_idx], 'mass', all_dims
            )

        elif single_modality == 'rna':
            rna_x_train_scaled = rna_scaler.transform(rna_x_raw[rna_train_idx])
            rna_x_val_scaled = rna_scaler.transform(rna_x_raw[rna_val_idx])
            train_rna_x = torch.tensor(rna_x_train_scaled, dtype=torch.float32)
            val_rna_x = torch.tensor(rna_x_val_scaled, dtype=torch.float32)

            train_dataset = SingleModalityDataset(
                train_rna_x, torch.tensor(rna_y_encoded[rna_train_idx], dtype=torch.long),
                rna_ids_raw[rna_train_idx], 'rna', all_dims
            )
            val_dataset = SingleModalityDataset(
                val_rna_x, torch.tensor(rna_y_encoded[rna_val_idx], dtype=torch.long),
                rna_ids_raw[rna_val_idx], 'rna', all_dims
            )

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        return train_loader, val_loader

    # -------------------------
    # -------------------------
    if use_random_matching:
        logger.info("Creating dataset for RANDOM MATCHING mode.")
        mass_x_train_scaled = mass_scaler.transform(mass_x_raw[mass_train_idx])
        rna_x_train_scaled = rna_scaler.transform(rna_x_raw[rna_train_idx])
        mass_x_val_scaled = mass_scaler.transform(mass_x_raw[mass_val_idx])
        rna_x_val_scaled = rna_scaler.transform(rna_x_raw[rna_val_idx])

        train_mass_x = torch.tensor(mass_x_train_scaled, dtype=torch.float32)
        train_rna_x = torch.tensor(rna_x_train_scaled, dtype=torch.float32)
        val_mass_x = torch.tensor(mass_x_val_scaled, dtype=torch.float32)
        val_rna_x = torch.tensor(rna_x_val_scaled, dtype=torch.float32)

        train_dataset = RandomMatchedDataset(
            gly_data=train_gly_x, gly_labels=train_gly_y, mass_data=train_mass_x, rna_data=train_rna_x,
            gly_ids=gly_ids_raw[gly_train_idx], mass_ids=mass_ids_raw[mass_train_idx], rna_ids=rna_ids_raw[rna_train_idx],
            seed=config['SEED'], split_type="train"
        )
        val_dataset = RandomMatchedDataset(
            gly_data=val_gly_x, gly_labels=val_gly_y, mass_data=val_mass_x, rna_data=val_rna_x,
            gly_ids=gly_ids_raw[gly_val_idx], mass_ids=mass_ids_raw[mass_val_idx], rna_ids=rna_ids_raw[rna_val_idx],
            seed=config['SEED'] + 1000, split_type="validation"
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        return train_loader, val_loader

    # -------------------------
    # -------------------------
    logger.info("Creating dataset for OT MATCHING mode (modality-specific OT config).")

    reg_mass = config.get('OT_REG_MASS', getattr(cfg, 'OT_REG_MASS', 1e-3))
    reg_rna  = config.get('OT_REG_RNA',  getattr(cfg, 'OT_REG_RNA',  1e-3))
    method_mass = config.get('OT_METHOD_MASS', getattr(cfg, 'OT_METHOD_MASS', 'standard'))
    method_rna  = config.get('OT_METHOD_RNA',  getattr(cfg, 'OT_METHOD_RNA',  'standard'))
    cost_metric_mass = config.get('OT_COST_METRIC_MASS', getattr(cfg, 'OT_COST_METRIC_MASS', 'sqeuclidean'))
    cost_metric_rna  = config.get('OT_COST_METRIC_RNA',  getattr(cfg, 'OT_COST_METRIC_RNA',  'sqeuclidean'))

    dynamic_train_dataset = DynamicHybridDataset(
        gly_data=train_gly_x,
        gly_labels=train_gly_y,
        all_mass_data=pool_mass_x,
        all_rna_data=pool_rna_x,
        ps_gly=ps_train_gly,
        ps_mass=ps_pool_mass,
        ps_rna=ps_pool_rna,
        dynamic_resample=dynamic_resample,

        reg_mass=reg_mass,
        reg_rna=reg_rna,
        ot_method_mass=method_mass,
        ot_method_rna=method_rna,
        ot_cost_metric_mass=cost_metric_mass,
        ot_cost_metric_rna=cost_metric_rna,

        resample_every=5,
        sinkhorn_numItermax_mass=400,
        sinkhorn_stopThr_mass=1e-3,
        sinkhorn_numItermax_rna=600,
        sinkhorn_stopThr_rna=1e-3,
    )
    train_loader = DataLoader(dynamic_train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    val_mass_coupling = get_ot_coupling_modality_specific(
        ps_val_gly, ps_pool_mass, modality='mass',
        reg_mass=reg_mass, reg_rna=reg_rna,
        method_mass=method_mass, method_rna=method_rna,
        cost_metric_mass=cost_metric_mass, cost_metric_rna=cost_metric_rna,
        purpose="eval",
    )
    val_rna_coupling = get_ot_coupling_modality_specific(
        ps_val_gly, ps_pool_rna, modality='rna',
        reg_mass=reg_mass, reg_rna=reg_rna,
        method_mass=method_mass, method_rna=method_rna,
        cost_metric_mass=cost_metric_mass, cost_metric_rna=cost_metric_rna,
        purpose="eval",
    )

    if fold_dir is not None:
        save_val_ps_ot(fold_dir, ps_val_gly, ps_pool_mass, ps_pool_rna, val_mass_coupling, val_rna_coupling)

    val_dataset = StaticMatchedDataset(
        val_gly_x, val_gly_y,
        pool_mass_x, pool_rna_x,
        val_mass_coupling, val_rna_coupling,
        gly_ids_raw[gly_val_idx]
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader

def create_modified_model(gly_dim: int, mass_dim: int, rna_dim: int, config: dict):
    """Construct a model from an ablation configuration."""
    latent_dim = int(config.get("LATENT_DIM", cfg.LATENT_DIM))
    dropout_rate = float(config.get("DROPOUT_RATE", cfg.DROPOUT_RATE))

    use_cross_attention = bool(config.get("USE_CROSS_ATTENTION", True))
    use_decoupling = bool(config.get("USE_DECOUPLING", True))

    fusion_method = config.get("FUSION_METHOD", "attention")
    attention_num_heads = int(config.get("ATTENTION_NUM_HEADS", 1))

    single_modality_mode = config.get("SINGLE_MODALITY_MODE", None)
    if single_modality_mode is not None:
        single_modality_mode = str(single_modality_mode).lower().strip()
        if single_modality_mode not in ("gly", "mass", "rna"):
            raise ValueError(f"Invalid SINGLE_MODALITY_MODE: {single_modality_mode}")

    if use_cross_attention and fusion_method == "attention":
        if latent_dim % attention_num_heads != 0:
            raise ValueError(
                f"Invalid attention heads: latent_dim={latent_dim} not divisible by heads={attention_num_heads}"
            )

    model = DecoupledModel(
        gly_dim=gly_dim,
        mass_dim=mass_dim,
        rna_dim=rna_dim,
        latent_dim=latent_dim,
        use_cross_attention=use_cross_attention,
        use_decoupling=use_decoupling,
        attention_num_heads=attention_num_heads,
        fusion_method=fusion_method,
        dropout_rate=dropout_rate,
        single_modality_mode=single_modality_mode,
    )
    return model


def modified_main(exp_id):
    """Run one ablation experiment."""
    run_dir, log_dir = cfg.setup_directories(cfg.SEED, exp_id)
    global logger
    logger = cfg.setup_logging(log_dir)
    cfg.VisualizationConfig.apply_style()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    logger.info(f"\n{'='*20} Ablation Experiment {'='*20}")
    logger.info(f"Experiment ID: {exp_id}")
    logger.info(f"Experiment Name: {cfg.CURRENT_RUN_NAME}")
    logger.info(f"Seed: {cfg.SEED}")

    logger.info("--- Effective Experiment Configuration ---")
    keys_to_log = [
        'USE_CROSS_ATTENTION', 'USE_DECOUPLING', 'DYNAMIC_OT_RESAMPLE',
        'USE_RANDOM_MATCHING', 'USE_OT_MATCHING', 'SINGLE_MODALITY_MODE',
        'ATTENTION_NUM_HEADS', 'FUSION_METHOD', 'LATENT_DIM', 'DROPOUT_RATE',
        'RECON_LOSS_WEIGHT', 'SEMI_SUPERVISED_WEIGHT', 'ORTHO_LOSS_WEIGHT'
    ]
    current_config = {key: getattr(cfg, key) for key in keys_to_log + ['SEED']}
    for key in keys_to_log:
        logger.info(f"{key:>28}: {current_config.get(key, 'Not Set')}")
    logger.info(f"{'='*60}")

    logger.info("--- Loading model-ready data ---")
    gly_ids_raw, gly_x_raw, gly_y_str_raw = load_data(cfg.GLY_PATH)
    mass_ids_raw, mass_x_raw, mass_y_str_raw = load_data(cfg.MASS_PATH)
    rna_ids_raw, rna_x_raw, rna_y_str_raw = load_data(
        cfg.RNA_PATH, cfg.RNA_FEATURE_LIST_PATH
    )

    label_map = {v: k for k, v in cfg.VisualizationConfig.LABEL_NAMES.items()}
    reverse_label_map = cfg.VisualizationConfig.LABEL_NAMES
    gly_y_encoded = pd.Series(gly_y_str_raw).map(label_map).values
    mass_y_encoded = pd.Series(mass_y_str_raw).map(label_map).values
    rna_y_encoded = pd.Series(rna_y_str_raw).map(label_map).values

    # Match the reported main-model runs: use fixed class weights derived from
    # the complete glycomics anchor cohort.
    class_counts = np.bincount(gly_y_encoded, minlength=cfg.NUM_CLASSES)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(cfg.DEVICE)

    rskf_gly = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED)
    rskf_mass = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED)
    rskf_rna = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS_K_FOLD, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED)

    gly_splits = list(rskf_gly.split(gly_x_raw, gly_y_encoded))
    mass_splits = list(rskf_mass.split(mass_x_raw, mass_y_encoded))
    rna_splits = list(rskf_rna.split(rna_x_raw, rna_y_encoded))

    all_runs_cls_metrics = []
    all_runs_recon_metrics = {mod: [] for mod in ['gly', 'mass', 'rna']}
    all_runs_report_dfs = []

    def validate_no_data_leakage_by_index(train_idx, val_idx, modality_name):
        """Check that training and validation indices do not overlap."""
        train_set = set(train_idx)
        val_set = set(val_idx)
        overlap = train_set.intersection(val_set)
        if overlap:
            logger.error(f"Data leakage detected in {modality_name}! Overlapping indices: {overlap}")
            raise ValueError(f"Data leakage in {modality_name} modality")

    for repeat in range(cfg.N_REPEATS):
        logger.info(f"\n{'='*25} Starting Repeat {repeat+1}/{cfg.N_REPEATS} {'='*25}")
        recon_data_dir = run_dir / "reconstruction_data"
        recon_data_dir.mkdir(exist_ok=True, parents=True)

        mode = current_config.get('SINGLE_MODALITY_MODE')
        if mode == 'mass':
            n_samples = len(mass_y_encoded)
            logger.info(f"Using Mass modality samples for OOF: {n_samples}")
        elif mode == 'rna':
            n_samples = len(rna_y_encoded)
            logger.info(f"Using RNA modality samples for OOF: {n_samples}")
        else:
            n_samples = len(gly_y_encoded)
            logger.info(f"Using Gly modality samples for OOF: {n_samples}")

        oof_val_true_labels = np.zeros(n_samples, dtype=int)
        oof_val_pred_scores = np.zeros((n_samples, cfg.NUM_CLASSES), dtype=float)
        oof_train_true_labels = np.zeros(n_samples, dtype=int)
        oof_train_pred_scores = np.zeros((n_samples, cfg.NUM_CLASSES), dtype=float)
        repeat_misclassified_samples = []

        for fold in range(cfg.N_SPLITS_K_FOLD):
            global_fold_idx = repeat * cfg.N_SPLITS_K_FOLD + fold
            fold_name = f"Repeat_{repeat+1}-Fold_{fold+1}"
            fold_dir = run_dir / fold_name
            fold_dir.mkdir(exist_ok=True)
            logger.info(f"\n--- {fold_name} ---")

            gly_train_idx, gly_val_idx = gly_splits[global_fold_idx]
            mass_train_idx, mass_val_idx = mass_splits[global_fold_idx]
            rna_train_idx, rna_val_idx = rna_splits[global_fold_idx]

            validate_no_data_leakage_by_index(mass_train_idx, mass_val_idx, "Mass")
            validate_no_data_leakage_by_index(rna_train_idx, rna_val_idx, "RNA")

            gly_scaler = StandardScaler().fit(gly_x_raw[gly_train_idx])
            gly_x_train_scaled = gly_scaler.transform(gly_x_raw[gly_train_idx])
            gly_x_val_scaled = gly_scaler.transform(gly_x_raw[gly_val_idx])

            mass_scaler = StandardScaler().fit(mass_x_raw[mass_train_idx])
            mass_x_train_scaled = mass_scaler.transform(mass_x_raw[mass_train_idx])

            rna_scaler = StandardScaler().fit(rna_x_raw[rna_train_idx])
            rna_x_train_scaled = rna_scaler.transform(rna_x_raw[rna_train_idx])

            train_gly_x = torch.tensor(gly_x_train_scaled, dtype=torch.float32)
            train_gly_y = torch.tensor(gly_y_encoded[gly_train_idx], dtype=torch.long)
            val_gly_x = torch.tensor(gly_x_val_scaled, dtype=torch.float32)
            val_gly_y = torch.tensor(gly_y_encoded[gly_val_idx], dtype=torch.long)

            pool_mass_x = torch.tensor(mass_x_train_scaled, dtype=torch.float32)
            pool_mass_y = torch.tensor(mass_y_encoded[mass_train_idx], dtype=torch.long)
            pool_rna_x = torch.tensor(rna_x_train_scaled, dtype=torch.float32)
            pool_rna_y = torch.tensor(rna_y_encoded[rna_train_idx], dtype=torch.long)

            ps_train_gly = ps_val_gly = ps_pool_mass = ps_pool_rna = None

            if not current_config.get('SINGLE_MODALITY_MODE') and not current_config.get('USE_RANDOM_MATCHING'):
                logger.info("Training Propensity Score classifiers...")

                gly_classifier = train_classifier(
                    PropensityScoreClassifier(train_gly_x.shape[1]),
                    train_gly_x, train_gly_y, cfg.DEVICE, epochs=cfg.EPOCHS_PS
                )
                mass_classifier = train_classifier(
                    PropensityScoreClassifier(pool_mass_x.shape[1]),
                    pool_mass_x, pool_mass_y, cfg.DEVICE, epochs=cfg.EPOCHS_PS
                )
                rna_classifier = train_classifier(
                    PropensityScoreClassifier(pool_rna_x.shape[1]),
                    pool_rna_x, pool_rna_y, cfg.DEVICE, epochs=cfg.EPOCHS_PS
                )

                if getattr(cfg, "USE_PS_TEMPERATURE_SCALING", False):
                    logger.info("Applying temperature scaling calibration to PS...")

                    ps_train_gly, temp_gly_train = get_propensity_scores_calibrated(
                        gly_classifier, train_gly_x, train_gly_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                    ps_val_gly, temp_gly_val = get_propensity_scores_calibrated(
                        gly_classifier, val_gly_x, val_gly_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                    ps_pool_mass, temp_mass = get_propensity_scores_calibrated(
                        mass_classifier, pool_mass_x, pool_mass_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )
                    ps_pool_rna, temp_rna = get_propensity_scores_calibrated(
                        rna_classifier, pool_rna_x, pool_rna_y, cfg.DEVICE,
                        calibrate=True,
                        max_iter=getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100),
                        lr=getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01),
                    )

                    logger.info(
                        f"Calibration temperatures: Gly(train)={temp_gly_train:.3f}, "
                        f"Gly(val)={temp_gly_val:.3f}, Mass={temp_mass:.3f}, RNA={temp_rna:.3f}"
                    )
                else:
                    ps_train_gly = get_propensity_scores(gly_classifier, train_gly_x, cfg.DEVICE)
                    ps_val_gly = get_propensity_scores(gly_classifier, val_gly_x, cfg.DEVICE)
                    ps_pool_mass = get_propensity_scores(mass_classifier, pool_mass_x, cfg.DEVICE)
                    ps_pool_rna = get_propensity_scores(rna_classifier, pool_rna_x, cfg.DEVICE)
            else:
                logger.info("Skipping Propensity Score calculation for this ablation mode.")

            train_loader, val_loader = create_modified_dataset(
                train_gly_x, train_gly_y, val_gly_x, val_gly_y,
                pool_mass_x, pool_rna_x,
                ps_train_gly, ps_val_gly, ps_pool_mass, ps_pool_rna,
                gly_train_idx, gly_val_idx, gly_ids_raw, current_config,
                gly_x_raw, mass_x_raw, rna_x_raw,
                mass_train_idx, rna_train_idx,
                mass_scaler, rna_scaler, mass_ids_raw, rna_ids_raw,
                mass_splits, rna_splits, global_fold_idx,
                mass_y_encoded, rna_y_encoded, fold_dir=fold_dir
            )

            model = create_modified_model(
                gly_x_raw.shape[1], mass_x_raw.shape[1], rna_x_raw.shape[1], current_config
            ).to(cfg.DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=cfg.LR_DOWNSTREAM, weight_decay=cfg.WEIGHT_DECAY_DOWNSTREAM)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=False, min_lr=1e-6)
            cls_criterion = FocalLoss(alpha=class_weights)
            recon_criterion = nn.MSELoss()

            history = {'train_loss': [], 'train_class_loss': [], 'train_recon_loss': [], 'train_semi_loss': [], 'train_ortho_loss': [], 'val_class_loss': [], 'val_auc': []}
            if cfg.USE_EARLY_STOPPING:
                patience_counter = 0; best_score = np.Inf; best_epoch = -1
                best_model_path = fold_dir / f"best_model_{fold_name}.pth"

            for epoch in range(cfg.EPOCHS_DOWNSTREAM):
                model.train()
                epoch_loss, epoch_cls, epoch_recon_paired, epoch_recon_semi, epoch_ortho = 0, 0, 0, 0, 0

                if (not current_config.get('USE_RANDOM_MATCHING', False) and
                    hasattr(train_loader.dataset, 'resample_and_match')):
                    train_loader.dataset.resample_and_match()

                for batch in train_loader:
                    optimizer.zero_grad()
                    gly_x, mass_x, rna_x, labels = batch['gly_x'].to(cfg.DEVICE), batch['mass_x'].to(cfg.DEVICE), batch['rna_x'].to(cfg.DEVICE), batch['label'].to(cfg.DEVICE)
                    logits, recon_matched, reps = model(gly_x, mass_x, rna_x)

                    loss_cls = cls_criterion(logits, labels)

                    loss_recon = torch.tensor(0.0, device=cfg.DEVICE)
                    loss_semi = torch.tensor(0.0, device=cfg.DEVICE)
                    loss_ortho = torch.tensor(0.0, device=cfg.DEVICE)

                    if cfg.RECON_LOSS_WEIGHT > 0:
                        active_mode = current_config.get('SINGLE_MODALITY_MODE')
                        if active_mode:
                            if active_mode == 'gly': loss_recon = recon_criterion(recon_matched['gly'], gly_x)
                            elif active_mode == 'mass': loss_recon = recon_criterion(recon_matched['mass'], mass_x)
                            elif active_mode == 'rna': loss_recon = recon_criterion(recon_matched['rna'], rna_x)
                        else:
                            loss_recon = (recon_criterion(recon_matched['gly'], gly_x) +
                                          recon_criterion(recon_matched['mass'], mass_x) +
                                          recon_criterion(recon_matched['rna'], rna_x))

                    if cfg.SEMI_SUPERVISED_WEIGHT > 0 and 'real_gly' in batch:
                        real_gly, real_mass, real_rna = batch['real_gly'].to(cfg.DEVICE), batch['real_mass'].to(cfg.DEVICE), batch['real_rna'].to(cfg.DEVICE)
                        recon_gly_real, _, _ = model.reconstruct_single_modality(real_gly, 'gly')
                        recon_mass_real, _, _ = model.reconstruct_single_modality(real_mass, 'mass')
                        recon_rna_real, _, _ = model.reconstruct_single_modality(real_rna, 'rna')
                        loss_semi = recon_criterion(recon_gly_real, real_gly) + recon_criterion(recon_mass_real, real_mass) + recon_criterion(recon_rna_real, real_rna)

                    if cfg.ORTHO_LOSS_WEIGHT > 0 and current_config.get('USE_DECOUPLING', True):
                        active_mode = current_config.get('SINGLE_MODALITY_MODE')
                        if active_mode:
                            loss_ortho = torch.abs(F.cosine_similarity(reps['shared'][active_mode], reps['private'][active_mode], dim=1)).mean()
                        else:
                            loss_ortho = sum(torch.abs(F.cosine_similarity(reps['shared'][m], reps['private'][m], dim=1)).mean() for m in ['gly', 'mass', 'rna'])

                    total_loss = loss_cls + cfg.RECON_LOSS_WEIGHT * loss_recon + cfg.SEMI_SUPERVISED_WEIGHT * loss_semi + cfg.ORTHO_LOSS_WEIGHT * loss_ortho

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += total_loss.item()
                    epoch_cls += loss_cls.item()
                    epoch_recon_paired += loss_recon.item()
                    epoch_recon_semi += loss_semi.item()
                    epoch_ortho += loss_ortho.item()

                num_batches = len(train_loader)
                history['train_loss'].append(epoch_loss / num_batches)
                history['train_class_loss'].append(epoch_cls / num_batches)
                history['train_recon_loss'].append(epoch_recon_paired / num_batches)
                history['train_semi_loss'].append(epoch_recon_semi / num_batches)
                history['train_ortho_loss'].append(epoch_ortho / num_batches)

                val_true_labels, val_pred_scores, val_pred_logits, _, _, _ = get_all_metrics_from_loader(model, val_loader, cfg.DEVICE)
                val_metrics, _, _ = evaluate_and_visualize({}, val_true_labels, val_pred_scores, val_pred_logits, {}, run_dir, fold_name, "EpochValidation", save_plots=False, print_report=False, class_weights=class_weights.cpu())
                avg_val_loss = val_metrics.get('loss', float('inf'))
                history['val_class_loss'].append(avg_val_loss)
                scheduler.step(avg_val_loss)

                if cfg.USE_EARLY_STOPPING and epoch >= cfg.EARLY_STOPPING_MIN_EPOCHS:
                    if avg_val_loss < best_score:
                        best_score, patience_counter, best_epoch = avg_val_loss, 0, epoch + 1
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                    if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                        logger.warning(f"Early stopping at epoch {epoch + 1}.")
                        break

            if cfg.USE_EARLY_STOPPING and Path(best_model_path).exists():
                logger.info(f"Loading best model from epoch {best_epoch} for final evaluation.")
                model.load_state_dict(torch.load(best_model_path))
            else:
                logger.warning("No best model found or early stopping disabled. Using final model.")

            val_true, val_scores, val_logits, val_recon, val_ids, val_orig_indices = get_all_metrics_from_loader(model, val_loader, cfg.DEVICE)

            if len(val_true) > 0:
                active_mode = current_config.get('SINGLE_MODALITY_MODE')
                if active_mode == 'mass':
                    oof_val_true_labels[mass_val_idx] = val_true
                    oof_val_pred_scores[mass_val_idx] = val_scores
                elif active_mode == 'rna':
                    oof_val_true_labels[rna_val_idx] = val_true
                    oof_val_pred_scores[rna_val_idx] = val_scores
                else: # gly-based or multimodal
                    oof_val_true_labels[gly_val_idx] = val_true
                    oof_val_pred_scores[gly_val_idx] = val_scores

            val_cls, val_rec_met, val_report_dict = evaluate_and_visualize(
                history, val_true, val_scores, val_logits, val_recon, run_dir, fold_name,
                "Validation (Best Model)", class_weights=class_weights.cpu(),
                single_modality_mode=current_config.get('SINGLE_MODALITY_MODE')
            )

            if len(val_true) > 0:
                val_pred = np.argmax(val_scores, axis=1)
                for i in np.where(val_true != val_pred)[0]:
                    active_mode = current_config.get('SINGLE_MODALITY_MODE')
                    if active_mode == 'mass':
                        original_index = mass_val_idx[val_orig_indices[i]]
                    elif active_mode == 'rna':
                        original_index = rna_val_idx[val_orig_indices[i]]
                    else: # gly-based or multimodal
                        original_index = gly_val_idx[val_orig_indices[i]]

                    sample_id = val_ids[i]
                    repeat_misclassified_samples.append({
                        'original_index': original_index, 'id': sample_id, 'fold': fold_name,
                        'true_group': reverse_label_map[val_true[i]],
                        'predicted_group': reverse_label_map[val_pred[i]]
                    })

            np.savez(recon_data_dir / f"{fold_name}_validation.npz", **{f"{mod}_true": data['true'] for mod, data in val_recon.items()}, **{f"{mod}_pred": data['pred'] for mod, data in val_recon.items()})
            all_runs_cls_metrics.append({'set': 'validation', 'fold': fold_name, **val_cls})
            for mod, metrics in val_rec_met.items(): all_runs_recon_metrics[mod].append({'set': 'validation', 'fold': fold_name, **metrics})
            if val_report_dict:
                val_report_df = pd.DataFrame(val_report_dict).T
                val_report_df['fold'] = fold_name; val_report_df['set'] = 'validation'
                all_runs_report_dfs.append(val_report_df)

            train_static_loader = None
            if current_config.get('SINGLE_MODALITY_MODE') or current_config.get('USE_RANDOM_MATCHING'):
                train_static_loader = train_loader
            elif not current_config.get('SINGLE_MODALITY_MODE') and not current_config.get('USE_RANDOM_MATCHING'):
                train_mass_coupling = get_ot_coupling_modality_specific(
                    ps_train_gly, ps_pool_mass, modality='mass',
                    reg_mass=getattr(cfg, "OT_REG_MASS", 1e-3),
                    reg_rna=getattr(cfg, "OT_REG_RNA", 1e-3),
                    method_mass=getattr(cfg, "OT_METHOD_MASS", "standard"),
                    method_rna=getattr(cfg, "OT_METHOD_RNA", "standard"),
                    cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                    cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
                    purpose="eval",
                )

                train_rna_coupling = get_ot_coupling_modality_specific(
                    ps_train_gly, ps_pool_rna, modality='rna',
                    reg_mass=getattr(cfg, "OT_REG_MASS", 1e-3),
                    reg_rna=getattr(cfg, "OT_REG_RNA", 1e-3),
                    method_mass=getattr(cfg, "OT_METHOD_MASS", "standard"),
                    method_rna=getattr(cfg, "OT_METHOD_RNA", "standard"),
                    cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                    cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
                    purpose="eval",
                )
                train_static_dataset = StaticMatchedDataset(train_gly_x, train_gly_y, pool_mass_x, pool_rna_x, train_mass_coupling, train_rna_coupling, gly_ids_raw[gly_train_idx])
                train_static_loader = DataLoader(train_static_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

            if train_static_loader:
                train_true, train_scores, train_logits, train_recon, _, _ = get_all_metrics_from_loader(model, train_static_loader, cfg.DEVICE)

                if len(train_true) > 0:
                    active_mode = current_config.get('SINGLE_MODALITY_MODE')
                    if active_mode == 'mass':
                        oof_train_true_labels[mass_train_idx] = train_true
                        oof_train_pred_scores[mass_train_idx] = train_scores
                    elif active_mode == 'rna':
                        oof_train_true_labels[rna_train_idx] = train_true
                        oof_train_pred_scores[rna_train_idx] = train_scores
                    else:
                        oof_train_true_labels[gly_train_idx] = train_true
                        oof_train_pred_scores[gly_train_idx] = train_scores

                train_cls, train_rec_met, train_report_dict = evaluate_and_visualize(
                    {}, train_true, train_scores, train_logits, train_recon, run_dir, fold_name,
                    "Training (Best Model)", class_weights=class_weights.cpu(),
                    single_modality_mode=current_config.get('SINGLE_MODALITY_MODE')
                )
                np.savez(recon_data_dir / f"{fold_name}_training.npz", **{f"{mod}_true": data['true'] for mod, data in train_recon.items()}, **{f"{mod}_pred": data['pred'] for mod, data in train_recon.items()})
                all_runs_cls_metrics.append({'set': 'training', 'fold': fold_name, **train_cls})
                for mod, metrics in train_rec_met.items(): all_runs_recon_metrics[mod].append({'set': 'training', 'fold': fold_name, **metrics})
                if train_report_dict:
                    train_report_df = pd.DataFrame(train_report_dict).T
                    train_report_df['fold'] = fold_name; train_report_df['set'] = 'training'
                    all_runs_report_dfs.append(train_report_df)


        if repeat_misclassified_samples:
            misclassified_dir = run_dir / "misclassified_reports"
            misclassified_dir.mkdir(exist_ok=True, parents=True)
            pd.DataFrame(repeat_misclassified_samples).to_csv(misclassified_dir / f"misclassified_samples_repeat_{repeat+1}.csv", index=False)

        oof_dir = run_dir / "oof_predictions"
        oof_dir.mkdir(exist_ok=True, parents=True)

        np.savez(oof_dir / f"oof_val_predictions_repeat_{repeat+1}.npz",
                 true_labels=oof_val_true_labels, pred_scores=oof_val_pred_scores)
        np.savez(oof_dir / f"oof_train_predictions_repeat_{repeat+1}.npz",
                 true_labels=oof_train_true_labels, pred_scores=oof_train_pred_scores)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    cls_summary_df = pd.DataFrame(all_runs_cls_metrics)
    cls_summary_df.to_csv(metrics_dir / "classification_metrics_per_fold.csv", index=False)
    if all_runs_report_dfs:
        all_reports_df = pd.concat(all_runs_report_dfs)
        all_reports_df.index.name = 'class'
        all_reports_df.reset_index(inplace=True)
        all_reports_df.to_csv(metrics_dir / "classification_reports_per_fold_detailed.csv", index=False)
    for mod_name in ['gly', 'mass', 'rna']:
        if all_runs_recon_metrics[mod_name]:
            recon_summary_df = pd.DataFrame(all_runs_recon_metrics[mod_name])
            recon_summary_df.to_csv(metrics_dir / f"reconstruction_metrics_{mod_name}_per_fold.csv", index=False)

    logger.info("Experiment completed successfully.")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Runner')
    parser.add_argument('--config', required=True, help='Path to experiment config JSON file')
    parser.add_argument('--exp_id', required=True, help='Experiment ID')
    parser.add_argument('--seed', required=True, type=int, help='Random seed')

    args = parser.parse_args()

    load_experiment_config(args.config)

    modified_main(args.exp_id)

if __name__ == '__main__':
    main()
