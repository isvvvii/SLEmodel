import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from .dataset import StaticMatchedDataset, DynamicHybridDataset
from .utils import (
    get_ot_coupling_modality_specific,
    FocalLoss,
    train_classifier,
    get_propensity_scores,
    get_propensity_scores_calibrated,
)
from .models import DecoupledModel, PropensityScoreClassifier
from .evaluation import evaluate_and_visualize
from . import config as cfg

logger = cfg.logging.getLogger(__name__)

def load_data(path, feature_list_path=None):
    """Load identifiers, numeric features and labels from a modality table."""
    df = pd.read_csv(path)
    if feature_list_path is not None:
        feature_table = pd.read_csv(feature_list_path)
        expected = feature_table["ensembl_gene_id"].astype(str).tolist()
        observed = df.columns[2:].astype(str).tolist()
        if observed != expected:
            raise ValueError(
                "RNA columns do not match data/features/rna_features_1124.csv."
            )
    ids = df.iloc[:, 0].values
    labels_str = df.iloc[:, 1].values
    features = df.iloc[:, 2:].values
    return ids, features, labels_str

def save_val_ps_ot(fold_dir: Path,
                   ps_val_gly: torch.Tensor,
                   ps_pool_mass: torch.Tensor,
                   ps_pool_rna: torch.Tensor,
                   G_mass: torch.Tensor,
                   G_rna: torch.Tensor):
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

@torch.no_grad()
def get_all_metrics_from_loader(model, loader, device):
    """Collect predictions and reconstructions from a data loader."""
    model.eval()
    true_labels, pred_scores, pred_logits, ids, original_indices = [], [], [], [], []
    recon_data = {mod: {'true': [], 'pred': []} for mod in ['gly', 'mass', 'rna']}

    if len(loader) == 0:
        raise ValueError("The evaluation data loader is empty.")

    for batch in loader:
        gly_x = batch['gly_x'].to(device)
        mass_x = batch['mass_x'].to(device)
        rna_x = batch['rna_x'].to(device)
        labels = batch['label']

        logits, recon, _ = model(gly_x, mass_x, rna_x)

        if not torch.isfinite(logits).all():
            raise FloatingPointError("The model produced non-finite logits.")
        scores = torch.softmax(logits, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_scores.extend(scores.cpu().numpy())
        pred_logits.extend(logits.cpu().numpy())

        if 'id' in batch:
            ids.extend(batch['id'])
        if 'original_idx' in batch:
            original_indices.extend(batch['original_idx'].cpu().numpy())

        for mod_name in recon_data.keys():
            if f'{mod_name}_x' in batch:
                recon_data[mod_name]['true'].append(batch[f'{mod_name}_x'].cpu().numpy())
                recon_data[mod_name]['pred'].append(recon[mod_name].cpu().numpy())

    for mod_name in recon_data.keys():
        if recon_data[mod_name]['true']:
            recon_data[mod_name]['true'] = np.concatenate(recon_data[mod_name]['true'])
            recon_data[mod_name]['pred'] = np.concatenate(recon_data[mod_name]['pred'])
        else:
            recon_data[mod_name]['true'] = np.array([])
            recon_data[mod_name]['pred'] = np.array([])

    return (np.array(true_labels), np.array(pred_scores), np.array(pred_logits),
            recon_data, np.array(ids), np.array(original_indices))

def main():
    """Run one repeated fivefold experiment for the configured seed."""
    run_dir, log_dir = cfg.setup_directories(cfg.SEED)
    global logger
    logger = cfg.setup_logging(log_dir)
    cfg.VisualizationConfig.apply_style()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    logger.info(f"\n{'='*20} SLEmodel Execution {'='*20}")
    logger.info(f"Experiment Name: {cfg.CURRENT_RUN_NAME}")
    logger.info(f"Master Seed: {cfg.SEED}")
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Latent dim: {cfg.LATENT_DIM}")
    logger.info(f"Attention heads: {cfg.ATTENTION_NUM_HEADS}")
    logger.info(f"CV Repeats: {cfg.N_REPEATS}, K-Fold Splits: {cfg.N_SPLITS_K_FOLD}")
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

    # The reported runs used one fixed set of weights derived from the complete
    # glycomics anchor cohort; the same weights were applied in every fold.
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

    for repeat in range(cfg.N_REPEATS):
        logger.info(f"\n{'='*25} Starting Repeat {repeat+1}/{cfg.N_REPEATS} {'='*25}")
        recon_data_dir = run_dir / "reconstruction_data"
        recon_data_dir.mkdir(exist_ok=True, parents=True)

        oof_val_true_labels = np.zeros(len(gly_y_encoded), dtype=int)
        oof_val_pred_scores = np.zeros((len(gly_y_encoded), cfg.NUM_CLASSES), dtype=float)
        oof_train_true_labels = np.zeros(len(gly_y_encoded), dtype=int)
        oof_train_pred_scores = np.zeros((len(gly_y_encoded), cfg.NUM_CLASSES), dtype=float)
        repeat_misclassified_samples = []

        for fold in range(cfg.N_SPLITS_K_FOLD):
            global_fold_idx = repeat * cfg.N_SPLITS_K_FOLD + fold
            fold_name = f"Repeat_{repeat+1}-Fold_{fold+1}"
            fold_dir = run_dir / fold_name
            fold_dir.mkdir(exist_ok=True)
            logger.info(f"\n--- {fold_name} ---")

            gly_train_idx, gly_val_idx = gly_splits[global_fold_idx]
            mass_train_idx, _ = mass_splits[global_fold_idx]
            rna_train_idx, _ = rna_splits[global_fold_idx]

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

            # ============================================================
            # ============================================================
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

            if cfg.USE_PS_TEMPERATURE_SCALING:
                logger.info("Applying temperature scaling calibration to PS...")
                ps_train_gly, temp_gly_train = get_propensity_scores_calibrated(
                    gly_classifier, train_gly_x, train_gly_y, cfg.DEVICE,
                    calibrate=True,
                    max_iter=cfg.PS_TEMPERATURE_SCALING_MAX_ITER,
                    lr=cfg.PS_TEMPERATURE_SCALING_LR
                )
                ps_val_gly, temp_gly_val = get_propensity_scores_calibrated(
                    gly_classifier, val_gly_x, val_gly_y, cfg.DEVICE,
                    calibrate=True,
                    max_iter=cfg.PS_TEMPERATURE_SCALING_MAX_ITER,
                    lr=cfg.PS_TEMPERATURE_SCALING_LR
                )
                ps_pool_mass, temp_mass = get_propensity_scores_calibrated(
                    mass_classifier, pool_mass_x, pool_mass_y, cfg.DEVICE,
                    calibrate=True,
                    max_iter=cfg.PS_TEMPERATURE_SCALING_MAX_ITER,
                    lr=cfg.PS_TEMPERATURE_SCALING_LR
                )
                ps_pool_rna, temp_rna = get_propensity_scores_calibrated(
                    rna_classifier, pool_rna_x, pool_rna_y, cfg.DEVICE,
                    calibrate=True,
                    max_iter=cfg.PS_TEMPERATURE_SCALING_MAX_ITER,
                    lr=cfg.PS_TEMPERATURE_SCALING_LR
                )
                logger.info(f"Calibration temperatures: Gly(train)={temp_gly_train:.3f}, "
                           f"Gly(val)={temp_gly_val:.3f}, Mass={temp_mass:.3f}, RNA={temp_rna:.3f}")
            else:
                ps_train_gly = get_propensity_scores(gly_classifier, train_gly_x, cfg.DEVICE)
                ps_val_gly = get_propensity_scores(gly_classifier, val_gly_x, cfg.DEVICE)
                ps_pool_mass = get_propensity_scores(mass_classifier, pool_mass_x, cfg.DEVICE)
                ps_pool_rna = get_propensity_scores(rna_classifier, pool_rna_x, cfg.DEVICE)

            # ============================================================
            # ============================================================
            train_dataset = DynamicHybridDataset(
                gly_data=train_gly_x, gly_labels=train_gly_y,
                all_mass_data=pool_mass_x, all_rna_data=pool_rna_x,
                ps_gly=ps_train_gly, ps_mass=ps_pool_mass, ps_rna=ps_pool_rna,
                dynamic_resample=True,
                reg_mass=cfg.OT_REG_MASS,
                reg_rna=cfg.OT_REG_RNA,
                ot_method_mass=cfg.OT_METHOD_MASS,
                ot_method_rna=cfg.OT_METHOD_RNA,
                ot_cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                ot_cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),

                resample_every=5,
                sinkhorn_numItermax_mass=400,
                sinkhorn_stopThr_mass=1e-3,
                sinkhorn_numItermax_rna=600,
                sinkhorn_stopThr_rna=1e-3,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )

            val_mass_coupling = get_ot_coupling_modality_specific(
                ps_val_gly, ps_pool_mass, modality='mass',
                reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
                cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean")
            )
            val_rna_coupling = get_ot_coupling_modality_specific(
                ps_val_gly, ps_pool_rna, modality='rna',
                reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,
                cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean")
            )
            save_val_ps_ot(fold_dir, ps_val_gly, ps_pool_mass, ps_pool_rna, val_mass_coupling, val_rna_coupling)
            val_dataset = StaticMatchedDataset(
                val_gly_x, val_gly_y, pool_mass_x, pool_rna_x,
                val_mass_coupling, val_rna_coupling, gly_ids_raw[gly_val_idx]
            )
            val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

            model = DecoupledModel(
                gly_x_raw.shape[1], mass_x_raw.shape[1], rna_x_raw.shape[1]
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

                train_loader.dataset.resample_and_match()

                for batch in train_loader:
                    optimizer.zero_grad()
                    gly_x, mass_x, rna_x, labels = batch['gly_x'].to(cfg.DEVICE), batch['mass_x'].to(cfg.DEVICE), batch['rna_x'].to(cfg.DEVICE), batch['label'].to(cfg.DEVICE)

                    logits, recon_matched, reps = model(gly_x, mass_x, rna_x)

                    loss_cls = cls_criterion(logits, labels)

                    loss_recon = (recon_criterion(recon_matched['gly'], gly_x) +
                                  recon_criterion(recon_matched['mass'], mass_x) +
                                  recon_criterion(recon_matched['rna'], rna_x))

                    real_gly, real_mass, real_rna = batch['real_gly'].to(cfg.DEVICE), batch['real_mass'].to(cfg.DEVICE), batch['real_rna'].to(cfg.DEVICE)
                    recon_gly_real, _, _ = model.reconstruct_single_modality(real_gly, 'gly')
                    recon_mass_real, _, _ = model.reconstruct_single_modality(real_mass, 'mass')
                    recon_rna_real, _, _ = model.reconstruct_single_modality(real_rna, 'rna')
                    loss_semi = recon_criterion(recon_gly_real, real_gly) + recon_criterion(recon_mass_real, real_mass) + recon_criterion(recon_rna_real, real_rna)

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
                oof_val_true_labels[gly_val_idx] = val_true
                oof_val_pred_scores[gly_val_idx] = val_scores

            val_cls, val_rec_met, val_report_dict = evaluate_and_visualize(
                history, val_true, val_scores, val_logits, val_recon, run_dir, fold_name,
                "Validation (Best Model)", class_weights=class_weights.cpu()
            )

            if len(val_true) > 0:
                val_pred = np.argmax(val_scores, axis=1)
                for i in np.where(val_true != val_pred)[0]:
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

            train_mass_coupling = get_ot_coupling_modality_specific(
                ps_train_gly, ps_pool_mass, modality='mass',
                reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,

                cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
            )

            train_rna_coupling = get_ot_coupling_modality_specific(
                ps_train_gly, ps_pool_rna, modality='rna',
                reg_mass=cfg.OT_REG_MASS, reg_rna=cfg.OT_REG_RNA,
                method_mass=cfg.OT_METHOD_MASS, method_rna=cfg.OT_METHOD_RNA,

                cost_metric_mass=getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
                cost_metric_rna=getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),
            )
            train_static_dataset = StaticMatchedDataset(
                train_gly_x, train_gly_y, pool_mass_x, pool_rna_x,
                train_mass_coupling, train_rna_coupling, gly_ids_raw[gly_train_idx]
            )
            train_static_loader = DataLoader(train_static_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

            train_true, train_scores, train_logits, train_recon, _, _ = get_all_metrics_from_loader(model, train_static_loader, cfg.DEVICE)

            if len(train_true) > 0:
                oof_train_true_labels[gly_train_idx] = train_true
                oof_train_pred_scores[gly_train_idx] = train_scores

            train_cls, train_rec_met, train_report_dict = evaluate_and_visualize(
                {}, train_true, train_scores, train_logits, train_recon, run_dir, fold_name,
                "Training (Best Model)", class_weights=class_weights.cpu()
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

    logger.info("\n" + "="*25 + " Final Summary " + "="*25)
    logger.info("\nValidation Set Metrics (mean +/- std):\n" + cls_summary_df[cls_summary_df['set'] == 'validation'].describe().loc[['mean', 'std']].to_string(float_format="%.4f"))
    logger.info("\nTraining Set Metrics (mean +/- std):\n" + cls_summary_df[cls_summary_df['set'] == 'training'].describe().loc[['mean', 'std']].to_string(float_format="%.4f"))

    logger.info("Experiment completed successfully.")


def run_repeated_cv(seeds=(42, 100, 2025, 7, 123)):
    """Run the reported fivefold cross-validation for each master seed."""
    for seed in seeds:
        cfg.SEED = seed
        cfg.CURRENT_RUN_NAME = f"SLEmodel_Run_Seed_{seed}"
        main()


if __name__ == '__main__':
    run_repeated_cv()
