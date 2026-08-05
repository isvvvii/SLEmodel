# ablation_runner.py

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

from . import config as cfg
from .config import AblationConfig


def setup_experiment_logging(exp_name: str):
    """Configure a separate log file for an ablation run."""
    log_dir = Path("ablation_logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{exp_name}_{timestamp}.log"

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def create_experiment_config(exp_id: str, seed: int):
    """Build the configuration for one experiment and master seed."""
    base_config = {
        "EXPERIMENT_TAG": f"Ablation_Study_{exp_id}",
        "CURRENT_RUN_NAME": f"{AblationConfig.EXPERIMENTS[exp_id]}_seed_{seed}",
        "SEED": int(seed),
        "N_REPEATS": 1,
        "N_SPLITS_K_FOLD": int(getattr(cfg, "N_SPLITS_K_FOLD", 5)),

        "USE_CROSS_ATTENTION": bool(getattr(cfg, "USE_CROSS_ATTENTION", True)),
        "USE_DECOUPLING": bool(getattr(cfg, "USE_DECOUPLING", True)),
        "DYNAMIC_OT_RESAMPLE": bool(getattr(cfg, "DYNAMIC_OT_RESAMPLE", True)),
        "USE_RANDOM_MATCHING": bool(getattr(cfg, "USE_RANDOM_MATCHING", False)),
        "USE_OT_MATCHING": bool(getattr(cfg, "USE_OT_MATCHING", True)),
        "SINGLE_MODALITY_MODE": getattr(cfg, "SINGLE_MODALITY_MODE", None),
        "ATTENTION_NUM_HEADS": int(getattr(cfg, "ATTENTION_NUM_HEADS", 1)),
        "FUSION_METHOD": getattr(cfg, "FUSION_METHOD", "attention"),
        "LATENT_DIM": int(getattr(cfg, "LATENT_DIM", 64)),
        "DROPOUT_RATE": float(getattr(cfg, "DROPOUT_RATE", 0.6)),
        "RECON_LOSS_WEIGHT": float(getattr(cfg, "RECON_LOSS_WEIGHT", 0.8)),
        "SEMI_SUPERVISED_WEIGHT": float(getattr(cfg, "SEMI_SUPERVISED_WEIGHT", 0.6)),
        "ORTHO_LOSS_WEIGHT": float(getattr(cfg, "ORTHO_LOSS_WEIGHT", 0.1)),

        "OT_REG_MASS": float(getattr(cfg, "OT_REG_MASS", 2e-4)),
        "OT_REG_RNA": float(getattr(cfg, "OT_REG_RNA", 2e-3)),
        "OT_METHOD_MASS": getattr(cfg, "OT_METHOD_MASS", "standard"),
        "OT_METHOD_RNA": getattr(cfg, "OT_METHOD_RNA", "standard"),
        "OT_COST_METRIC_MASS": getattr(cfg, "OT_COST_METRIC_MASS", "sqeuclidean"),
        "OT_COST_METRIC_RNA": getattr(cfg, "OT_COST_METRIC_RNA", "sqeuclidean"),

        "USE_PS_TEMPERATURE_SCALING": bool(getattr(cfg, "USE_PS_TEMPERATURE_SCALING", False)),
        "PS_TEMPERATURE_SCALING_MAX_ITER": int(getattr(cfg, "PS_TEMPERATURE_SCALING_MAX_ITER", 100)),
        "PS_TEMPERATURE_SCALING_LR": float(getattr(cfg, "PS_TEMPERATURE_SCALING_LR", 0.01)),
    }

    experiment_config = AblationConfig.get_config_for_experiment(exp_id, base_config)
    return experiment_config


def run_single_experiment(exp_id: str, seed: int, logger: logging.Logger):
    """Run one ablation experiment in a separate process."""
    logger.info(f"Starting experiment {exp_id} with seed {seed}")

    try:
        exp_name = AblationConfig.EXPERIMENTS[exp_id]
    except KeyError:
        logger.error(f"Unknown exp_id: {exp_id}, please check AblationConfig.EXPERIMENTS.")
        return False

    exp_tag = f"Ablation_Study_{exp_id}"
    run_name = f"{exp_id}_{exp_name}_seed_{seed}"
    exp_path = Path(cfg.BASE_EXPERIMENT_DIR) / exp_tag / run_name / f"master_seed_{seed}"
    metrics_file = exp_path / "metrics" / "classification_metrics_per_fold.csv"

    if metrics_file.exists():
        logger.info(f"[SKIP] Experiment {exp_id}_seed_{seed} already has metrics: {metrics_file}")
        return True
    # === skip end ===

    temp_dir = Path(__file__).resolve().parent / "temp_configs"
    temp_dir.mkdir(exist_ok=True)
    config_file = temp_dir / f"temp_config_{exp_id}_{seed}.json"

    try:
        exp_config = create_experiment_config(exp_id, seed)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(exp_config, f, indent=2, ensure_ascii=False)

        cmd = [
            sys.executable,
            "-m", "analyses.ablation.main_ablation",
            "--config", str(config_file),
            "--exp_id", exp_id,
            "--seed", str(seed),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
        )

        logger.info(f"--- STDOUT for {exp_id}_seed_{seed} ---")
        logger.info(result.stdout if result.stdout else "No standard output.")

        logger.info(f"--- STDERR for {exp_id}_seed_{seed} ---")
        if result.stderr:
            logger.warning(result.stderr)
        else:
            logger.info("No standard error.")
        logger.info(f"--- End of Output for {exp_id}_seed_{seed} ---")

        if result.returncode == 0:
            logger.info(f"Experiment {exp_id}_seed_{seed} completed successfully (Return Code: 0)")
            return True

        logger.error(f"Experiment {exp_id}_seed_{seed} failed (Return Code: {result.returncode})")
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"Experiment {exp_id}_seed_{seed} timed out after 2 hours")
        return False

    except Exception as e:
        logger.error(f"An unexpected error occurred while running experiment {exp_id}_seed_{seed}: {str(e)}", exc_info=True)
        return False

    finally:
        config_file.unlink(missing_ok=True)


def main():
    """Run all reported ablations across the five master seeds."""
    print("=== SLEmodel ablation experiments ===")
    print(f"Experiments: {len(AblationConfig.EXPERIMENTS)}")
    print(f"Seeds per experiment: {len(AblationConfig.EXPERIMENT_SEEDS)}")
    print(f"Total runs: {len(AblationConfig.EXPERIMENTS) * len(AblationConfig.EXPERIMENT_SEEDS)}")

    main_logger = setup_experiment_logging("MAIN_ABLATION")

    results = {}
    total_experiments = len(AblationConfig.EXPERIMENTS) * len(AblationConfig.EXPERIMENT_SEEDS)
    completed = 0

    for exp_id, exp_name in AblationConfig.EXPERIMENTS.items():
        main_logger.info(f"\n{'='*50}")
        main_logger.info(f"Starting experiment group: {exp_id} - {exp_name}")
        main_logger.info(f"{'='*50}")

        exp_logger = setup_experiment_logging(f"EXP_{exp_id}")
        results[exp_id] = {}

        for seed in AblationConfig.EXPERIMENT_SEEDS:
            success = run_single_experiment(exp_id, seed, exp_logger)
            results[exp_id][seed] = success
            completed += 1

            main_logger.info(
                f"Progress: {completed}/{total_experiments} "
                f"({completed/total_experiments*100:.1f}%) - "
                f"{exp_id}_seed_{seed}: {'SUCCESS' if success else 'FAILED'}"
            )

    main_logger.info(f"\n{'='*50}")
    main_logger.info("FINAL SUMMARY")
    main_logger.info(f"{'='*50}")

    total_success = 0
    for exp_id, exp_results in results.items():
        success_count = sum(exp_results.values())
        total_success += success_count
        main_logger.info(f"{exp_id}: {success_count}/{len(AblationConfig.EXPERIMENT_SEEDS)} successful")

    main_logger.info(f"\nOverall: {total_success}/{total_experiments} experiments completed successfully")
    main_logger.info(f"Success rate: {total_success/total_experiments*100:.1f}%")

    with open("ablation_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
