# config.py

import torch
import logging
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EXPERIMENT_TAG = "ablation-runs"
CURRENT_RUN_NAME = "main_analysis"
BASE_EXPERIMENT_DIR = "experiments"

GLY_PATH = 'data/gly/glycomics_model_input.csv'
MASS_PATH = 'data/mass/mass_model_input.csv'
RNA_PATH = 'data/rna/rna_model_input.csv'
RNA_FEATURE_LIST_PATH = 'data/features/rna_features_1124.csv'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_CLASSES = 3

N_SPLITS_K_FOLD = 5
N_REPEATS = 1

BATCH_SIZE = 32
EPOCHS_PS = 100
LATENT_DIM = 64
ATTENTION_NUM_HEADS = 4

EPOCHS_DOWNSTREAM = 500
LR_DOWNSTREAM = 1e-4

RECON_LOSS_WEIGHT = 0.8
SEMI_SUPERVISED_WEIGHT = 0.6
ORTHO_LOSS_WEIGHT = 0.1
WEIGHT_DECAY_DOWNSTREAM = 0.005
DROPOUT_RATE = 0.6

# =============================================================================
# Early Stopping Configuration
# =============================================================================
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 40
EARLY_STOPPING_METRIC = 'val_class_loss'
EARLY_STOPPING_MODE = 'min'
EARLY_STOPPING_MIN_EPOCHS = 25

class VisualizationConfig:
    LABEL_NAMES = {0: "Stable", 1: "Active", 2: "Control"}
    LABEL_COLORS = {0: "#8aab82", 1: "#E68B81", 2: "#7DA6C6"}

    MODALITY_NAMES = {"gly": "Glycan", "mass": "Mass", "rna": "RNA"}
    MODALITY_COLORS = {"gly": "#E5A79A", "mass": "#ABC8E5", "rna": "#7DA494"}
    METRIC_COLORS = {"r2": "#6D8A96", "rmse": "#B7A29E", "mae": "#9A9B73"}
    RECON_SCATTER_COLOR = "#6B5B95"
    RECON_IDEAL_LINE_COLOR = "#BF616A"
    HEATMAP_CMAP = LinearSegmentedColormap.from_list(
        "morandi_heatmap",
        ["#F7F3F0", "#E2D7D1", "#C89F9C", "#748B75"]
    )
    LEARNING_CURVE_COLORS = {'train': '#B48E89', 'validation': '#778D8E'}

    STYLE = 'seaborn-v0_8-whitegrid'
    DPI = 300
    TITLE_FONTSIZE = 18
    LABEL_FONTSIZE = 15
    TICK_FONTSIZE = 13
    LEGEND_FONTSIZE = 20
    ANNOTATION_FONTSIZE = 18

    @classmethod
    def apply_style(cls):
        """Apply the plotting style used for the ablation figures."""
        plt.style.use(cls.STYLE)

        font_family = 'serif'
        font_serif_list = ['Times New Roman', 'Helvetica', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif']

        plt.rcParams.update({
            'figure.dpi': cls.DPI,
            'font.family': font_family,
            'font.serif': font_serif_list,
            'font.size': cls.LABEL_FONTSIZE,
            'axes.labelsize': cls.LABEL_FONTSIZE,
            'axes.titlesize': cls.TITLE_FONTSIZE,
            'xtick.labelsize': cls.TICK_FONTSIZE,
            'ytick.labelsize': cls.TICK_FONTSIZE,
            'legend.fontsize': cls.LEGEND_FONTSIZE,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'mathtext.fontset': 'stix',
            'mathtext.rm': 'Times New Roman',
            'mathtext.it': 'Times New Roman:italic',
            'mathtext.bf': 'Times New Roman:bold',
        })


def setup_directories(run_seed, exp_id=None):
    """Create the output and log directories for one ablation run."""
    if exp_id is not None:
        run_dir = Path(BASE_EXPERIMENT_DIR) / EXPERIMENT_TAG / f"{exp_id}_{CURRENT_RUN_NAME}" / f"master_seed_{run_seed}"
    else:
        run_dir = Path(BASE_EXPERIMENT_DIR) / EXPERIMENT_TAG / CURRENT_RUN_NAME / f"master_seed_{run_seed}"

    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, log_dir

def setup_logging(log_dir):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = log_dir / f"training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info(f"Logging to: {log_filepath}")
    return logger

# =============================================================================
# Ablation Study Configuration
# =============================================================================
class AblationConfig:
    """Definitions of the reported ablation experiments."""

    EXPERIMENTS = {
        'A0': 'SLEmodel',
        'A1': 'No_CrossAttention',
        'A2': 'Only_PairRecon',
        'A3': 'Only_SemiRecon',
        'A4': 'No_Decoupling',
        'A5': 'Static_OT',
        'A6': 'Random_Matching',
        'B1': 'Gly_Only',
        'B2': 'Mass_Only',
        'B3': 'RNA_Only',

        # Fusion: attention-head ablations relative to the four-head SLEmodel
        'C1_1': 'MultiHead_1',
        'C1_2': 'MultiHead_2',

        'C2_05': 'Dropout_05',
        'C2_09': 'Dropout_09',
        'C3': 'Concat_MLP_Fusion',
        'C4_32': 'LatentDim_32',
        'C4_128': 'LatentDim_128',
    }

    EXPERIMENT_SEEDS = [42, 100, 2025, 7, 123]

    @classmethod
    def get_config_for_experiment(cls, exp_id, base_config_dict):
        config = base_config_dict.copy()

        if exp_id == 'A0':
            pass
        elif exp_id == 'A1':
            config['USE_CROSS_ATTENTION'] = False
        elif exp_id == 'A2':
            config['SEMI_SUPERVISED_WEIGHT'] = 0.0
        elif exp_id == 'A3':
            config['RECON_LOSS_WEIGHT'] = 0.0
        elif exp_id == 'A4':
            config['ORTHO_LOSS_WEIGHT'] = 0.0
            config['USE_DECOUPLING'] = False
        elif exp_id == 'A5':
            config['DYNAMIC_OT_RESAMPLE'] = False
        elif exp_id == 'A6':
            config['USE_RANDOM_MATCHING'] = True
            config['USE_OT_MATCHING'] = False
            config['DYNAMIC_OT_RESAMPLE'] = False
        elif exp_id in ['B1', 'B2', 'B3']:
            modality_map = {'B1': 'gly', 'B2': 'mass', 'B3': 'rna'}
            config['SINGLE_MODALITY_MODE'] = modality_map[exp_id]
            config['USE_OT_MATCHING'] = False
            config['USE_RANDOM_MATCHING'] = False
        elif exp_id.startswith('C1_'):
            num_heads = int(exp_id.split('_')[1])
            config['ATTENTION_NUM_HEADS'] = num_heads
        elif exp_id.startswith('C2_'):
            dropout_int = int(exp_id.split('_')[1])
            config['DROPOUT_RATE'] = float(dropout_int) / 10.0
        elif exp_id == 'C3':
            config['FUSION_METHOD'] = 'concat_mlp'
        elif exp_id.startswith('C4_'):
            latent_dim = int(exp_id.split('_')[1])
            config['LATENT_DIM'] = latent_dim

        return config

USE_CROSS_ATTENTION = True
USE_DECOUPLING = True
DYNAMIC_OT_RESAMPLE = True
USE_RANDOM_MATCHING = False
USE_OT_MATCHING = True

SINGLE_MODALITY_MODE = None  # None, 'gly', 'mass', 'rna'

# SLEmodel uses four-head attention.
ATTENTION_NUM_HEADS = 4

FUSION_METHOD = 'attention'  # 'attention', 'average', 'concat_mlp'

# =============================================================================
# Alignment Optimization Configuration (sync from alignment folder)
# =============================================================================

# Modality-specific OT regularization (sensitivity-tuned)
OT_REG_MASS = 2e-4
OT_REG_RNA  = 2e-3

# OT method per modality
OT_METHOD_MASS = "soft_stratified"
OT_METHOD_RNA  = "soft_stratified"   # mitigate RNA-Control imbalance

# OT cost metric per modality
OT_COST_METRIC_MASS = "hellinger"
OT_COST_METRIC_RNA  = "sqeuclidean"

# PS temperature scaling (optional)
USE_PS_TEMPERATURE_SCALING = False
PS_TEMPERATURE_SCALING_MAX_ITER = 100
PS_TEMPERATURE_SCALING_LR = 0.01
