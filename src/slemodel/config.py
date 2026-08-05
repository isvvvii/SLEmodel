import torch
import logging
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EXPERIMENT_TAG = "SLEmodel"
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
WEIGHT_DECAY_DOWNSTREAM = 0.005
DROPOUT_RATE = 0.6

RECON_LOSS_WEIGHT = 0.8
SEMI_SUPERVISED_WEIGHT = 0.6
ORTHO_LOSS_WEIGHT = 0.1

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

    RECON_SCATTER_COLORS = {
        "gly": RECON_SCATTER_COLOR,
        "mass": "#2A9D8F",
        "rna": "#E07A5F",
    }

    RECON_SCATTER_ALPHA = 0.35
    RECON_SCATTER_SIZE = 10
    RECON_REG_LINE_WIDTH = 2.2
    RECON_MARGINAL_BINS = 30

    HEATMAP_CMAP = LinearSegmentedColormap.from_list("morandi_heatmap", ["#F7F3F0", "#E2D7D1", "#C89F9C", "#748B75"])
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
        plt.style.use(cls.STYLE)
        plt.rcParams.update({
            'figure.dpi': cls.DPI, 'font.family': 'sans-serif', 'font.sans-serif': ['Times New Roman', 'Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': cls.LABEL_FONTSIZE, 'axes.labelsize': cls.LABEL_FONTSIZE, 'axes.titlesize': cls.TITLE_FONTSIZE,
            'xtick.labelsize': cls.TICK_FONTSIZE, 'ytick.labelsize': cls.TICK_FONTSIZE, 'legend.fontsize': cls.LEGEND_FONTSIZE,
            'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'
        })

def setup_directories(run_seed):
    """Create the output and log directories for one master seed."""
    run_dir = Path(BASE_EXPERIMENT_DIR) / EXPERIMENT_TAG / CURRENT_RUN_NAME / f"master_seed_{run_seed}"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, log_dir

def setup_logging(log_dir):
    """Configure file and console logging for a run."""
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

OT_REG_MASS = 2e-4
OT_REG_RNA = 2e-3
OT_METHOD_MASS = "soft_stratified"
OT_METHOD_RNA = "soft_stratified"

OT_COST_METRIC_MASS = "hellinger"
OT_COST_METRIC_RNA  = "sqeuclidean"

USE_PS_TEMPERATURE_SCALING = False
PS_TEMPERATURE_SCALING_MAX_ITER = 100
PS_TEMPERATURE_SCALING_LR = 0.01
