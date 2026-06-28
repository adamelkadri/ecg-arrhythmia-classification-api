"""
Central configuration for the ECG arrhythmia project.

Everything that you might want to tweak (paths, hyperparameters, class names)
lives here so the other modules stay clean and there is a single source of
truth. Import it like:

    from src import config
    print(config.NUM_CLASSES)
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# PROJECT_ROOT = the repo root (one level up from this src/ file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "mitbih_train.csv"
TEST_CSV = DATA_DIR / "mitbih_test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "ecg_cnn.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# --------------------------------------------------------------------------- #
# Data / model dimensions
# --------------------------------------------------------------------------- #
SIGNAL_LENGTH = 187      # number of ECG samples per heartbeat (CSV cols 0..186)
NUM_CLASSES = 5          # N, S, V, F, Q

# Human-readable class names, indexed by label 0..4.
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_DESCRIPTIONS = {
    0: "Normal beat",
    1: "Supraventricular ectopic beat",
    2: "Ventricular ectopic beat",
    3: "Fusion beat",
    4: "Unknown / unclassifiable beat",
}

# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# How we fight class imbalance. Options: "class_weights" or "weighted_sampler".
IMBALANCE_STRATEGY = "class_weights"

# Fraction of the training set held out for validation during training.
VAL_SPLIT = 0.1

# Reproducibility.
RANDOM_SEED = 42
