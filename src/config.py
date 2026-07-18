
from pathlib import Path

# =====================================
# PROJECT ROOT
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =====================================
# DATA DIRECTORIES
# =====================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# =====================================
# MODEL DIRECTORIES
# =====================================

MODELS_DIR = PROJECT_ROOT / "models"

TRAINED_MODELS_DIR = MODELS_DIR / "trained"
METRICS_DIR = MODELS_DIR / "metrics"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# =====================================
# DATASET CONFIGURATION
# Change these whenever a new dataset arrives
# =====================================

TRAINING_BATCH = "18062026"
INFERENCE_BATCH = "24062026"

TRAINING_FOLDER = (
    RAW_DIR
    / "training"
    / TRAINING_BATCH
)

INFERENCE_FOLDER = (
    RAW_DIR
    / "inference"
    / INFERENCE_BATCH
)

# =====================================
# FILES
# =====================================

GROUND_TRUTH_FILE = (
    RAW_DIR
    / "metadata"
    / "ground_truth.csv"
)

PROCESSED_TRAINING_FILE = (
    PROCESSED_DIR
    / "training_dataset.csv"
)

PREDICTION_OUTPUT_FILE = (
    PREDICTIONS_DIR
    / "predictions.csv"
)

# =====================================
# MACHINE LEARNING
# =====================================

RANDOM_STATE = 42
TEST_SIZE = 0.30

# =====================================
# MODEL PATHS
# =====================================

SVM_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "svm.pkl"
)

XGB_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "xgb.pkl"
)

SVM_METRICS_FILE = (
    METRICS_DIR
    / "svm_metrics.json"
)

XGB_METRICS_FILE = (
    METRICS_DIR
    / "xgb_metrics.json"
)