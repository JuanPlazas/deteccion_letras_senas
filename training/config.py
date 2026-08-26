"""Training hyperparameters and paths.

Centralizes every tuneable value so that a run is fully reproducible from
this single file.  Path constants are imported from ``utils.config`` to stay
in sync with the capture pipeline.
"""

from pathlib import Path

# Paths.
TRAINING_DIR = Path(__file__).resolve().parent
MLFLOW_DB = TRAINING_DIR / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
EXPERIMENT_NAME = "lse-sign-language"

# Data split ratios.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# Model hyper-parameters.
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# Training loop.
BATCH_SIZE = 8
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
