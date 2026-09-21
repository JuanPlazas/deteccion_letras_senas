"""Training hyperparameters and MLflow settings.

Kept separate from the training logic so experiments can be tuned without
touching the rest of the pipeline.
"""

from utils.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'training' / 'mlflow.db'}"
EXPERIMENT_NAME = "deteccion-letras-senas"

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
LSTM_UNITS = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5
