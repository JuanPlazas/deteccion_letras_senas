"""LSTM model for sign-language letter classification.

A two-layer LSTM that reads 30 frames of 42 normalized hand-landmark
coordinates and outputs a probability distribution over the captured letters.
"""

from tensorflow import keras
from tensorflow.keras import layers, models

from training.config import DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE, LSTM_UNITS


def build_model(
    num_classes: int,
    sequence_length: int,
    num_features: int,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> models.Model:
    """Build and compile the LSTM classifier.

    Architecture::

        Input  (30, 42)
        LSTM   (64, return_sequences=True)   → capture short-range motion
        Dropout (0.3)
        LSTM   (32)                           → summarize the sequence
        Dropout (0.3)
        Dense  (32, relu)                     → non-linear combination
        Dense  (N, softmax)                   → class probabilities

    Args:
        num_classes: Number of letters to classify.
        sequence_length: Frames per sequence (from config).
        num_features: Features per frame (21 landmarks × 2 coords).
        lstm_units: Hidden units in the first LSTM layer.
        dense_units: Units in the penultimate Dense layer.
        dropout_rate: Fraction of units dropped after each LSTM layer.
        learning_rate: Adam optimizer learning rate.

    Returns:
        A compiled ``keras.Model``.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, num_features)),
            layers.LSTM(lstm_units, return_sequences=True, name="lstm_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.LSTM(lstm_units // 2, name="lstm_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            layers.Dense(dense_units, activation="relu", name="dense_hidden"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
