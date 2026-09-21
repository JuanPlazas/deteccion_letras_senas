"""LSTM model architecture for sign-language sequence classification.

Given ``(SEQUENCE_LENGTH, NUM_FEATURES)`` normalized landmark sequences, the
network learns temporal patterns and outputs a probability per letter class.
"""

from __future__ import annotations

from tensorflow import keras

from training.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    LSTM_UNITS,
    LSTM_UNITS_2,
)


def build_model(
    *,
    num_classes: int,
    sequence_length: int,
    num_features: int,
) -> keras.Model:
    """Build and compile the two-layer LSTM classifier.

    Args:
        num_classes: Number of letter classes to predict.
        sequence_length: Number of frames per sequence (input timesteps).
        num_features: Number of features per frame (input dimension).

    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, num_features)),
            keras.layers.LSTM(LSTM_UNITS, return_sequences=True),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.LSTM(LSTM_UNITS_2),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(DENSE_UNITS, activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
