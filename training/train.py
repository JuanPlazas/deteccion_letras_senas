"""Train the LSTM classifier and log everything to MLflow.

Run from the project root::

    uv run python -m training.train

Every hyperparameter, per-epoch metric, artifact and the final test result is
logged, so any run is fully reproducible from the MLflow UI.
"""

from __future__ import annotations

import json
import tempfile

import joblib
import mlflow
import mlflow.tensorflow
from tensorflow import keras

from training.config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENT_NAME,
    LEARNING_RATE,
    LSTM_UNITS,
    LSTM_UNITS_2,
    MAX_EPOCHS,
    MLFLOW_URI,
    RANDOM_STATE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    TRAIN_RATIO,
    VAL_RATIO,
)
from training.data_loader import prepare_data
from training.model import build_model
from utils.config import NUM_FEATURES, NUM_LANDMARKS, SEQUENCE_LENGTH


def main() -> None:
    """Load data, build the model, train it and log the run to MLflow."""
    # ------------------------------------------------------------------
    # 1. DATA
    # ------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, label_names, scaler = prepare_data()
    num_classes = len(label_names)

    print(f"Letters discovered: {label_names}")
    print(f"Samples: {len(y_train)} train / {len(y_val)} val / {len(y_test)} test")

    # ------------------------------------------------------------------
    # 2. MODEL
    # ------------------------------------------------------------------
    model = build_model(
        num_classes=num_classes,
        sequence_length=SEQUENCE_LENGTH,
        num_features=NUM_FEATURES,
    )
    model.summary()

    # ------------------------------------------------------------------
    # 3. MLFLOW SETUP
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lstm-baseline") as run:
        # --- Parameters ------------------------------------------------
        mlflow.log_params(
            {
                "model": "LSTM-two-layer",
                "sequence_length": SEQUENCE_LENGTH,
                "num_features": NUM_FEATURES,
                "num_landmarks": NUM_LANDMARKS,
                "num_classes": num_classes,
                "labels": ",".join(label_names),
                "lstm_units": LSTM_UNITS,
                "lstm_units_2": LSTM_UNITS_2,
                "dense_units": DENSE_UNITS,
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "reduce_lr_patience": REDUCE_LR_PATIENCE,
                "reduce_lr_factor": REDUCE_LR_FACTOR,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "train_samples": len(y_train),
                "val_samples": len(y_val),
                "test_samples": len(y_test),
                "random_state": RANDOM_STATE,
            }
        )

        # --- Callbacks ------------------------------------------------
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
            ),
        ]

        # --- Training loop ---------------------------------------------
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=MAX_EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

        # --- Per-epoch metrics -----------------------------------------
        for epoch in range(len(history.history["loss"])):
            metrics = {
                "train_loss": history.history["loss"][epoch],
                "train_accuracy": history.history["accuracy"][epoch],
                "val_loss": history.history["val_loss"][epoch],
                "val_accuracy": history.history["val_accuracy"][epoch],
            }
            if "lr" in history.history:
                metrics["learning_rate"] = float(history.history["lr"][epoch])
            mlflow.log_metrics(metrics, step=epoch + 1)

        # --- Test evaluation -------------------------------------------
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)})
        print(f"\nTest loss: {test_loss:.4f}  Test accuracy: {test_accuracy:.4f}")

        # --- Log the trained model with signature -----------------------
        sample_input = X_test[:1]
        sample_output = model.predict(sample_input, verbose=0)
        signature = mlflow.models.infer_signature(sample_input, sample_output)
        mlflow.tensorflow.log_model(model, name="model", signature=signature)

        # --- Log the fitted scaler -------------------------------------
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(scaler, f.name)
        mlflow.log_artifact(f.name, artifact_path="preprocessing")

        # --- Log the label mapping -------------------------------------
        # The file must be closed before logging so its buffer is flushed.
        label_map = {i: name for i, name in enumerate(label_names)}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(label_map, f)
        mlflow.log_artifact(f.name, artifact_path="preprocessing")

        print(f"\nMLflow run:  {run.info.run_id}")
        print(f"Tracking URI: {MLFLOW_URI}")


if __name__ == "__main__":
    main()
