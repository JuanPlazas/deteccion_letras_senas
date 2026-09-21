"""Load the captured sequences and prepare train/val/test tensors.

The dataset lives in ``data/sequences/<LETTER>/<LETTER>_NNNN.npy``. Each file
holds ``(SEQUENCE_LENGTH, NUM_FEATURES)`` normalized landmarks. This module
reads them, builds the feature/label tensors and applies a stratified split
plus per-feature standardization (fit only on the training set).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.config import RANDOM_STATE, TRAIN_RATIO, VAL_RATIO
from utils.config import NUM_FEATURES, SEQUENCE_LENGTH, SEQUENCES_DIR


def _load_sequences() -> tuple[list[str], NDArray[np.float32], NDArray[np.int64]]:
    """Walk the sequences directory and return labels, X and y arrays.

    Letters are discovered from the subdirectory names and sorted, so the
    class-to-index mapping is deterministic.

    Returns:
        ``(label_names, X, y)`` where ``label_names[i]`` is the letter at
        index ``i``. ``X`` has shape ``(n, SEQUENCE_LENGTH, NUM_FEATURES)``.

    Raises:
        ValueError: If no sequences are found or a file has the wrong shape.
    """
    label_names = sorted(p.name for p in SEQUENCES_DIR.iterdir() if p.is_dir())
    if not label_names:
        raise ValueError(
            f"No letter folders found under {SEQUENCES_DIR}. "
            "Capture a dataset first with scripts/capture_data.py."
        )

    sequences: list[NDArray[np.float32]] = []
    labels: list[int] = []
    for class_index, letter in enumerate(label_names):
        letter_dir = SEQUENCES_DIR / letter
        for seq_file in sorted(letter_dir.glob(f"{letter}_*.npy")):
            data = np.load(seq_file)
            if data.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                raise ValueError(
                    f"{seq_file} has shape {data.shape}, expected "
                    f"({SEQUENCE_LENGTH}, {NUM_FEATURES})."
                )
            sequences.append(data.astype(np.float32))
            labels.append(class_index)

    if not sequences:
        raise ValueError(f"No sequence files found under {SEQUENCES_DIR}.")

    X = np.stack(sequences)
    y = np.asarray(labels, dtype=np.int64)
    return label_names, X, y


def prepare_data() -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.int64],
    list[str],
    StandardScaler,
]:
    """Load, split and standardize the dataset.

    Returns:
        ``X_train, y_train, X_val, y_val, X_test, y_test, label_names, scaler``.
        The scaler is fitted on the training set only.
    """
    label_names, X, y = _load_sequences()

    # Stratified split so the letter distribution is preserved in every set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=1.0 - TRAIN_RATIO,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    # Split the remaining 30% into validation and test halves.
    val_test_ratio = VAL_RATIO / (1.0 - TRAIN_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - val_test_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Standardize per feature; fit on train only to avoid data leakage.
    scaler = StandardScaler()
    flat_train = X_train.reshape(-1, NUM_FEATURES)
    scaler.fit(flat_train)

    def _transform(data: NDArray[np.float32]) -> NDArray[np.float32]:
        flat = data.reshape(-1, NUM_FEATURES)
        standardized = scaler.transform(flat)
        return standardized.reshape(data.shape)

    return (
        _transform(X_train),
        y_train,
        _transform(X_val),
        y_val,
        _transform(X_test),
        y_test,
        label_names,
        scaler,
    )
