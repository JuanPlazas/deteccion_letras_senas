"""Load, split and standardize sign-language landmark sequences.

Reads ``.npy`` files from ``data/sequences/<LETTER>/``, partitions them into
train / validation / test sets (stratified when possible), and fits a
``StandardScaler`` on the training split only.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.config import (
    RANDOM_STATE,
    TEST_RATIO,
    VAL_RATIO,
)
from utils.config import NUM_FEATURES, SEQUENCE_LENGTH, SEQUENCES_DIR


def load_sequences(sequences_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read every ``.npy`` file under *sequences_dir* and return arrays.

    Returns:
        X: Array of shape ``(N, SEQUENCE_LENGTH, NUM_FEATURES)``.
        y: Integer label array of shape ``(N,)``.
        label_names: Sorted list of discovered letter names.
    """
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    label_names = sorted(d.name for d in sequences_dir.iterdir() if d.is_dir())

    for label_idx, letter in enumerate(label_names):
        letter_dir = sequences_dir / letter
        for npy_file in sorted(letter_dir.glob("*.npy")):
            seq = np.load(npy_file)
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                sequences.append(seq)
                labels.append(label_idx)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, label_names


def _stratified_or_random(X, y, test_size, random_state):
    """Try a stratified split; fall back to random if a class is too small."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prepare_data() -> tuple[np.ndarray, ...]:
    """Full pipeline: load -> split -> standardize.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test,
        label_names, scaler.
    """
    X, y, label_names = load_sequences(SEQUENCES_DIR)

    # First split: train vs (val + test).
    test_val_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = _stratified_or_random(
        X, y, test_size=test_val_ratio, random_state=RANDOM_STATE
    )

    # Second split: val vs test.
    relative_test = TEST_RATIO / test_val_ratio
    X_val, X_test, y_val, y_test = _stratified_or_random(
        X_temp, y_temp, test_size=relative_test, random_state=RANDOM_STATE
    )

    # Standardize per feature (fit ONLY on train to avoid data leakage).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, NUM_FEATURES)).reshape(
        -1, SEQUENCE_LENGTH, NUM_FEATURES
    )
    X_val = scaler.transform(X_val.reshape(-1, NUM_FEATURES)).reshape(
        -1, SEQUENCE_LENGTH, NUM_FEATURES
    )
    X_test = scaler.transform(X_test.reshape(-1, NUM_FEATURES)).reshape(
        -1, SEQUENCE_LENGTH, NUM_FEATURES
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, label_names, scaler
