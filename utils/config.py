"""Shared configuration constants for the sign-language pipeline.

Centralizing constants here keeps the capture and training steps in sync:
a sequence must use the same length and feature format everywhere.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths.
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models"

# Sequence format.
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 2  # x, y per landmark.

# Hand detection.
NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Class labels (A-Z in alphabetical order).
LABELS = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
