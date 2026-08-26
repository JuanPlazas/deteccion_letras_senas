"""Wrapper around the MediaPipe HandLandmarker task.

Handles model download, hand detection and landmark extraction for the
sign-language dataset capture pipeline.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from numpy.typing import NDArray

from utils import config

# Hand landmark connection pairs as defined in the MediaPipe documentation:
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

COLOR_CONNECTIONS = (0, 255, 0)
COLOR_DOTS = (255, 0, 0)
COLOR_BBOX = (0, 0, 255)


class HandsDetector:
    """Detect hands on a frame and extract wrist-normalized landmarks.

    Attributes:
        hands_detected: Whether at least one hand was found in the last frame.
        landmarks_px: Pixel coordinates ``[id, x, y]`` for every landmark.
        bbox: ``[x_min, y_min, x_max, y_max]`` padded hand bounding box.
    """

    def __init__(self, confidence: float = config.MIN_DETECTION_CONFIDENCE) -> None:
        """Initialize the detector, downloading the model if it is missing.

        Args:
            confidence: Minimum confidence for hand detection (0-1).

        Raises:
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        self.hand_max = config.NUM_HANDS  # Detect only the first hand.
        self.hands_detected = False
        self.conf_detected = confidence
        self.conf_seg = config.MIN_TRACKING_CONFIDENCE  # Minimum tracking confidence.
        self.results: vision.HandLandmarkerResult | None = None
        self.landmarks_px: list[list[int]] = []
        self.bbox: list[float] = []
        self.model_dir = config.MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / MODEL_FILENAME

        if not model_path.exists() or model_path.stat().st_size == 0:
            self._download_model(model_path)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.hand_max,
            min_hand_detection_confidence=self.conf_detected,
            min_tracking_confidence=self.conf_seg,
        )
        try:
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize HandLandmarker from {model_path}. "
                "Delete the file and run again to re-download it."
            ) from exc

    def _download_model(self, model_path: Path) -> None:
        """Download the HandLandmarker model into the given path.

        Args:
            model_path: Destination path for the downloaded ``.task`` file.

        Raises:
            RuntimeError: If the download fails.
        """
        print(f"Downloading HandLandmarker model to {model_path}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to download the HandLandmarker model from {MODEL_URL}"
            ) from exc
        print("Model downloaded.")

    def detect(
        self,
        frame: NDArray[np.uint8],
        margin: int = 40,
        draw_connections: bool = True,
        draw_bbox: bool = True,
    ) -> NDArray[np.uint8]:
        """Detect the hand on a BGR frame and draw the requested overlays.

        The overlays are drawn in place on ``frame``, which is also returned.

        Args:
            frame: BGR image as read by OpenCV.
            margin: Padding in pixels around the hand bounding box.
            draw_connections: Draw the hand skeleton lines and landmark dots.
            draw_bbox: Draw the hand bounding box.

        Returns:
            The input frame with the overlays drawn on it.
        """
        self.landmarks_px = []
        self.bbox = []

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.detector.detect(mp_image)
        self.hands_detected = bool(self.results.hand_landmarks)

        if not self.hands_detected:
            return frame

        hand_landmarks = self.results.hand_landmarks[0]
        h, w = frame.shape[:2]
        for id, lm in enumerate(hand_landmarks):
            self.landmarks_px.append([id, int(lm.x * w), int(lm.y * h)])

        xs = [pt[1] for pt in self.landmarks_px]
        ys = [pt[2] for pt in self.landmarks_px]

        self.bbox = [
            min(xs) - margin,
            min(ys) - margin,
            max(xs) + margin,
            max(ys) + margin,
        ]

        if draw_connections:
            for conn in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    (xs[conn[0]], ys[conn[0]]),
                    (xs[conn[1]], ys[conn[1]]),
                    COLOR_CONNECTIONS,
                    2,
                )
            for pt in self.landmarks_px:
                cv2.circle(frame, (pt[1], pt[2]), 4, COLOR_DOTS, cv2.FILLED)

        if draw_bbox:
            x_min, y_min, x_max, y_max = self.bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_BBOX, 2)

        return frame

    def get_landmarks_normalized(self) -> list[float]:
        """Return the wrist-normalized, scale-invariant (x, y) landmarks.

        Landmark 0 (the wrist) is used as the origin, so the output is invariant
        to the hand position in the frame. The x-axis is mirrored for left hands
        so the model learns a single pattern regardless of which hand is used.
        Coordinates are then divided by the wrist-to-middle-finger-MCP distance
        (landmark 9), making the output invariant to the distance from the camera.

        Returns:
            A flat list of ``config.NUM_FEATURES`` floats.
        """
        if not self.hands_detected or self.results is None:
            return [0.0] * config.NUM_FEATURES

        hand_landmarks = self.results.hand_landmarks[0]
        # handedness is List[List[Category]]; the first Category holds the index.
        handedness = self.results.handedness[0][0].index  # 0 right, 1 left.

        # Landmark 0 is the wrist.
        wrist_x = hand_landmarks[0].x
        wrist_y = hand_landmarks[0].y

        coords: list[tuple[float, float]] = []
        for lm in hand_landmarks:
            if handedness == 0:  # Right hand.
                x = lm.x - wrist_x
            else:  # Left hand -> mirror the x-axis.
                x = wrist_x - lm.x
            coords.append((x, lm.y - wrist_y))

        # Landmark 9 is the middle-finger MCP: reference distance for scaling.
        ref_x, ref_y = coords[9]
        scale = math.hypot(ref_x, ref_y)
        if scale > 0:
            coords = [(x / scale, y / scale) for x, y in coords]

        flattened: list[float] = []
        for x, y in coords:
            flattened.extend([x, y])
        return flattened
