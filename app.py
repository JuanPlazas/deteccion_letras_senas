"""Real-time sign-language text transcription interface.

Designed for recording clean demo videos (e.g. spelling words like "HOLA"):
- Hides hand skeleton overlays for a natural presentation.
- Accumulates recognized letters into a subtitle bar at the bottom.
- Ignores consecutive duplicate predictions so holding a sign doesn't repeat letters.
- Press 'X' to clear accumulated text, Space for spaces, Backspace to delete.
- Press 'Q' or ESC to exit.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient

from training.config import EXPERIMENT_NAME, MLFLOW_URI
from utils import config
from utils.hands_detector import HandsDetector

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Keybindings
KEY_EXIT = ord("q")
KEY_ESC = 27
KEY_CLEAR = ord("x")
KEY_SPACE = ord(" ")
KEY_BACKSPACE = 8

# Gesture logic settings
MAX_MISSING_FRAMES = 6  # Lost-hand frames tolerance before resetting sequence buffer
MIN_PREDICTION_CONFIDENCE = 0.55  # Minimum confidence to accept a letter

# Visual Palette (BGR)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_MUTED = (180, 180, 180)
COLOR_ACCENT = (0, 230, 118)  # Bright Green / Emerald
COLOR_BAR_BG = (50, 50, 50)
COLOR_BAR_FILL = (0, 200, 83)
COLOR_BANNER_BG = (20, 20, 24)


class LetterPredictor:
    """Loads the newest trained model and scaler from MLflow."""

    def __init__(self) -> None:
        """Resolve the newest finished run and load model, scaler and labels.

        Raises:
            RuntimeError: If experiment or finished training run is missing.
        """
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment {EXPERIMENT_NAME!r} not found in {MLFLOW_URI}. "
                "Train a model first with 'uv run python -m training.train'."
            )

        run = self._latest_finished_run(client, experiment.experiment_id)
        if run is None:
            raise RuntimeError(
                f"No finished training run found in experiment {EXPERIMENT_NAME!r}. "
                "Run 'uv run python -m training.train' first."
            )
        self.run_id = run.info.run_id

        self.model = mlflow.pyfunc.load_model(f"runs:/{self.run_id}/model")
        self.scaler = self._load_scaler(client)
        self.label_map = self._load_label_map(run)

        print(f"Model successfully loaded from run: {self.run_id}")
        print(f"Available classes ({len(self.label_map)}): {list(self.label_map.values())}")

    @staticmethod
    def _latest_finished_run(client: MlflowClient, experiment_id: str):
        """Return the most recent run whose status is FINISHED."""
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
        )
        return next((run for run in runs if run.info.status == "FINISHED"), None)

    def _load_scaler(self, client: MlflowClient):
        """Download the fitted scaler artifact from the run."""
        dest = Path(tempfile.mkdtemp(prefix="mlflow-preproc-"))
        client.download_artifacts(self.run_id, "preprocessing", dst_path=str(dest))
        scaler_file = next((dest / "preprocessing").glob("*.joblib"))
        return joblib.load(scaler_file)

    @staticmethod
    def _load_label_map(run) -> dict[int, str]:
        """Build index -> letter mapping from the run params."""
        labels_param = run.data.params.get("labels")
        if labels_param:
            return {i: letter for i, letter in enumerate(labels_param.split(","))}
        raise RuntimeError(
            f"Run {run.info.run_id} does not contain 'labels' parameter. "
            "Please re-train using training/train.py."
        )

    def predict(self, sequence: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Standardize sequence and return prediction.

        Args:
            sequence: Array of shape (SEQUENCE_LENGTH, NUM_FEATURES).

        Returns:
            Tuple of (predicted_letter, confidence_score, class_probabilities).
        """
        flat = np.asarray(sequence, dtype=np.float32).reshape(-1, config.NUM_FEATURES)
        standardized = self.scaler.transform(flat).reshape(
            1, config.SEQUENCE_LENGTH, config.NUM_FEATURES
        )
        probabilities = np.asarray(self.model.predict(standardized)).reshape(-1)
        top_index = int(np.argmax(probabilities))
        letter = str(self.label_map.get(top_index, "?"))
        return letter, float(probabilities[top_index]), probabilities


def draw_modern_banner(
    frame: np.ndarray,
    accumulated_text: list[str],
    last_detected: tuple[str, float] | None,
    buffer_len: int,
    target_len: int,
    hand_detected: bool,
) -> None:
    """Render a clean bottom subtitle bar and minimal top progress bar."""
    h, w = frame.shape[:2]

    # --- 1. Top subtle progress bar ---
    bar_h = 6
    bar_y = 0
    progress = min(1.0, max(0.0, buffer_len / target_len))
    # Background strip
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), COLOR_BAR_BG, -1)
    if progress > 0:
        fill_w = int(w * progress)
        fill_color = COLOR_BAR_FILL if hand_detected else (100, 100, 100)
        cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), fill_color, -1)

    # --- 2. Bottom Subtitle Banner ---
    banner_height = 110
    banner_top = h - banner_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_top), (w, h), COLOR_BANNER_BG, -1)
    # Blend for semi-transparent elegant look (alpha = 0.75)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Banner accent top border line
    cv2.line(frame, (0, banner_top), (w, banner_top), (60, 60, 70), 1)

    # Display text
    display_str = "".join(accumulated_text)
    if not display_str:
        placeholder = "Realiza una sena con la mano..." if hand_detected else "Esperando mano..."
        cv2.putText(
            frame,
            placeholder,
            (25, banner_top + 52),
            FONT,
            0.9,
            COLOR_TEXT_MUTED,
            2,
            cv2.LINE_AA,
        )
    else:
        # Display letters with spacious tracking for readability
        spaced_text = " ".join(accumulated_text)
        cv2.putText(
            frame,
            spaced_text,
            (25, banner_top + 55),
            FONT,
            1.2,
            COLOR_TEXT_PRIMARY,
            3,
            cv2.LINE_AA,
        )

    # Last recognized badge (on the right of banner)
    if last_detected is not None:
        letter, conf = last_detected
        badge_text = f"Ultima: {letter} ({conf:.0%})"
        (tw, _), _ = cv2.getTextSize(badge_text, FONT, 0.65, 2)
        cv2.putText(
            frame,
            badge_text,
            (w - tw - 25, banner_top + 50),
            FONT,
            0.65,
            COLOR_ACCENT,
            2,
            cv2.LINE_AA,
        )

    # Bottom helper instructions
    instructions = "[X] Borrar todo   [ESPACIO] Separar   [RETROCESO] Borrar 1   [Q] Salir"
    cv2.putText(
        frame,
        instructions,
        (25, h - 18),
        FONT,
        0.45,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    """Build and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sign Language Live Transcription (Clean Video Interface)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=MIN_PREDICTION_CONFIDENCE,
        help=f"Minimum prediction confidence threshold (default: {MIN_PREDICTION_CONFIDENCE})",
    )
    return parser.parse_args()


def main() -> None:
    """Main transcription loop."""
    args = parse_args()

    # Enforce execution from the project root
    if Path.cwd().resolve() != config.PROJECT_ROOT:
        print(f"ERROR: run this script from the project root: {config.PROJECT_ROOT}")
        sys.exit(1)

    predictor = LetterPredictor()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: failed to open camera index {args.camera}")
        return

    # Hands detector instance
    detector = HandsDetector()

    # State variables
    buffer: list[np.ndarray] = []
    accumulated_letters: list[str] = []
    last_detected_info: tuple[str, float] | None = None
    missing_frames = 0

    print("\n--- Live Transcription Started ---")
    print("Controles:")
    print("  'X'       -> Borrar todas las letras")
    print("  'ESPACIO' -> Agregar un espacio")
    print("  'BACKSPACE' -> Borrar la ultima letra")
    print("  'Q' o ESC -> Salir")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame horizontally for natural selfie view
            frame = cv2.flip(frame, 1)

            # Detect hand without drawing skeleton connections or bounding boxes
            detector.detect(frame, draw_connections=False, draw_bbox=False)

            if detector.hands_detected:
                missing_frames = 0
                # Extract normalized scale-invariant hand landmarks
                norm_landmarks = detector.get_landmarks_normalized()
                buffer.append(np.asarray(norm_landmarks, dtype=np.float32))

                # Once the configured sequence length is reached, evaluate model
                if len(buffer) == config.SEQUENCE_LENGTH:
                    letter, confidence, _ = predictor.predict(np.stack(buffer))
                    buffer = []

                    if confidence >= args.min_conf:
                        last_detected_info = (letter, confidence)

                        # Deduplication: do not add if identical to the previous letter
                        if not accumulated_letters or accumulated_letters[-1] != letter:
                            accumulated_letters.append(letter)
                            print(
                                f"+ Letra detectada: {letter} ({confidence:.1%}) -> "
                                f"Texto: {''.join(accumulated_letters)}"
                            )
                        else:
                            print(f"= Letra repetida ignorada: {letter} ({confidence:.1%})")
                    else:
                        print(f"? Confianza baja ignorada: {letter} ({confidence:.1%})")
            else:
                missing_frames += 1
                if missing_frames > MAX_MISSING_FRAMES:
                    buffer = []

            # Draw clean subtitle banner & top progress bar
            draw_modern_banner(
                frame=frame,
                accumulated_text=accumulated_letters,
                last_detected=last_detected_info,
                buffer_len=len(buffer),
                target_len=config.SEQUENCE_LENGTH,
                hand_detected=detector.hands_detected,
            )

            cv2.imshow("Traductor LSE - Demo", frame)

            raw_key = cv2.waitKey(1) & 0xFF
            if raw_key in (KEY_EXIT, ord("Q"), KEY_ESC):
                break
            elif raw_key in (KEY_CLEAR, ord("X")):
                accumulated_letters.clear()
                last_detected_info = None
                print(">> Texto borrado.")
            elif raw_key == KEY_SPACE:
                if accumulated_letters and accumulated_letters[-1] != " ":
                    accumulated_letters.append(" ")
                    print(">> Espacio agregado.")
            elif raw_key == KEY_BACKSPACE:
                if accumulated_letters:
                    removed = accumulated_letters.pop()
                    print(f">> Borrada letra '{removed}'.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nTraduccion final:", "".join(accumulated_letters))


if __name__ == "__main__":
    main()
