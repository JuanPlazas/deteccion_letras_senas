"""Interactive dataset capture for sign-language letters.

Records wrist-normalized hand-landmark sequences from the webcam and stores
them as ``.npy`` files under ``data/sequences/<LETTER>``.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from utils import config
from utils.hands_detector import HandsDetector

KEY_EXIT = ord("q")
KEY_SAVE = ord("s")


def letter_type(value: str) -> str:
    """Validate and normalize the ``--letter`` argument.

    Args:
        value: Raw value passed on the command line.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a single letter A-Z.

    Returns:
        The uppercase letter.
    """
    letter = value.upper()
    if letter not in config.LABELS:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid letter; expected a single character A-Z"
        )
    return letter


def parse_args() -> argparse.Namespace:
    """Build and parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Signal Language Dataset Creator")
    parser.add_argument(
        "--letter", required=True, type=letter_type, help="Signal letter (A-Z) to capture"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--length",
        type=int,
        default=config.SEQUENCE_LENGTH,
        help=f"Frames per sequence (default: {config.SEQUENCE_LENGTH})",
    )
    return parser.parse_args()


def main() -> None:
    """Capture landmark sequences for a letter using the webcam."""
    args = parse_args()

    # Enforce execution from the project root so that paths resolve correctly.
    if Path.cwd().resolve() != config.PROJECT_ROOT:
        print(f"ERROR: run this script from the project root: {config.PROJECT_ROOT}")
        sys.exit(1)

    data_dir = config.SEQUENCES_DIR
    letter = args.letter
    save_dir = data_dir / letter
    save_dir.mkdir(parents=True, exist_ok=True)
    count_file = len(os.listdir(save_dir))  # Existing sequences for this letter.

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: failed to open camera index {args.camera}")
        return

    count = 0
    sequence_data: list[list[float]] = []
    recording = False

    try:
        detector = HandsDetector()

        print(f"Capturing sequences for '{letter}'")
        print("Commands: 'S' to capture sequence | 'Q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detector.detect(frame)

            # UI overlays.
            cv2.putText(
                frame,
                f"Letter: {letter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame, f"Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            if recording:
                cv2.putText(
                    frame,
                    f"RECORDING... {len(sequence_data)}/{args.length}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                if detector.hands_detected:
                    sequence_data.append(detector.get_landmarks_normalized())
                else:
                    # Abort the sequence if the hand is lost mid-recording.
                    print("Hand lost. Sequence discarded.")
                    sequence_data = []
                    recording = False

                if len(sequence_data) >= args.length:
                    # Save the completed sequence.
                    file_path = save_dir / f"{letter}_{count_file:04d}.npy"
                    np.save(file_path, np.array(sequence_data))
                    count_file += 1
                    count += 1
                    sequence_data = []
                    recording = False
                    print(f"Saved sequence {count}")
            elif detector.hands_detected:
                cv2.putText(
                    frame, "HAND DETECTED", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

            cv2.imshow("Sequence Creator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_EXIT:
                break

            # Start recording a new sequence.
            if key == KEY_SAVE and not recording:
                recording = True
                sequence_data = []
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Captured {count} sequences.")


if __name__ == "__main__":
    main()
