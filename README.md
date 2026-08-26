# Sign Language Letter Detection

Hand-detection pipeline for building a sign language dataset and training a sequence-classification model (LSTM) for real-time letter recognition.

## Roadmap

1. **Step 1 — Data capture:** Record normalized hand-landmark sequences with MediaPipe and OpenCV.
2. **Step 2 — Training:** Train an LSTM (Keras) with MLflow experiment tracking.
3. **Step 3 — Inference:** Real-time letter prediction with OpenCV.
4. **Step 4 — Deployment:** Hugging Face Spaces app with Gradio.

## Project Structure

```text
deteccion_letras_senas/
├── data/                       # Generated when capturing the dataset
├── models/                     # Auto-downloaded model files
├── scripts/
│   └── capture_data.py         # Step 1: capture landmark sequences (.npy)
├── utils/
│   ├── config.py               # Shared constants and paths
│   └── hands_detector.py       # MediaPipe HandLandmarker wrapper
├── .pre-commit-config.yaml     # Git hooks: black + ruff
├── .gitignore
├── .python-version
├── pyproject.toml              # uv-based project configuration
├── uv.lock                     # Locked dependency versions
└── README.md
```

Captured data lives in `data/sequences/<LETTER>/<LETTER>_NNNN.npy`; each file has shape `(30, 42)`: 30 frames × 21 landmarks × 2 x/y coordinates normalized relative to the wrist.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync
```

## Usage

Run every command from the project root. `capture_data.py` validates the working directory and exits with an error otherwise.

### Step 1 — Capture sequences for a letter

```bash
uv run scripts/capture_data.py --letter S
```

Options:

- `--letter` — letter to capture (A–Z), required.
- `--camera` — camera index (default: `0`).
- `--length` — frames per sequence (default: `30`).
- `--confidence` — minimum hand-detection confidence (default: `0.7`).

Keys during capture:

- **'S'** — record one sequence (30 landmark frames).
- **'Q'** — quit.

The MediaPipe HandLandmarker model is downloaded automatically to `models/hand_landmarker.task` on first run.

## Development

Code quality is enforced with [Black](https://black.readthedocs.io/) (formatting) and [Ruff](https://docs.astral.sh/ruff/) (linting). Both are installed as dev dependencies and wired into git hooks via pre-commit, so they run automatically on every commit.

```bash
uv run black .              # format the code
uv run ruff check .         # lint the code
uv run ruff check --fix .   # auto-fix lint issues
uv run pre-commit install   # enable the git hooks
```

## Design Notes

- Landmarks are normalized by subtracting the wrist (landmark 0) and dividing by the wrist-to-middle-finger distance, making the model invariant to the hand's position in the frame and its distance from the camera.
- X-coordinates are mirrored for left hands, so the model learns a single pattern regardless of which hand is used.
- If the hand is lost mid-recording, the sequence is discarded to keep the dataset clean.
