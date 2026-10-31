# Driver Drowsiness Detection

A real-time webcam prototype that estimates eye openness from MediaPipe face landmarks. It smooths the eye aspect ratio (EAR) across five frames and shows an alert when both eyes appear closed for a configurable duration. The webcam feed is processed locally and is not saved or transmitted.

## Setup

Use Python 3.10 or newer, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Start the detector:

```bash
python driver_drowsiness_detection.py
```

On first run, the script downloads Google's MediaPipe Face Landmarker task model to this directory. For an offline setup, download the model separately and pass its path with `--model-path`.

## Options

```text
--camera N              Camera device index (default: 0)
--ear-threshold VALUE   Closed-eye EAR threshold (default: 0.21)
--closed-seconds VALUE  Seconds of closure before alert (default: 1.5)
--alarm-interval VALUE  Repeat alert interval in seconds (default: 1.5)
--model-path PATH       Existing Face Landmarker .task file
--no-mirror             Disable the mirrored preview
```

Press **Q** or **Esc** to close the camera window. Brief blinks keep the eye outlines green and do not show an alert. After the eyes remain closed continuously for the configured delay (1.5 seconds by default), the outlines and warning turn red and the sound alarm starts. The alarm repeats at the `--alarm-interval` (1.5 seconds by default). Adjust `--closed-seconds` to change the delay. If the detector misses closed eyes or reports false closures, adjust `--ear-threshold` for the camera, lighting, and driver position. Windows uses its system alert beep; macOS plays the Glass alert sound, with a terminal bell fallback. Other systems use the terminal bell.

## How it works

The app computes an EAR value from six landmarks around each eye. Lower values indicate a narrower eye opening. A five-frame median reduces flicker, and a monotonic timer checks how long the smoothed value stays below the threshold. The app uses MediaPipe Tasks' video mode for webcam frames.

This is a learning/demo project, not a certified driver monitoring or vehicle safety system. Lighting, glasses, camera placement, face angle, and individual differences can affect detection. It can miss drowsiness and can alert incorrectly; never rely on it as the sole safety measure while driving.
