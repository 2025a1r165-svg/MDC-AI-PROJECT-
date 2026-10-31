# Driver Drowsiness Detection

## Browser app

Open the live browser demo at https://2025a1r165-svg.github.io/MDC-AI-PROJECT-/ after GitHub Pages is enabled. Press Start camera and allow camera access. Detection runs in your browser; the app does not record or send camera frames. Brief blinks stay green. The default drowsiness alert starts after 1.5 seconds of continuous eye closure.

## Python webcam app

Use Python 3.10 or newer. From this folder, create an environment and install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Start the detector:

    python driver_drowsiness_detection.py

The MediaPipe model downloads automatically if it is not present. For offline use, the repository includes face_landmarker.task. Press Q or Esc to close the camera window.

## Options

- --camera N: webcam device index (default 0)
- --ear-threshold VALUE: closed-eye threshold (default 0.21)
- --closed-seconds VALUE: closure duration before alert (default 1.5 seconds)
- --alarm-interval VALUE: repeat alert interval (default 1.5 seconds)
- --model-path PATH: choose a model file
- --no-mirror: disable the mirrored camera preview

## How it works

The app computes the eye aspect ratio from MediaPipe face landmarks and smooths values across five frames. A sustained closure triggers the visual and sound alert; brief blinks do not.

This is a learning prototype, not a certified vehicle safety system. Lighting, glasses, camera placement, face angle, and individual differences can affect detection. It can miss drowsiness or alert incorrectly; never rely on it as the sole safety measure while driving.
