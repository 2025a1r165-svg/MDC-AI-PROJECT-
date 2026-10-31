"""Real-time driver drowsiness demo using eye aspect ratio (EAR).

The webcam feed is processed locally. No frames are saved or transmitted.
This is a prototype and must not be used as a vehicle safety system.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

# Six-point eye contours in the MediaPipe Face Landmarker mesh.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def download_model(destination: Path) -> Path:
    """Download the official Face Landmarker task model if it is missing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "driver-drowsiness-demo/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as model_file:
                while chunk := response.read(1024 * 1024):
                    model_file.write(chunk)
        if temporary.stat().st_size < 100_000:
            raise RuntimeError("The downloaded model file is unexpectedly small.")
        temporary.replace(destination)
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe face landmark model. Check your "
            "internet connection, or download face_landmarker.task and pass "
            "its location with --model-path."
        ) from exc
    return destination


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Compute EAR for six eye contour points in clockwise order."""
    vertical_a = np.linalg.norm(points[1] - points[5])
    vertical_b = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal <= 1e-6:
        return 0.0
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def landmarks_to_pixels(
    landmarks: Sequence[object], width: int, height: int
) -> np.ndarray:
    return np.asarray(
        [(landmark.x * width, landmark.y * height) for landmark in landmarks],
        dtype=np.float32,
    )


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.7,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        frame,
        (x - 6, y - text_height - 7),
        (x + text_width + 6, y + baseline + 5),
        (20, 20, 20),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def play_alert() -> None:
    """Make a short platform-appropriate alert without blocking the video loop."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif sys.platform == "darwin":
            sound_file = Path("/System/Library/Sounds/Glass.aiff")
            try:
                subprocess.run(
                    ["/usr/bin/afplay", str(sound_file)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
            except (OSError, subprocess.SubprocessError):
                # Keep an audible fallback if macOS playback is unavailable.
                print("\a", end="", flush=True)
        else:
            # The terminal bell is the dependency-free fallback on other systems.
            print("\a", end="", flush=True)
    except Exception:
        # A failed sound device should not stop the visual alert.
        pass


def start_alert() -> None:
    Thread(target=play_alert, daemon=True).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warn when the driver's eyes remain closed for too long."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="webcam device index (default: 0)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Face Landmarker .task model; downloaded automatically if missing",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="eye aspect ratio below which eyes count as closed (default: 0.21)",
    )
    parser.add_argument(
        "--closed-seconds",
        type=float,
        default=1.5,
        help="closed-eye duration before the drowsiness alert (default: 1.5)",
    )
    parser.add_argument(
        "--alarm-interval",
        type=float,
        default=1.5,
        help="seconds between repeated alerts while eyes remain closed",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="show the camera image without horizontal mirroring",
    )
    args = parser.parse_args()
    if not 0.05 <= args.ear_threshold <= 0.5:
        parser.error("--ear-threshold must be between 0.05 and 0.5")
    if args.closed_seconds <= 0 or args.alarm_interval <= 0:
        parser.error("--closed-seconds and --alarm-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        print(f"Downloading face landmark model to {args.model_path}...")
        download_model(args.model_path)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try another index with --camera."
        )

    closed_since: float | None = None
    last_alert_at = 0.0
    ear_samples: deque[float] = deque(maxlen=5)
    start_time = time.monotonic()
    previous_timestamp_ms = -1
    frame_count = 0
    fps = 0.0
    fps_window_start = start_time

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames.")

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = max(
                    int((time.monotonic() - start_time) * 1000),
                    previous_timestamp_ms + 1,
                )
                previous_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)
                now = time.monotonic()

                if result.face_landmarks:
                    landmarks = landmarks_to_pixels(
                        result.face_landmarks[0], width, height
                    )
                    left_ear = eye_aspect_ratio(landmarks[list(LEFT_EYE)])
                    right_ear = eye_aspect_ratio(landmarks[list(RIGHT_EYE)])
                    ear = (left_ear + right_ear) / 2.0
                    ear_samples.append(ear)
                    smoothed_ear = float(np.median(ear_samples))
                    eyes_closed = smoothed_ear < args.ear_threshold

                    if eyes_closed:
                        if closed_since is None:
                            closed_since = now
                        closed_duration = now - closed_since
                    else:
                        closed_since = None
                        last_alert_at = 0.0
                        closed_duration = 0.0

                    prolonged_closure = closed_duration >= args.closed_seconds

                    # Keep brief blinks green; turn red only after the configured delay.
                    eye_color = (0, 0, 255) if prolonged_closure else (0, 220, 0)
                    for eye_indices in (LEFT_EYE, RIGHT_EYE):
                        contour = landmarks[list(eye_indices)].astype(np.int32)
                        cv2.polylines(frame, [contour], True, eye_color, 2, cv2.LINE_AA)
                        for point in contour:
                            cv2.circle(frame, tuple(point), 2, eye_color, -1)

                    if prolonged_closure:
                        draw_label(
                            frame,
                            f"EYES CLOSED  {closed_duration:.1f}s",
                            (20, 42),
                            (0, 80, 255),
                        )
                        draw_label(
                            frame,
                            "DROWSINESS ALERT - TAKE A BREAK",
                            (20, 86),
                            (0, 0, 255),
                            0.75,
                        )
                        if (
                            last_alert_at == 0.0
                            or now - last_alert_at >= args.alarm_interval
                        ):
                            start_alert()
                            last_alert_at = now
                    else:
                        if not eyes_closed:
                            draw_label(frame, "Eyes open", (20, 42), (0, 220, 0))

                    draw_label(frame, f"EAR: {smoothed_ear:.3f}", (20, height - 22), (255, 255, 255), 0.55)
                else:
                    # Do not count time while the face is not reliably visible.
                    closed_since = None
                    last_alert_at = 0.0
                    ear_samples.clear()
                    draw_label(frame, "Face not detected", (20, 42), (0, 190, 255))

                frame_count += 1
                if now - fps_window_start >= 1.0:
                    fps = frame_count / (now - fps_window_start)
                    frame_count = 0
                    fps_window_start = now
                draw_label(frame, f"FPS: {fps:.0f}", (width - 105, 32), (255, 255, 255), 0.55)
                draw_label(frame, "Press Q to quit", (width - 155, height - 22), (220, 220, 220), 0.5)

                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
