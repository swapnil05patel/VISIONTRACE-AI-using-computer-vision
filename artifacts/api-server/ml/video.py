from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The video file could not be opened.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("The video file did not yield a readable frame.")
        return {
            "fps": fps if fps > 0 else None,
            "frameCount": frame_count,
            "width": width,
            "height": height,
            "durationSeconds": duration if duration > 0 else None,
        }
    finally:
        capture.release()


def iter_sampled_frames(
    path: Path,
    sample_fps: float = 1.0,
    start_frame: int = 0,
) -> Iterator[dict[str, Any]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The video file could not be opened.")
    try:
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or sample_fps
        interval = max(1, int(round(native_fps / sample_fps)))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index >= start_frame and index % interval == 0:
                timestamp = index / native_fps
                yield {
                    "frameNumber": index,
                    "timestampSeconds": float(timestamp),
                    "image": frame,
                }
            index += 1
    finally:
        capture.release()


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode a JPEG frame.")
    return buffer.tobytes()
