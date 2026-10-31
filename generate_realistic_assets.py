"""Download and prepare real public demo assets for VisionTrace.

This script intentionally uses known public sample media from OpenCV and Ultralytics so
that the repo's YOLO and Re-ID pipeline sees realistic, detectable image/video content.
It creates the sample assets under data/test images/ using naming conventions already used
by the project and is safe for local demo/testing.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("data/test images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSET_URLS = {
    "test_video_vtest.avi": [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
    ],
    "query_messi5.jpg": [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
    ],
    "query_bus.jpg": [
        "https://raw.githubusercontent.com/ultralytics/assets/main/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/assets/main/images/bus.jpg",
    ],
    "query_zidane.jpg": [
        "https://raw.githubusercontent.com/ultralytics/assets/main/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/assets/main/images/zidane.jpg",
    ],
}


def download(urls: list[str] | str, destination: Path) -> Path:
    if destination.exists():
        print(f"  [exists] {destination}")
        return destination

    candidates = [urls] if isinstance(urls, str) else urls
    last_error: Exception | None = None
    for url in candidates:
        try:
            print(f"  [download] {destination.name} from {url}")
            urllib.request.urlretrieve(url, destination)
            return destination
        except Exception as exc:  # pragma: no cover - network fallback path
            last_error = exc
            if destination.exists():
                return destination
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No working download URL found for {destination.name}")


def save_copy(source: Path, target_name: str) -> Path:
    target = OUT_DIR / target_name
    if source.exists():
        target.write_bytes(source.read_bytes())
    print(f"  [copy] {target}")
    return target


def build_mp4_from_avi(avi_path: Path, output_name: str) -> Path:
    output = OUT_DIR / output_name
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {avi_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {output}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"  [video] {output} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames)")
    return output


def crop_center(image: np.ndarray, scale: float = 0.75) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = int((w - w * scale) / 2)
    y0 = int((h - h * scale) / 2)
    x1 = int(w - x0)
    y1 = int(h - y0)
    return image[y0:y1, x0:x1]


def create_alias_images() -> None:
    messi = cv2.imread(str(OUT_DIR / "query_messi5.jpg"), cv2.IMREAD_COLOR)
    bus = cv2.imread(str(OUT_DIR / "query_bus.jpg"), cv2.IMREAD_COLOR)
    zidane = cv2.imread(str(OUT_DIR / "query_zidane.jpg"), cv2.IMREAD_COLOR)

    if messi is not None:
        cv2.imwrite(str(OUT_DIR / "query_person_A.jpg"), messi, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("  [image] query_person_A.jpg")
    if zidane is not None:
        cv2.imwrite(str(OUT_DIR / "query_person_B.jpg"), zidane, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("  [image] query_person_B.jpg")
    if bus is not None:
        bus_crop = crop_center(bus, 0.8)
        cv2.imwrite(str(OUT_DIR / "query_person_C.jpg"), bus_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT_DIR / "query_crowd_scene.jpg"), bus, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("  [image] query_person_C.jpg")
        print("  [image] query_crowd_scene.jpg")
    if messi is not None:
        cv2.imwrite(str(OUT_DIR / "query_multi_person.jpg"), messi, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("  [image] query_multi_person.jpg")
    if zidane is not None:
        cv2.imwrite(str(OUT_DIR / "query_reid_target.jpg"), zidane, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("  [image] query_reid_target.jpg")


def main() -> None:
    print("=== VisionTrace — loading a small real demo set ===")
    downloaded: list[Path] = []
    for filename, urls in ASSET_URLS.items():
        downloaded.append(download(urls, OUT_DIR / filename))

    avi_path = next(path for path in downloaded if path.name.endswith(".avi"))
    build_mp4_from_avi(avi_path, "test_video_person_A.mp4")
    create_alias_images()

    # Keep a couple of the repo’s common names for compatibility with the demo/test flow.
    save_copy(OUT_DIR / "query_person_A.jpg", "query_cam_A_person.jpg")
    save_copy(OUT_DIR / "query_person_B.jpg", "query_cam_B_person.jpg")
    print("\n=== Done ===")
    print(f"Assets are stored in: {OUT_DIR}")


if __name__ == "__main__":
    main()
