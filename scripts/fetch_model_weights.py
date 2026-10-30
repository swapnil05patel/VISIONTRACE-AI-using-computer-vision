#!/usr/bin/env python3.12
"""Download documented public YOLO and OSNet checkpoints into models/.

This never runs at application start. Review licenses before use:
- YOLO11n: Ultralytics AGPL-3.0, https://github.com/ultralytics/assets
- OSNet x1.0 ImageNet: Kaiyang Zhou deep-person-reid, MIT model zoo
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
PERSON_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
OSNET_URLS = [
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_imagenet.pth",
    "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.2.1/osnet_x1_0_imagenet.pth",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "VisionTrace-weight-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"SHA256 {destination.name}={digest}")


def main() -> int:
    yolo = MODELS / "yolo11n.pt"
    osnet = MODELS / "osnet_x1_0.pth"
    sample = SAMPLES / "authorized-person.jpg"
    if not yolo.is_file():
        download(YOLO_URL, yolo)
    if not osnet.is_file():
        last_error = None
        for url in OSNET_URLS:
            try:
                download(url, osnet)
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                if osnet.exists():
                    osnet.unlink()
        else:
            print(f"Could not download OSNet weights: {last_error}", file=sys.stderr)
            return 1
    if not sample.is_file():
        download(PERSON_IMAGE_URL, sample)
    print("Weight files are ready. Run verify_models.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
