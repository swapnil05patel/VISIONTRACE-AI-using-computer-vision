#!/usr/bin/env python3.12
"""Prepare and generate high-fidelity Re-ID dataset fixtures for Market-1501, MSMT17, and MARS.

Builds multi-identity, multi-camera sets with realistic 256x128 person images
compatible with the VisionTrace training, evaluation, and import pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXTURES = DATA / "dataset-fixtures"
SAMPLE_PERSON = DATA / "samples" / "authorized-person.jpg"


def get_base_crops(sample_path: Path) -> list[np.ndarray]:
    if not sample_path.is_file():
        # Fallback synthetic silhouettes
        crops = []
        for i in range(8):
            canvas = np.zeros((256, 128, 3), dtype=np.uint8)
            color = [(50 * i + 80) % 255, (70 * i + 40) % 255, (90 * i + 100) % 255]
            cv2.rectangle(canvas, (32, 40), (96, 220), color, -1)
            cv2.circle(canvas, (64, 40), 20, (200, 180, 160), -1)
            crops.append(canvas)
        return crops

    img = cv2.imread(str(sample_path))
    from inference import detect_people
    dets = detect_people(img, require_verification=False)
    crops = []
    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 > 20 and y2 - y1 > 40:
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            crops.append(crop)
    if not crops:
        crop = cv2.resize(img, (128, 256))
        crops.append(crop)
    return crops


def apply_jitter(crop: np.ndarray, factor: float, cam_tint: tuple[float, float, float]) -> np.ndarray:
    out = crop.astype(np.float32)
    # Brightness & contrast
    out = out * (0.85 + 0.3 * factor)
    # Camera color tint
    out[:, :, 0] *= cam_tint[0]
    out[:, :, 1] *= cam_tint[1]
    out[:, :, 2] *= cam_tint[2]
    # Small noise
    noise = np.random.normal(0, 3, out.shape)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)
    return out


def build_market1501(root: Path, base_crops: list[np.ndarray], num_ids: int = 10) -> int:
    train_dir = root / "bounding_box_train"
    gallery_dir = root / "bounding_box_test"
    query_dir = root / "query"
    for d in (train_dir, gallery_dir, query_dir):
        d.mkdir(parents=True, exist_ok=True)

    camera_tints = [
        (1.0, 1.0, 1.0),
        (0.95, 1.05, 0.98),
        (1.02, 0.96, 1.04),
        (0.98, 0.98, 1.05),
        (1.05, 1.0, 0.95),
        (1.0, 0.97, 1.02),
    ]

    count = 0
    for idx in range(1, num_ids + 1):
        pid = f"{idx:04d}"
        crop = base_crops[(idx - 1) % len(base_crops)]

        # Training samples (cameras 1 to 4)
        for cam in range(1, 4):
            for seq in range(1, 3):
                frame = 100 * seq + cam * 10
                filename = f"{pid}_c{cam}s{seq}_{frame:06d}_00.jpg"
                img = apply_jitter(crop, (cam + seq) / 10.0, camera_tints[cam - 1])
                cv2.imwrite(str(train_dir / filename), img)
                count += 1

        # Gallery samples (cameras 4 to 6)
        for cam in [4, 5]:
            filename = f"{pid}_c{cam}s1_{300 + cam:06d}_00.jpg"
            img = apply_jitter(crop, 0.5 + cam / 10.0, camera_tints[cam - 1])
            cv2.imwrite(str(gallery_dir / filename), img)
            count += 1

        # Query samples (camera 6)
        query_filename = f"{pid}_c6s1_000999_00.jpg"
        img = apply_jitter(crop, 0.65, camera_tints[5])
        cv2.imwrite(str(query_dir / query_filename), img)
        count += 1

    return count


def build_msmt17(root: Path, base_crops: list[np.ndarray], num_ids: int = 10) -> int:
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    list_train_lines = []
    list_query_lines = []
    list_gallery_lines = []
    count = 0

    for idx in range(1, num_ids + 1):
        pid = f"{idx:04d}"
        crop = base_crops[(idx - 1) % len(base_crops)]

        id_train_dir = train_dir / pid
        id_test_dir = test_dir / pid
        id_train_dir.mkdir(parents=True, exist_ok=True)
        id_test_dir.mkdir(parents=True, exist_ok=True)

        for cam in [1, 2, 3]:
            rel_train = f"train/{pid}/{pid}_c{cam}_001.jpg"
            img = apply_jitter(crop, cam * 0.2, (1.0, 1.0, 1.0))
            cv2.imwrite(str(root / rel_train), img)
            list_train_lines.append(f"{rel_train} {pid}\n")
            count += 1

        # Query & gallery
        rel_query = f"test/{pid}/{pid}_c4_001.jpg"
        img_q = apply_jitter(crop, 0.8, (0.98, 1.02, 1.0))
        cv2.imwrite(str(root / rel_query), img_q)
        list_query_lines.append(f"{rel_query} {pid}\n")

        rel_gallery = f"test/{pid}/{pid}_c5_001.jpg"
        img_g = apply_jitter(crop, 0.85, (1.02, 0.98, 1.0))
        cv2.imwrite(str(root / rel_gallery), img_g)
        list_gallery_lines.append(f"{rel_gallery} {pid}\n")
        count += 2

    (root / "list_train.txt").write_text("".join(list_train_lines))
    (root / "list_query.txt").write_text("".join(list_query_lines))
    (root / "list_gallery.txt").write_text("".join(list_gallery_lines))
    return count


def build_mars(root: Path, base_crops: list[np.ndarray], num_ids: int = 10) -> int:
    train_dir = root / "bbox_train"
    test_dir = root / "bbox_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in range(1, num_ids + 1):
        pid = f"{idx:04d}"
        crop = base_crops[(idx - 1) % len(base_crops)]

        id_train = train_dir / pid
        id_test = test_dir / pid
        id_train.mkdir(parents=True, exist_ok=True)
        id_test.mkdir(parents=True, exist_ok=True)

        # Multiple tracklet frames in train
        for cam in [1, 2]:
            for frame in [1, 2, 3]:
                fname = f"{pid}C{cam}T0001F{frame:03d}.jpg"
                img = apply_jitter(crop, (cam + frame) * 0.1, (1.0, 1.0, 1.0))
                cv2.imwrite(str(id_train / fname), img)
                count += 1

        # Test tracklet frames
        for frame in [1, 2]:
            fname = f"{pid}C3T0002F{frame:03d}.jpg"
            img = apply_jitter(crop, frame * 0.2, (0.97, 1.0, 1.03))
            cv2.imwrite(str(id_test / fname), img)
            count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset fixtures for Market-1501, MSMT17, and MARS.")
    parser.add_argument("--base", default=str(FIXTURES))
    parser.add_argument("--num-identities", type=int, default=12)
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting base person crops from sample...")
    base_crops = get_base_crops(SAMPLE_PERSON)
    print(f"Extracted {len(base_crops)} base crops.")

    m1501_dir = base_dir / "market1501"
    msmt_dir = base_dir / "msmt17"
    mars_dir = base_dir / "mars"

    print("Generating Market-1501 fixture dataset...")
    n_m = build_market1501(m1501_dir, base_crops, args.num_identities)
    print(f"  Market-1501 generated: {n_m} images in {m1501_dir}")

    print("Generating MSMT17 fixture dataset...")
    n_ms = build_msmt17(msmt_dir, base_crops, args.num_identities)
    print(f"  MSMT17 generated: {n_ms} images in {msmt_dir}")

    print("Generating MARS fixture dataset...")
    n_ma = build_mars(mars_dir, base_crops, args.num_identities)
    print(f"  MARS generated: {n_ma} images in {mars_dir}")

    print("Datasets preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
