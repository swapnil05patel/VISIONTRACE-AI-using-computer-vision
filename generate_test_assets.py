"""
generate_test_assets.py
=======================
Generates additional meaningful test videos and query images for VisionTrace demo.

Uses real person detections from existing test assets to create:
  - test_video_cam_A.mp4   : 2 people, simulates Camera A feed (10s, 5fps)
  - test_video_cam_B.mp4   : 3 people, simulates Camera B feed (10s, 5fps)
  - test_video_cam_C.mp4   : 6 people, simulates crowded Camera C feed (10s, 5fps)
  - query_cam_A_person.jpg : single-person query from cam A scene
  - query_cam_B_person.jpg : single-person query from cam B scene
  - query_reid_target.jpg  : re-id target: person seen in cam A, search in cam B

Run:
    venv\Scripts\python.exe generate_test_assets.py
"""
from __future__ import annotations

import os
import sys
import random
import math

import cv2
import numpy as np

sys.path.insert(0, "artifacts/api-server/ml")
os.environ.setdefault("VISIONTRACE_YOLO_WEIGHTS", "models/yolo11n.pt")
os.environ.setdefault("VISIONTRACE_OSNET_WEIGHTS", "models/osnet_x1_0.pth")
os.environ.setdefault("VISIONTRACE_FAISS_PATH", "artifacts/api-server/data/visiontrace.faiss")
os.environ.setdefault("VISIONTRACE_MODEL_VERIFICATION_FILE", "artifacts/api-server/data/model-verification.json")
os.environ.setdefault("VISIONTRACE_ACTIVE_MODELS_FILE", "artifacts/api-server/data/active-models.json")

from inference import detect_people
from preprocess import decode_image

OUT_DIR = "data/test images"
W, H = 640, 480
FPS = 5
DURATION_S = 10
N_FRAMES = FPS * DURATION_S


# ── helpers ──────────────────────────────────────────────────────────────────

def load_person_crops(video_path: str, max_crops: int = 8) -> list[np.ndarray]:
    """Extract person crops from a video using YOLO."""
    crops: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while len(crops) < max_crops:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % 5 == 0:
            dets = detect_people(frame, require_verification=False)
            for det in dets:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 40 and crop.shape[1] > 20:
                    crops.append(crop)
                    if len(crops) >= max_crops:
                        break
        frame_idx += 1
    cap.release()
    return crops


def make_background(color: tuple[int, int, int] = (60, 60, 60)) -> np.ndarray:
    """Create a plain background frame."""
    bg = np.full((H, W, 3), color, dtype=np.uint8)
    # Add subtle grid lines for visual depth
    for x in range(0, W, 80):
        cv2.line(bg, (x, 0), (x, H), (color[0] + 15, color[1] + 15, color[2] + 15), 1)
    for y in range(0, H, 60):
        cv2.line(bg, (0, y), (W, y), (color[0] + 15, color[1] + 15, color[2] + 15), 1)
    return bg


def place_person(
    frame: np.ndarray,
    crop: np.ndarray,
    cx: float,
    cy: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Paste a person crop onto a frame at (cx, cy) center, scaled."""
    target_h = int(crop.shape[0] * scale)
    target_w = int(crop.shape[1] * scale)
    if target_h < 10 or target_w < 5:
        return frame
    resized = cv2.resize(crop, (target_w, target_h))
    x1 = int(cx - target_w / 2)
    y1 = int(cy - target_h / 2)
    x2 = x1 + target_w
    y2 = y1 + target_h
    # Clip to frame bounds
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    ex1 = max(0, x1)
    ey1 = max(0, y1)
    ex2 = min(frame.shape[1], x2)
    ey2 = min(frame.shape[0], y2)
    sw = ex2 - ex1
    sh = ey2 - ey1
    if sw <= 0 or sh <= 0:
        return frame
    frame[ey1:ey2, ex1:ex2] = resized[sy1:sy1 + sh, sx1:sx1 + sw]
    return frame


def add_timestamp(frame: np.ndarray, t: float, cam_id: str) -> np.ndarray:
    """Overlay camera ID and timestamp."""
    cv2.putText(frame, f"CAM {cam_id}  T={t:.1f}s", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def write_video(path: str, frames: list[np.ndarray], fps: int = FPS) -> None:
    # Try H.264 (avc1) first — browser-compatible. Fall back to mp4v.
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(path, fourcc, fps, (W, H))
        if out.isOpened():
            break
    for f in frames:
        out.write(f)
    out.release()
    size_kb = os.path.getsize(path) // 1024
    print(f"  Wrote {path}  ({len(frames)} frames, {size_kb} KB, codec={fourcc_str})")


def save_query_image(path: str, frame: np.ndarray) -> None:
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    size_kb = os.path.getsize(path) // 1024
    print(f"  Wrote {path}  ({size_kb} KB)")


# ── person motion paths ───────────────────────────────────────────────────────

def linear_path(x0: float, y0: float, x1: float, y1: float, n: int) -> list[tuple[float, float]]:
    return [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)) for i in range(n)]


def oscillate_path(cx: float, cy: float, amp: float, n: int, phase: float = 0.0) -> list[tuple[float, float]]:
    return [(cx + amp * math.sin(2 * math.pi * i / n + phase), cy) for i in range(n)]


# ── main generation ───────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)

    print("Loading person crops from existing test videos...")
    crops_A = load_person_crops(f"{OUT_DIR}/test_video_person_A.mp4", max_crops=6)
    crops_multi = load_person_crops(f"{OUT_DIR}/test_video_multi_person.mp4", max_crops=10)
    crops_mixed = load_person_crops(f"{OUT_DIR}/test_video_mixed_scenes.mp4", max_crops=6)

    if not crops_A or not crops_multi:
        print("ERROR: Could not extract person crops. Check that test videos exist and YOLO works.")
        sys.exit(1)

    print(f"  Crops from person_A: {len(crops_A)}, multi: {len(crops_multi)}, mixed: {len(crops_mixed)}")

    # ── Video 1: Camera A — 2 people walking left-to-right ──────────────────
    print("\nGenerating test_video_cam_A.mp4 (2 people, Camera A)...")
    p1_crop = crops_A[0]
    p2_crop = crops_A[1] if len(crops_A) > 1 else crops_A[0]
    path1 = linear_path(80, 240, 560, 240, N_FRAMES)
    path2 = linear_path(560, 300, 80, 300, N_FRAMES)
    frames_A = []
    for i in range(N_FRAMES):
        bg = make_background((55, 55, 65))
        bg = place_person(bg, p1_crop, path1[i][0], path1[i][1], scale=0.85)
        bg = place_person(bg, p2_crop, path2[i][0], path2[i][1], scale=0.80)
        bg = add_timestamp(bg, i / FPS, "A")
        frames_A.append(bg)
    write_video(f"{OUT_DIR}/test_video_cam_A.mp4", frames_A)

    # ── Video 2: Camera B — 3 people, two walking, one stationary ───────────
    print("Generating test_video_cam_B.mp4 (3 people, Camera B)...")
    p3_crop = crops_multi[0]
    p4_crop = crops_multi[1] if len(crops_multi) > 1 else crops_multi[0]
    p5_crop = crops_multi[2] if len(crops_multi) > 2 else crops_multi[0]
    path3 = linear_path(60, 200, 580, 200, N_FRAMES)
    path4 = oscillate_path(320, 350, 120, N_FRAMES, phase=0.0)
    path5 = [(320, 260)] * N_FRAMES  # stationary
    frames_B = []
    for i in range(N_FRAMES):
        bg = make_background((65, 55, 55))
        bg = place_person(bg, p3_crop, path3[i][0], path3[i][1], scale=0.90)
        bg = place_person(bg, p4_crop, path4[i][0], path4[i][1], scale=0.85)
        bg = place_person(bg, p5_crop, path5[i][0], path5[i][1], scale=0.80)
        bg = add_timestamp(bg, i / FPS, "B")
        frames_B.append(bg)
    write_video(f"{OUT_DIR}/test_video_cam_B.mp4", frames_B)

    # ── Video 3: Camera C — 6 people, crowded scene ─────────────────────────
    print("Generating test_video_cam_C.mp4 (6 people, Camera C)...")
    crowd_crops = (crops_multi + crops_A + crops_mixed)[:6]
    while len(crowd_crops) < 6:
        crowd_crops.append(crowd_crops[0])
    positions = [
        linear_path(100, 160, 540, 160, N_FRAMES),
        linear_path(540, 240, 100, 240, N_FRAMES),
        oscillate_path(200, 320, 80, N_FRAMES, phase=0.5),
        oscillate_path(440, 320, 80, N_FRAMES, phase=1.5),
        linear_path(160, 400, 480, 400, N_FRAMES),
        [(320, 200)] * N_FRAMES,
    ]
    frames_C = []
    for i in range(N_FRAMES):
        bg = make_background((55, 65, 55))
        for j, (crop, path) in enumerate(zip(crowd_crops, positions)):
            scale = 0.70 + 0.05 * j
            bg = place_person(bg, crop, path[i][0], path[i][1], scale=scale)
        bg = add_timestamp(bg, i / FPS, "C")
        frames_C.append(bg)
    write_video(f"{OUT_DIR}/test_video_cam_C.mp4", frames_C)

    # ── Query image 1: single person from cam A scene ────────────────────────
    print("\nGenerating query_cam_A_person.jpg...")
    mid_frame_A = frames_A[N_FRAMES // 2].copy()
    save_query_image(f"{OUT_DIR}/query_cam_A_person.jpg", mid_frame_A)

    # ── Query image 2: single person from cam B scene ────────────────────────
    print("Generating query_cam_B_person.jpg...")
    mid_frame_B = frames_B[N_FRAMES // 2].copy()
    save_query_image(f"{OUT_DIR}/query_cam_B_person.jpg", mid_frame_B)

    # ── Query image 3: re-id target — person from cam A, to search in cam B ──
    print("Generating query_reid_target.jpg...")
    reid_frame = frames_A[N_FRAMES // 4].copy()
    # Add label
    cv2.putText(reid_frame, "RE-ID TARGET", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    save_query_image(f"{OUT_DIR}/query_reid_target.jpg", reid_frame)

    # ── Verify detections on generated assets ────────────────────────────────
    print("\n=== Detection verification on generated assets ===")
    for fname in [
        "test_video_cam_A.mp4", "test_video_cam_B.mp4", "test_video_cam_C.mp4",
        "query_cam_A_person.jpg", "query_cam_B_person.jpg", "query_reid_target.jpg",
    ]:
        fpath = f"{OUT_DIR}/{fname}"
        if fname.endswith(".mp4"):
            cap = cv2.VideoCapture(fpath)
            ok, frame = cap.read()
            cap.release()
            if ok:
                dets = detect_people(frame, require_verification=False)
                print(f"  {fname}: first frame -> {len(dets)} people detected")
        else:
            img = decode_image(open(fpath, "rb").read())
            dets = detect_people(img, require_verification=False)
            print(f"  {fname}: {len(dets)} people detected")

    print("\nDone. New assets are in:", OUT_DIR)
    print("\nUpdated demo flow:")
    print("  1. Upload test_video_cam_A.mp4  -> Register -> index")
    print("  2. Upload test_video_cam_B.mp4  -> Register -> index")
    print("  3. Upload test_video_cam_C.mp4  -> Register -> index")
    print("  4. Gallery -> browse detections")
    print("  5. Search -> upload query_reid_target.jpg -> select person -> run search")
    print("     (should find same person in cam_B footage)")


if __name__ == "__main__":
    main()
