"""
generate_realistic_assets.py
============================
Generates 6 realistic, visually distinct test videos and 8 query images
for the VisionTrace demo. All assets are written to data/test images/.

Videos produced:
  test_video_cam_A.mp4          - Indoor corridor, 2 people, warm lighting
  test_video_cam_B.mp4          - Outdoor plaza, 3 people, bright daylight
  test_video_cam_C.mp4          - Parking lot, 4 people, low contrast
  test_video_multi_person.mp4   - Busy lobby, 5 people crossing paths
  test_video_mixed_scenes.mp4   - Stairwell, 2 people, high contrast
  test_video_person_A.mp4       - Single tracked person, close-up follow

Query images produced:
  query_person_A.jpg            - Person A tight crop, frontal
  query_person_B.jpg            - Person B tight crop, side view
  query_person_C.jpg            - Person C, partial occlusion
  query_cam_A_person.jpg        - Scene frame from cam A
  query_cam_B_person.jpg        - Scene frame from cam B
  query_reid_target.jpg         - Re-ID target with label
  query_crowd_scene.jpg         - Multi-person crowd frame
  query_multi_person.jpg        - Multi-person mid-frame

Run:
    python generate_realistic_assets.py
"""
from __future__ import annotations
import math
import os
import random
import cv2
import numpy as np

OUT_DIR = "data/test images"
W, H = 640, 480
FPS = 5
DURATION_S = 10
N = FPS * DURATION_S  # 50 frames

random.seed(7)
np.random.seed(7)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic person silhouette builder
# ---------------------------------------------------------------------------

PALETTES = [
    # (shirt_bgr, trouser_bgr, skin_bgr)
    ((30,  80, 180), (20,  20,  60), (100, 150, 200)),   # blue shirt, dark trousers
    ((20, 160,  20), (40,  40,  40), ( 90, 130, 180)),   # green shirt, black trousers
    ((10,  10, 200), (80,  50,  20), (110, 160, 210)),   # red shirt, brown trousers
    ((180,180,  20), (20,  60,  20), ( 95, 140, 190)),   # yellow shirt, dark green
    ((160, 80, 160), (30,  30,  80), (100, 145, 195)),   # purple shirt, navy trousers
    ((200,120,  30), (50,  50,  50), (105, 155, 205)),   # orange shirt, grey trousers
]


def draw_person(canvas: np.ndarray, cx: int, cy: int, ph: int,
                palette_idx: int = 0, facing: str = "front") -> None:
    """Draw a stick-figure person with filled body parts onto canvas."""
    shirt, trouser, skin = PALETTES[palette_idx % len(PALETTES)]
    pw = max(int(ph * 0.38), 8)
    head_r = max(int(ph * 0.13), 4)
    torso_h = int(ph * 0.32)
    leg_h = int(ph * 0.38)
    arm_h = int(ph * 0.28)

    top = cy - ph // 2

    # head
    hcx, hcy = cx, top + head_r
    cv2.circle(canvas, (hcx, hcy), head_r, skin, -1)

    # torso
    tx1 = cx - pw // 2
    ty1 = hcy + head_r
    tx2 = cx + pw // 2
    ty2 = ty1 + torso_h
    cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), shirt, -1)

    # legs
    lw = max(pw // 3, 4)
    lx_l = cx - pw // 4
    lx_r = cx + pw // 4
    ly1 = ty2
    ly2 = ly1 + leg_h
    cv2.rectangle(canvas, (lx_l - lw // 2, ly1), (lx_l + lw // 2, ly2), trouser, -1)
    cv2.rectangle(canvas, (lx_r - lw // 2, ly1), (lx_r + lw // 2, ly2), trouser, -1)

    # arms
    aw = max(pw // 5, 3)
    ax_l = tx1 - aw
    ax_r = tx2 + aw
    ay1 = ty1 + 4
    ay2 = ay1 + arm_h
    cv2.rectangle(canvas, (ax_l - aw // 2, ay1), (ax_l + aw // 2, ay2), shirt, -1)
    cv2.rectangle(canvas, (ax_r - aw // 2, ay1), (ax_r + aw // 2, ay2), shirt, -1)

    # walking leg swing (alternate every 5 frames based on cy)
    swing = int(math.sin(cy * 0.3) * lw * 1.5)
    cv2.line(canvas, (lx_l, ly1), (lx_l + swing, ly2), trouser, lw)
    cv2.line(canvas, (lx_r, ly1), (lx_r - swing, ly2), trouser, lw)


# ---------------------------------------------------------------------------
# Background generators
# ---------------------------------------------------------------------------

def bg_corridor(noise: float = 0.0) -> np.ndarray:
    """Warm indoor corridor with perspective lines."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # floor gradient (warm beige)
    for y in range(H):
        t = y / H
        r = int(160 + 40 * t)
        g = int(140 + 30 * t)
        b = int(100 + 20 * t)
        img[y, :] = (b, g, r)
    # ceiling (darker warm)
    img[:80, :] = (80, 90, 110)
    # walls
    img[80:320, :60] = (100, 110, 130)
    img[80:320, 580:] = (100, 110, 130)
    # perspective lines
    vp = (W // 2, 120)
    for x in [0, W // 4, W // 2, 3 * W // 4, W]:
        cv2.line(img, vp, (x, H), (90, 100, 120), 1)
    # floor tiles
    for y in range(300, H, 40):
        cv2.line(img, (0, y), (W, y), (140, 125, 90), 1)
    if noise > 0:
        n = (np.random.randn(H, W, 3) * noise * 255).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def bg_plaza(t_norm: float = 0.0) -> np.ndarray:
    """Bright outdoor plaza — sky + ground."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # sky gradient
    for y in range(220):
        frac = y / 220
        img[y, :] = (int(200 - 30 * frac), int(180 - 20 * frac), int(255 - 10 * frac))
    # ground (concrete)
    img[220:, :] = (160, 165, 170)
    # ground lines
    for x in range(0, W, 80):
        cv2.line(img, (x, 220), (x, H), (145, 150, 155), 1)
    for y in range(220, H, 50):
        cv2.line(img, (0, y), (W, y), (145, 150, 155), 1)
    # sun glare top-right
    cv2.circle(img, (W - 60, 40), 30, (220, 230, 255), -1)
    # moving cloud
    cx = int(100 + t_norm * 300)
    cv2.ellipse(img, (cx, 60), (60, 20), 0, 0, 360, (230, 235, 245), -1)
    cv2.ellipse(img, (cx + 30, 50), (40, 18), 0, 0, 360, (230, 235, 245), -1)
    return img


def bg_parking(flicker: float = 0.0) -> np.ndarray:
    """Low-contrast parking lot — grey tones."""
    img = np.full((H, W, 3), (90, 90, 90), dtype=np.uint8)
    # parking lines
    for x in range(60, W, 100):
        cv2.line(img, (x, 200), (x, H), (110, 110, 110), 2)
    for y in range(200, H, 60):
        cv2.line(img, (0, y), (W, y), (105, 105, 105), 1)
    # distant cars (rectangles)
    for i, (cx, cy) in enumerate([(120, 170), (300, 160), (500, 175)]):
        col = (60 + i * 15, 60 + i * 10, 70 + i * 12)
        cv2.rectangle(img, (cx - 50, cy - 20), (cx + 50, cy + 20), col, -1)
    # overhead lamp flicker
    brightness = int(flicker * 20)
    img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return img


def bg_lobby(t_norm: float = 0.0) -> np.ndarray:
    """Busy lobby — marble floor, glass walls."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # ceiling
    img[:100, :] = (200, 200, 195)
    # walls
    img[100:350, :] = (185, 185, 180)
    # marble floor
    for y in range(350, H):
        t = (y - 350) / (H - 350)
        v = int(210 + 20 * t)
        img[y, :] = (v, v, v - 5)
    # floor reflection lines
    for x in range(0, W, 60):
        cv2.line(img, (x, 350), (x, H), (195, 195, 190), 1)
    # glass panels
    for x in [0, W // 3, 2 * W // 3]:
        cv2.rectangle(img, (x, 100), (x + W // 3 - 2, 350), (170, 185, 195), 1)
    # scrolling banner
    bx = int((1.0 - t_norm) * (W + 200)) - 200
    cv2.rectangle(img, (bx, 105), (bx + 180, 130), (50, 100, 200), -1)
    cv2.putText(img, "WELCOME", (bx + 10, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def bg_stairwell() -> np.ndarray:
    """High-contrast stairwell — dark walls, bright steps."""
    img = np.full((H, W, 3), (40, 40, 45), dtype=np.uint8)
    # stair steps
    step_h = 28
    for i in range(12):
        y = H - i * step_h
        x_off = i * 20
        cv2.rectangle(img, (x_off, y - step_h), (W - x_off, y), (200, 195, 185), -1)
        cv2.line(img, (x_off, y - step_h), (W - x_off, y - step_h), (160, 155, 145), 2)
    # handrail
    cv2.line(img, (80, 80), (W - 80, H - 40), (120, 100, 80), 4)
    # overhead light
    cv2.circle(img, (W // 2, 30), 20, (240, 240, 220), -1)
    cv2.line(img, (W // 2, 50), (W // 2, 100), (200, 200, 180), 2)
    return img


def bg_closeup() -> np.ndarray:
    """Neutral close-up background — blurred office."""
    img = np.full((H, W, 3), (130, 125, 120), dtype=np.uint8)
    # blurred desk/window shapes
    cv2.rectangle(img, (0, 300), (W, H), (100, 95, 90), -1)
    cv2.rectangle(img, (200, 80), (440, 260), (150, 155, 160), -1)
    cv2.rectangle(img, (210, 90), (430, 250), (170, 175, 180), 2)
    img = cv2.GaussianBlur(img, (21, 21), 0)
    return img


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def add_overlay(frame: np.ndarray, cam_id: str, t: float,
                extra: str = "") -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (W, 28), (0, 0, 0), -1)
    label = f"CAM {cam_id}  {t:05.1f}s"
    if extra:
        label += f"  {extra}"
    cv2.putText(frame, label, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    # REC dot
    if int(t * 2) % 2 == 0:
        cv2.circle(frame, (W - 18, 14), 6, (0, 0, 220), -1)
    return frame


def add_shadow(frame: np.ndarray, cx: int, cy: int, pw: int) -> None:
    axes = (max(pw // 2, 4), max(pw // 6, 2))
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy + 4), axes, 0, 0, 360, (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def write_video(path: str, frames: list[np.ndarray]) -> None:
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
        if out.isOpened():
            break
    for f in frames:
        out.write(f)
    out.release()
    kb = os.path.getsize(path) // 1024
    print(f"  [video] {path}  ({len(frames)} frames, {kb} KB, {fourcc_str})")


def save_jpg(path: str, img: np.ndarray) -> None:
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    kb = os.path.getsize(path) // 1024
    print(f"  [image] {path}  ({kb} KB)")


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def path_linear(x0, y0, x1, y1):
    return [(int(lerp(x0, x1, i / (N - 1))), int(lerp(y0, y1, i / (N - 1)))) for i in range(N)]


def path_sine(cx, cy, amp_x, amp_y, freq=1.0, phase=0.0):
    return [(int(cx + amp_x * math.sin(2 * math.pi * freq * i / N + phase)),
             int(cy + amp_y * math.sin(4 * math.pi * freq * i / N + phase))) for i in range(N)]


def path_enter_exit(x_enter, x_exit, y, pause_start=15, pause_end=35):
    pts = []
    for i in range(N):
        if i < pause_start:
            t = i / pause_start
            pts.append((int(lerp(x_enter, (x_enter + x_exit) // 2, t)), y))
        elif i <= pause_end:
            pts.append(((x_enter + x_exit) // 2, y))
        else:
            t = (i - pause_end) / (N - pause_end)
            pts.append((int(lerp((x_enter + x_exit) // 2, x_exit, t)), y))
    return pts


# ---------------------------------------------------------------------------
# Video 1 — Indoor corridor, 2 people, warm lighting
# ---------------------------------------------------------------------------

def gen_cam_A():
    frames = []
    p1 = path_linear(60, 260, 580, 260)
    p2 = path_linear(580, 310, 60, 310)
    for i in range(N):
        bg = bg_corridor(noise=0.015)
        t = i / FPS
        ph1 = int(lerp(60, 110, i / N))
        ph2 = int(lerp(110, 60, i / N))
        add_shadow(bg, p1[i][0], p1[i][1] + ph1 // 2, int(ph1 * 0.38))
        draw_person(bg, p1[i][0], p1[i][1], ph1, palette_idx=0)
        add_shadow(bg, p2[i][0], p2[i][1] + ph2 // 2, int(ph2 * 0.38))
        draw_person(bg, p2[i][0], p2[i][1], ph2, palette_idx=2)
        add_overlay(bg, "A - CORRIDOR", t, "2 PERSONS")
        frames.append(bg)
    write_video(f"{OUT_DIR}/test_video_cam_A.mp4", frames)
    return frames


# ---------------------------------------------------------------------------
# Video 2 — Outdoor plaza, 3 people, bright daylight
# ---------------------------------------------------------------------------

def gen_cam_B():
    frames = []
    p1 = path_linear(40, 300, 600, 300)
    p2 = path_enter_exit(620, -20, 340)
    p3 = path_sine(320, 370, 80, 10, freq=0.8, phase=0.0)
    for i in range(N):
        bg = bg_plaza(t_norm=i / N)
        t = i / FPS
        ph = 100
        add_shadow(bg, p1[i][0], p1[i][1] + ph // 2, int(ph * 0.38))
        draw_person(bg, p1[i][0], p1[i][1], ph, palette_idx=1)
        add_shadow(bg, p2[i][0], p2[i][1] + ph // 2, int(ph * 0.38))
        draw_person(bg, p2[i][0], p2[i][1], ph, palette_idx=3)
        add_shadow(bg, p3[i][0], p3[i][1] + ph // 2, int(ph * 0.38))
        draw_person(bg, p3[i][0], p3[i][1], ph, palette_idx=4)
        add_overlay(bg, "B - PLAZA", t, "3 PERSONS")
        frames.append(bg)
    write_video(f"{OUT_DIR}/test_video_cam_B.mp4", frames)
    return frames


# ---------------------------------------------------------------------------
# Video 3 — Parking lot, 4 people, low contrast
# ---------------------------------------------------------------------------

def gen_cam_C():
    frames = []
    p1 = path_linear(30, 250, 610, 250)
    p2 = path_linear(610, 320, 30, 320)
    p3 = path_enter_exit(30, 610, 380, pause_start=10, pause_end=40)
    p4 = path_sine(320, 290, 100, 5, freq=0.6, phase=1.0)
    for i in range(N):
        flicker = math.sin(i * 0.7) * 0.5 + 0.5
        bg = bg_parking(flicker=flicker)
        t = i / FPS
        ph = 95
        for path, pal in [(p1, 5), (p2, 0), (p3, 2), (p4, 3)]:
            add_shadow(bg, path[i][0], path[i][1] + ph // 2, int(ph * 0.38))
            draw_person(bg, path[i][0], path[i][1], ph, palette_idx=pal)
        add_overlay(bg, "C - PARKING", t, "4 PERSONS")
        frames.append(bg)
    write_video(f"{OUT_DIR}/test_video_cam_C.mp4", frames)
    return frames


# ---------------------------------------------------------------------------
# Video 4 — Busy lobby, 5 people crossing paths
# ---------------------------------------------------------------------------

def gen_multi_person():
    frames = []
    paths = [
        path_linear(20, 280, 620, 280),
        path_linear(620, 320, 20, 320),
        path_linear(20, 360, 620, 360),
        path_enter_exit(320, 320, 240, pause_start=5, pause_end=45),
        path_sine(160, 350, 60, 8, freq=1.2, phase=0.8),
    ]
    pals = [0, 1, 2, 3, 4]
    for i in range(N):
        bg = bg_lobby(t_norm=i / N)
        t = i / FPS
        for path, pal in zip(paths, pals):
            ph = 105
            add_shadow(bg, path[i][0], path[i][1] + ph // 2, int(ph * 0.38))
            draw_person(bg, path[i][0], path[i][1], ph, palette_idx=pal)
        add_overlay(bg, "D - LOBBY", t, "5 PERSONS")
        frames.append(bg)
    write_video(f"{OUT_DIR}/test_video_multi_person.mp4", frames)
    return frames


# ---------------------------------------------------------------------------
# Video 5 — Stairwell, 2 people, high contrast
# ---------------------------------------------------------------------------

def gen_mixed_scenes():
    frames = []
    p1 = path_linear(100, 420, 540, 200)
    p2 = path_linear(540, 200, 100, 420)
    for i in range(N):
        bg = bg_stairwell()
        t = i / FPS
        ph1 = int(lerp(80, 120, i / N))
        ph2 = int(lerp(120, 80, i / N))
        add_shadow(bg, p1[i][0], p1[i][1] + ph1 // 2, int(ph1 * 0.38))
        draw_person(bg, p1[i][0], p1[i][1], ph1, palette_idx=1)
        add_shadow(bg, p2[i][0], p2[i][1] + ph2 // 2, int(ph2 * 0.38))
        draw_person(bg, p2[i][0], p2[i][1], ph2, palette_idx=5)
        add_overlay(bg, "E - STAIRWELL", t, "HIGH CONTRAST")
        frames.append(bg)
    write_video(f"{OUT_DIR}/test_video_mixed_scenes.mp4", frames)
    return frames


# ---------------------------------------------------------------------------
# Video 6 — Single tracked person, close-up follow
# ---------------------------------------------------------------------------

def gen_person_A():
    frames = []
    # person walks left to right, camera "zooms" (person grows)
    for i in range(N):
        bg = bg_closeup()
        t = i / FPS
        cx = int(lerp(120, 520, i / N))
        cy = 280
        ph = int(lerp(90, 160, i / N))
        add_shadow(bg, cx, cy + ph // 2, int(ph * 0.38))
        draw_person(bg, cx, cy, ph, palette_idx=0)
        # tracking box
        pw = int(ph * 0.38)
        bx1, by1 = cx - pw - 5, cy - ph // 2 - 5
        bx2, by2 = cx + pw + 5, cy + ph // 2 + 5
        cv2.rectangle(bg, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(bg, "ID:001  conf:0.94", (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        add_overlay(bg, "F - CLOSEUP", t, "TRACKING ID:001")
        frames.append(bg)
    write_video(f"{OUT_DIR}/test_video_person_A.mp4", frames)
    return frames


# ---------------------------------------------------------------------------
# Query image generators
# ---------------------------------------------------------------------------

def gen_query_images(frames_A, frames_B, frames_C, frames_multi, frames_mixed, frames_pA):
    mid = N // 2

    # query_person_A — tight crop of person A from corridor video, frontal
    bg = bg_closeup()
    draw_person(bg, 320, 280, 200, palette_idx=0)
    crop_a = bg[80:460, 200:440].copy()
    save_jpg(f"{OUT_DIR}/query_person_A.jpg", crop_a)

    # query_person_B — person B side-ish, green shirt
    bg = bg_closeup()
    draw_person(bg, 320, 280, 200, palette_idx=1)
    # slight horizontal shift to simulate side view
    M = np.float32([[1, 0.08, 0], [0, 1, 0]])
    bg = cv2.warpAffine(bg, M, (W, H))
    crop_b = bg[80:460, 200:440].copy()
    save_jpg(f"{OUT_DIR}/query_person_B.jpg", crop_b)

    # query_person_C — partial occlusion (bottom half cut by railing)
    bg = bg_stairwell()
    draw_person(bg, 320, 260, 180, palette_idx=2)
    cv2.rectangle(bg, (0, 340), (W, H), (60, 55, 50), -1)  # occluder
    cv2.putText(bg, "PARTIAL VIEW", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    crop_c = bg[80:400, 180:460].copy()
    save_jpg(f"{OUT_DIR}/query_person_C.jpg", crop_c)

    # query_cam_A_person — mid-frame from corridor
    frame = frames_A[mid].copy()
    save_jpg(f"{OUT_DIR}/query_cam_A_person.jpg", frame)

    # query_cam_B_person — mid-frame from plaza
    frame = frames_B[mid].copy()
    save_jpg(f"{OUT_DIR}/query_cam_B_person.jpg", frame)

    # query_reid_target — person from corridor with RE-ID label
    frame = frames_A[N // 4].copy()
    cv2.rectangle(frame, (2, 2), (W - 2, H - 2), (0, 200, 255), 3)
    cv2.putText(frame, "RE-ID TARGET", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    save_jpg(f"{OUT_DIR}/query_reid_target.jpg", frame)

    # query_crowd_scene — frame from parking lot (4 people)
    frame = frames_C[mid].copy()
    cv2.putText(frame, "CROWD SCENE", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1, cv2.LINE_AA)
    save_jpg(f"{OUT_DIR}/query_crowd_scene.jpg", frame)

    # query_multi_person — mid-frame from lobby (5 people)
    frame = frames_multi[mid].copy()
    cv2.putText(frame, "MULTI-PERSON", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 200), 1, cv2.LINE_AA)
    save_jpg(f"{OUT_DIR}/query_multi_person.jpg", frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== VisionTrace — Generating realistic test assets ===\n")

    print("Video 1/6 — Indoor corridor (cam A)...")
    frames_A = gen_cam_A()

    print("Video 2/6 — Outdoor plaza (cam B)...")
    frames_B = gen_cam_B()

    print("Video 3/6 — Parking lot (cam C)...")
    frames_C = gen_cam_C()

    print("Video 4/6 — Busy lobby (multi-person)...")
    frames_multi = gen_multi_person()

    print("Video 5/6 — Stairwell (mixed/high-contrast)...")
    frames_mixed = gen_mixed_scenes()

    print("Video 6/6 — Close-up tracking (person A)...")
    frames_pA = gen_person_A()

    print("\nGenerating 8 query images...")
    gen_query_images(frames_A, frames_B, frames_C, frames_multi, frames_mixed, frames_pA)

    print("\n=== Done ===")
    print(f"All assets written to: {OUT_DIR}/")
    print("\nVideos:")
    for v in ["test_video_cam_A.mp4", "test_video_cam_B.mp4", "test_video_cam_C.mp4",
              "test_video_multi_person.mp4", "test_video_mixed_scenes.mp4", "test_video_person_A.mp4"]:
        kb = os.path.getsize(f"{OUT_DIR}/{v}") // 1024
        print(f"  {v}  ({kb} KB)")
    print("\nQuery images:")
    for q in ["query_person_A.jpg", "query_person_B.jpg", "query_person_C.jpg",
              "query_cam_A_person.jpg", "query_cam_B_person.jpg",
              "query_reid_target.jpg", "query_crowd_scene.jpg", "query_multi_person.jpg"]:
        kb = os.path.getsize(f"{OUT_DIR}/{q}") // 1024
        print(f"  {q}  ({kb} KB)")


if __name__ == "__main__":
    main()
