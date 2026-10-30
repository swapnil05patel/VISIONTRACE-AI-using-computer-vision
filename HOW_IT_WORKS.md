# VisionTrace AI — How It Works

VisionTrace detects people across multiple videos, embeds each person into a 512-dimensional vector, and lets you search for visually similar people across all indexed footage using a query image or a detected person.

---

## Architecture Overview

```
Videos (MP4)
    │
    ▼
[Node.js API Server]  ──────────────────────────────────────────────────────
    │  job worker (video_index)                                              │
    ▼                                                                        │
[Python FastAPI ML Sidecar — port 18777]                                    │
    │                                                                        │
    ├─ 1. Frame Sampling (OpenCV)          → raw frames (images)            │
    ├─ 2. YOLO11n Detection                → bounding boxes per frame       │
    ├─ 3. IoU Tracker                      → tracklet IDs within video      │
    ├─ 4. OSNet x1.0 Re-ID                 → 512-d L2 embedding per person  │
    └─ 5. FAISS IndexFlatIP                → cosine similarity index        │
                                                                             │
[PostgreSQL]  ◄──────────────────────────────────────────────────────────────
    vision_source_videos, vision_indexed_frames,
    vision_person_detections, vision_reid_embeddings
```

---

## Step-by-Step Pipeline

### 1. Video Upload & Registration

- User uploads a video file to owner-scoped object storage (`data/objects/`).
- A `POST /api/visiontrace/videos` call registers the video and queues a `video_index` job in PostgreSQL.

### 2. Frame Sampling

**File:** [`ml/video.py`](artifacts/api-server/ml/video.py)

- OpenCV reads the video and samples frames at a configurable rate (default: **1 FPS**).
- Each sampled frame yields: `frameNumber`, `timestampSeconds`, and the raw BGR image array.

```
native_fps = 30
sample_fps = 1
interval   = 30  → every 30th frame is processed
```

### 3. Person Detection — YOLO11n

**File:** [`ml/inference.py`](artifacts/api-server/ml/inference.py) → `detect_people()`

- Model: **YOLO11n** (Ultralytics), COCO class 0 = "person"
- Config: confidence ≥ 0.25, input size 640×640
- Output per frame: list of detections, each with:
  - `bbox` — `[x1, y1, x2, y2]` pixel coordinates
  - `detectorScore` — YOLO confidence (0–1)
  - `id` — UUID
  - `className` — `"person"`

This is where **people are detected as image regions** from each video frame.

### 4. IoU Tracking (within one video)

**File:** [`ml/tracking.py`](artifacts/api-server/ml/tracking.py) → `IoUTracker`

- Assigns a `trackletId` (e.g. `t1`, `t2`) to each detection by matching bounding boxes across frames using **Intersection over Union (IoU ≥ 0.3)**.
- Tracks expire after 8 missed frames.
- Track IDs are **video-scoped only** — they are not global identities.

### 5. Person Embedding — OSNet x1.0

**File:** [`ml/inference.py`](artifacts/api-server/ml/inference.py) → `embed_person()`  
**File:** [`ml/preprocess.py`](artifacts/api-server/ml/preprocess.py)

For each detected person:
1. Crop the bounding box from the frame.
2. Resize to **256×128 px** (height × width).
3. Convert BGR → RGB, normalize with ImageNet mean/std.
4. Run through **OSNet x1.0** → raw 512-d feature vector.
5. L2-normalize → unit vector (enables cosine similarity via dot product).

```
Embedding space: osnet_x1_0_imagenet_l2
Dimension:       512
```

### 6. FAISS Vector Index

**File:** [`ml/faiss_index.py`](artifacts/api-server/ml/faiss_index.py)

- Index type: `IndexIDMap2(IndexFlatIP)` — exact inner product search on L2-normalized vectors = **cosine similarity**.
- Each vector is stored with a stable `vectorId` (PostgreSQL `vision_reid_embeddings.vector_id`).
- Persisted to disk at `data/visiontrace.faiss`.
- Thread-safe via `RLock`.

### 7. Database Storage

PostgreSQL stores all metadata:

| Table | Contents |
|---|---|
| `vision_source_videos` | video file path, camera ID, status |
| `vision_indexed_frames` | frame number, timestamp, thumbnail path |
| `vision_person_detections` | bbox, score, tracklet ID, crop path |
| `vision_reid_embeddings` | 512-d vector, embedding space, vector ID |

---

## Output Images

Yes — the pipeline produces image outputs at multiple levels:

| Output | Format | Where |
|---|---|---|
| Frame thumbnail | JPEG (base64 in API, path in DB) | `thumbnailPath` on `vision_indexed_frames` |
| Person crop | JPEG file | `cropPath` on `vision_person_detections` |
| Bounding box | `[x1,y1,x2,y2]` coordinates | `bbox` on `vision_person_detections` |

The `/process-video-chunk` ML endpoint returns `thumbnailJpegBase64` for each processed frame along with all detections.

---

## Search Modes

### Image Search
1. Upload a query image.
2. YOLO detects people in it → user selects one.
3. OSNet embeds the selected crop → 512-d vector.
4. FAISS searches for top-K nearest vectors.
5. Results are joined to source frames/videos in PostgreSQL → returned with `thumbnailPath`, `similarity`, `cameraId`, `timestampSeconds`.

### Detection Search (Gallery)
- Select any already-indexed detection.
- Its stored 512-d embedding is used directly for FAISS search (no re-inference needed).
- Results are temporally collapsed (±1.5 s window per video) to avoid duplicate hits.

### Tracklet Search
- Select a `trackletId` from an indexed video.
- The first stored embedding for that tracklet seeds the FAISS search.

---

## Models

| Model | Purpose | Weights |
|---|---|---|
| YOLO11n | Person detection | `models/yolo11n.pt` |
| OSNet x1.0 | Re-ID embedding (512-d) | `models/osnet_x1_0.pth` |

Both must pass `verify_models.py` before search is enabled. Verification checks SHA-256 hashes, embedding dimension, preprocess ID, and embedding space — stored in `data/model-verification.json`.

---

## Key Configuration

| Env Variable | Default | Meaning |
|---|---|---|
| `VISIONTRACE_SAMPLE_FPS` | `1` | Frames per second to sample |
| `VISIONTRACE_YOLO_CONF` | `0.25` | YOLO detection confidence threshold |
| `VISIONTRACE_YOLO_IMGSZ` | `640` | YOLO input image size |
| `VISIONTRACE_OSNET_ARCHITECTURE` | `osnet_x1_0` | Re-ID model architecture |
| `VISIONTRACE_EMBEDDING_SPACE` | `osnet_x1_0_imagenet_l2` | Embedding space identifier |
| `VISIONTRACE_FAISS_PATH` | `data/visiontrace.faiss` | FAISS index file path |

---

## Important Notes

- Similarity scores are **cosine similarities (0–1)**, not identity probabilities.
- Track IDs (`t1`, `t2`, ...) are **video-scoped** — the same person in two videos will have different track IDs.
- Cross-video identity matching is done purely by **embedding similarity** via FAISS.
- Changing the embedding space (e.g. fine-tuning OSNet) requires a full FAISS index rebuild.
- Only process footage you are authorized to use.
