# VisionTrace AI — Deep File-by-File Analysis

## Project Structure Overview

```
VisionTrace-AI-Lab/
├── artifacts/api-server/
│   ├── ml/                  ← Python FastAPI ML sidecar (port 18777)
│   └── src/                 ← Node.js Express API server (port 8080)
├── lib/
│   ├── db/                  ← Drizzle ORM schema + PostgreSQL client
│   ├── api-zod/             ← Zod validation schemas (shared types)
│   └── api-client-react/    ← Generated React API client
├── artifacts/visiontrace/   ← React frontend (port 3000)
├── data/                    ← Videos, models, FAISS index, uploads
├── models/                  ← yolo11n.pt + osnet_x1_0.pth
├── scripts/                 ← Setup, weight download, demo scripts
├── .env                     ← Environment config
├── requirements.txt         ← Python dependencies
├── start_all.sh             ← One-command startup script
└── pnpm-workspace.yaml      ← Monorepo workspace config
```

---

## PART 1 — Python ML Sidecar (`artifacts/api-server/ml/`)

### `config.py`
Central configuration loaded from environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `cuda` or `cpu` | Auto-detected via `torch.cuda.is_available()` |
| `YOLO_WEIGHTS` | `models/yolo11n.pt` | YOLO detector weights path |
| `OSNET_WEIGHTS` | `models/osnet_x1_0.pth` | Re-ID model weights path |
| `VERIFICATION_FILE` | `data/model-verification.json` | Hash-verified model record |
| `FAISS_PATH` | `data/visiontrace.faiss` | Persistent vector index |
| `YOLO_CONF` | `0.25` | Minimum detection confidence |
| `YOLO_IMGSZ` | `640` | YOLO input resolution |
| `OSNET_EMBEDDING_DIM` | `512` | Output vector size |
| `OSNET_INPUT_HEIGHT` | `256` | Person crop height |
| `OSNET_INPUT_WIDTH` | `128` | Person crop width |
| `EMBEDDING_SPACE` | `osnet_x1_0_imagenet_l2` | Space identifier for FAISS |
| `PREPROCESS_ID` | `osnet_256x128_imagenet_rgb` | Preprocessing fingerprint |
| `ML_SHARED_SECRET` | `""` | Optional auth token between Node and Python |

---

### `osnet.py`
Full re-implementation of OSNet x1.0 in pure PyTorch — no dependency on torchreid.

**Key classes:**

- `ConvLayer` — Conv2d + BatchNorm2d + ReLU. Standard building block.
- `Conv1x1` — 1×1 pointwise conv + BN + ReLU. Used for channel projection.
- `Conv1x1Linear` — 1×1 conv + BN only (no ReLU). Used before residual add.
- `LightConv3x3` — Depthwise-separable 3×3 conv. Reduces parameters vs full conv.
- `ChannelGate` — Squeeze-and-excitation attention. Global avg pool → FC → sigmoid → scale channels.
- `OSBlock` — Core residual block with 4 parallel branches of increasing depth (1, 2, 3, 4 LightConv3x3 layers), each gated by ChannelGate, then summed and projected.
- `OSNet` — Full network: conv1 → maxpool → conv2 → conv3 → conv4 → conv5 → global avg pool → FC → classifier.

**`load_osnet(weights, device)`**
1. Loads `.pth` checkpoint (handles `state_dict`, `model`, or raw dict keys).
2. Strips `module.` prefix (from DataParallel training).
3. Detects `num_classes` from classifier weight shape.
4. Loads weights with `strict=False` (classifier mismatch is OK for inference).
5. Sets `model.eval()` and moves to device.

During inference (`model.training == False`), `forward()` returns the 512-d FC feature vector, not class logits.

---

### `preprocess.py`
All image preprocessing before model inference.

- `decode_image(data)` — Decodes raw bytes via OpenCV. Validates: non-empty, ≤32 MB, color image, ≥8×8 px.
- `clip_bbox(bbox, width, height)` — Clamps bbox to image bounds. Rejects boxes smaller than 4×8 px.
- `crop_person(image, bbox)` — Slices the numpy array to the bbox region.
- `osnet_tensor(crop_bgr)` — Full preprocessing pipeline:
  1. BGR → RGB
  2. Resize to 128×256 (width × height)
  3. Divide by 255.0
  4. Subtract ImageNet mean `[0.485, 0.456, 0.406]`
  5. Divide by ImageNet std `[0.229, 0.224, 0.225]`
  6. Transpose HWC → CHW, add batch dim → shape `[1, 3, 256, 128]`
- `l2_normalize(vector, expected_dim)` — L2-normalizes the output vector. Validates finite values and correct dimension (512).
- `preprocess_record()` — Returns a dict describing the preprocessing config (used in verification).

---

### `inference.py`
Loads models and runs detection + embedding.

**Model loading:**
- `_detector` and `_osnet` are module-level singletons (loaded once, reused).
- `load_models()` checks `runtime_status()` before loading. Raises HTTP 503 if not ready.
- `verification_record()` reads `data/model-verification.json` and validates SHA-256 hashes of both weight files match, plus embedding dimension and preprocess ID.

**`detect_people(image)`**
1. Calls `detector.predict()` with `classes=[0]` (COCO person class), `conf=0.25`, `imgsz=640`.
2. Iterates YOLO result boxes.
3. Clips each bbox to image bounds.
4. Returns list of dicts: `{id, detectionIndex, bbox, detectorScore, cropPath, trackletId, classIndex, className}`.

**`embed_person(image, detection)`**
1. Crops person from image using bbox.
2. Runs `osnet_tensor()` preprocessing.
3. Passes tensor through OSNet with `torch.no_grad()`.
4. L2-normalizes the 512-d output.
5. Returns as Python `list[float]`.

**`runtime_status(vector_count)`**
Returns a status dict: `ready` / `partial` / `unavailable` based on whether Ultralytics is installed, weight files exist, and verification passes.

---

### `tracking.py`
IoU-based multi-object tracker scoped to a single video.

**`Track` dataclass** — holds `track_id`, `bbox`, `missed` frame count, `last_frame`.

**`IoUTracker.update(detections, frame_number)`**
1. Computes IoU between every detection bbox and every existing track bbox.
2. Sorts all (score, det_idx, track_id) pairs descending.
3. Greedily assigns detections to tracks (highest IoU first, each used once).
4. Unmatched detections → new track with next sequential ID (`t1`, `t2`, ...).
5. Tracks not matched this frame → `missed += 1`. Expired after `max_missed=8` frames.
6. Assigns `trackletId` string (e.g. `"t3"`) to each detection dict in-place.

Track IDs are video-scoped only. The Node.js worker prefixes them with `videoId:` before storing in PostgreSQL (e.g. `"abc-uuid:t3"`).

---

### `video.py`
OpenCV video I/O utilities.

**`probe_video(path)`** — Opens video, reads FPS, frame count, width, height, duration. Validates at least one readable frame exists.

**`iter_sampled_frames(path, sample_fps, start_frame)`**
- Calculates `interval = round(native_fps / sample_fps)`.
- Yields every `interval`-th frame starting from `start_frame`.
- Each yield: `{frameNumber, timestampSeconds, image}` where `image` is a BGR numpy array.

**`encode_jpeg(image, quality=80)`** — Encodes numpy BGR array to JPEG bytes via `cv2.imencode`.

---

### `faiss_index.py`
Thread-safe persistent FAISS vector index.

**Index type:** `IndexIDMap2(IndexFlatIP(512))` — exact inner product search on L2-normalized vectors = cosine similarity. No approximation.

**`FaissVectorIndex`**
- Persists to `data/visiontrace.faiss` + `data/visiontrace.faiss.meta.json`.
- Uses `RLock` for thread safety.
- `add(vector_ids, vectors, embedding_space)` — Validates vectors (finite, correct dim, non-zero norm), L2-normalizes, skips already-known IDs, calls `index.add_with_ids()`, saves atomically via `os.replace()`.
- `search(vector, top_k)` — L2-normalizes query, calls `index.search()`, returns `[{vectorId, similarity}]` sorted by score.
- `remove(vector_ids)` — Uses `IDSelectorBatch` to remove by ID. Clears index if empty.
- `rebuild(vector_ids, vectors, embedding_space)` — Wipes and rebuilds from scratch. Used for FAISS restore.
- `_save()` — Writes to a temp file then atomically renames to prevent corruption.

---

### `app.py`
FastAPI application — the ML sidecar HTTP interface.

**Middleware:** All routes except `/healthz` require `x-visiontrace-ml-secret` header if `ML_SHARED_SECRET` is set.

**Endpoints:**

| Method | Path | What it does |
|---|---|---|
| GET | `/healthz` | Returns runtime status, FAISS size, model card |
| POST | `/detect-image` | Upload image → YOLO detection → returns bbox list |
| POST | `/search-image` | Upload image → detect → embed selected person → FAISS search |
| POST | `/embed-image` | Upload image → detect → embed → return vector |
| POST | `/probe-video` | Returns video metadata (fps, duration, dimensions) |
| POST | `/process-video-chunk` | Sample frames → detect → track → embed → return frames+detections+embeddings |
| POST | `/vectors/add` | Add vectors to FAISS index |
| POST | `/vectors/search` | Search FAISS by vector |
| POST | `/vectors/remove` | Remove vectors from FAISS by ID |
| POST | `/vectors/rebuild` | Wipe and rebuild FAISS from provided vectors |

**`/process-video-chunk`** is the core pipeline endpoint. It:
1. Iterates sampled frames from the video file path.
2. Runs `detect_people()` on each frame.
3. Optionally runs `IoUTracker.update()`.
4. Runs `embed_person()` for each detection.
5. Encodes frame thumbnail as base64 JPEG.
6. Returns up to `maxFrames` frames with all detections and embeddings.

---

### `hashing.py`
Single function `sha256_file(path)` — streams file in 1 MB chunks, returns hex SHA-256 digest. Used to verify model weight files haven't changed.

---

### `verify_models.py`
CLI tool that must be run once before search is enabled.

1. Loads both weight files.
2. Runs `detect_people()` on a sample image (bypassing verification check).
3. Runs `embed_person()` on the first detection.
4. Validates embedding dimension == 512.
5. Writes `data/model-verification.json` with SHA-256 hashes, embedding space, preprocess ID, timestamp.
6. Writes `data/active-models.json` with detector + reid model info.

Search stays disabled until this file exists and matches the current weight hashes.

---

### `train_osnet.py`
CLI fine-tuning script for OSNet on a local Re-ID dataset.

1. Parses dataset using the appropriate parser (market1501/msmt17/mars).
2. Loads existing OSNet weights, replaces classifier head with new identity count.
3. Trains with Adam optimizer + CrossEntropyLoss for N epochs.
4. Saves best checkpoint (lowest train loss) to `data/training/osnet-finetune.pth`.
5. Writes a JSON training log alongside the checkpoint.
6. Does NOT activate the new model — activation is a separate admin action.

---

### `evaluate_reid.py`
CLI evaluation script. Computes Rank-1, Rank-5, Rank-10, and mAP on a local dataset.

1. Embeds all query and gallery images.
2. Computes cosine similarity matrix (query × gallery).
3. For each query, ranks gallery by similarity, checks identity matches.
4. Reports metrics as JSON.

---

### `import_dataset.py`
CLI tool to import Re-ID dataset metadata into JSON files.

- Validates dataset layout via the appropriate parser.
- Generates a stable SHA-256 sample key per image: `sha256(datasetId:originalId:path)`.
- Writes a JSON manifest to `data/dataset-imports/`.
- Does NOT copy or move media files — paths stay on the local filesystem.

---

### `inspect_dataset.py`
Dry-run version of `import_dataset.py`. Validates layout and prints stats without writing anything.

---

### `datasets/base.py`
Shared dataset infrastructure.

- `DatasetSample` — dataclass: `dataset_id`, `split`, `identity_id`, `camera_id`, `path`, `original_id`.
- `ValidationReport` — dataclass: layout_ok, counts, errors, sample list.
- `iter_images(root)` — recursively finds `.jpg/.jpeg/.png/.bmp` files.
- `validate_and_list()` — runs parser, applies split/max_items filters, optionally clears samples for dry-run.

---

### `datasets/market1501.py`
Parser for Market-1501 dataset.

- Expects: `bounding_box_train/`, `bounding_box_test/`, `query/` directories.
- Filename format: `0001_c1s1_000151_00.jpg` → identity=`0001`, camera=`1`.
- Skips junk identities (`-1`, `0000`) and duplicate filenames.

---

### `datasets/mars.py`
Parser for MARS (video-based Re-ID) dataset.

- Expects: `bbox_train/` and `bbox_test/` with identity subdirectories.
- Extracts camera from filename stem (character after `C`).
- Handles both flat and identity-folder layouts.

---

### `datasets/msmt17.py`
Parser for MSMT17 dataset. Supports two layouts:

1. List-file layout: `list_train.txt`, `list_query.txt`, `list_gallery.txt` with `relative/path identity` per line.
2. Folder layout: `train/` and `test/` with identity subdirectories.

Camera extracted from `_c{N}_` in filename stem.

---

## PART 2 — Node.js API Server (`artifacts/api-server/src/`)

### `index.ts`
Entry point. Requires `PORT` env var. On startup:
1. Calls `startMlSidecar()` — spawns the Python FastAPI process.
2. Calls `startJobWorker()` — starts the 2-second polling interval.
3. Calls `app.listen(port)` — starts Express.
4. Registers `SIGTERM`/`SIGINT` handlers for graceful shutdown (stops job worker, kills ML sidecar, closes HTTP server).

---

### `app.ts`
Express application setup.

- `pino-http` — structured request logging (redacts auth headers).
- Clerk proxy middleware — proxies Clerk auth requests.
- CORS — enabled with credentials.
- `express.json()` + `express.urlencoded()` — body parsing.
- Clerk middleware — only mounted if `CLERK_SECRET_KEY` is set.
- All routes mounted under `/api`.

---

### `src/lib/mlSidecar.ts`
Spawns the Python FastAPI sidecar as a child process.

- If `VISIONTRACE_ML_URL` is set → skips spawning (uses external runtime).
- Searches for `app.py` in `ml/` or `artifacts/api-server/ml/`.
- Runs: `python3.12 -m uvicorn app:app --app-dir {dir} --host 127.0.0.1 --port 18777 --workers 1`
- Pipes stderr to the Node.js logger.
- Logs unexpected exits as errors.

---

### `src/lib/visionRuntime.ts`
HTTP client for the ML sidecar.

**`getVisionRuntime()`** — GETs `/healthz` with 1.5s timeout. Returns parsed runtime status or `unavailableRuntime` on any error.

**`callMl<T>(path, init, timeoutMs)`** — Generic fetch wrapper:
- Adds `x-visiontrace-ml-secret` header if configured.
- Returns `{ok: true, data: T}` or `{ok: false, status, error}`.
- Never throws — all errors become `{ok: false}`.

---

### `src/lib/jobWorker.ts`
Background job processor. Polls PostgreSQL every 2 seconds.

**`tick()`** — Claims one queued job (atomic update to `running`), dispatches to handler, catches errors and marks job `failed`.

**`processVideoIndex(job)`** — Full video indexing pipeline:
1. Validates runtime is ready.
2. Downloads video from object storage to a temp file.
3. Calls `/probe-video` → stores fps, duration, dimensions in DB.
4. Loops calling `/process-video-chunk` with `maxFrames=8` until `exhausted=true`.
5. For each frame: inserts `vision_indexed_frames` row.
6. For each detection: inserts `vision_person_detections` row with `trackletId = videoId:tN`.
7. For each detection: inserts `vision_reid_embeddings` row with `indexStatus=pending`.
8. After each chunk: calls `indexPendingEmbeddings()` → batches pending vectors to `/vectors/add`.
9. Updates video status to `indexed` with final frame/person counts.

**`processVideoSearch(job)`** — Query video search:
1. Downloads query video to temp file.
2. Calls `/process-video-chunk` to get embeddings from up to 24 frames.
3. For each of up to 32 embeddings, calls `/vectors/search`.
4. Deduplicates and aggregates hits (±1.5s temporal collapse).
5. Stores result JSON on the job record.

**`processDatasetImport(job)`** — Runs `import_dataset.py` as a subprocess, reads the output JSON manifest, inserts rows into `vision_dataset_imports` and `vision_dataset_samples`.

**`aggregateHits(hits)`** — Groups hits by `videoId:trackletId`, picks best score per group, averages adjacent-frame scores within ±1.5s window.

---

### `src/lib/clerkAuth.ts`
Authentication helpers.

**`authenticatedUserId(req, res)`** — Returns Clerk user ID if `CLERK_SECRET_KEY` is set, otherwise falls back to `VISIONTRACE_DEV_USER_ID` (dev only). Returns `null` and sends 401 if unauthenticated.

**`requireAdmin(req, res)`** — Calls `authenticatedUserId`, then checks against `VISIONTRACE_ADMIN_USER_IDS` comma-separated list. In dev with no admins configured, any authenticated user is admin.

---

### `src/lib/objectStorage.ts`
Dual-backend object storage abstraction.

- In local mode (`VISIONTRACE_OBJECT_BACKEND=fs`): uses `LocalObjectStorage` backed by `data/objects/`.
- In GCS mode: uses `@google-cloud/storage` with Replit sidecar token exchange.
- `getObjectEntityUploadURL()` — generates a UUID upload path and returns a signed PUT URL (GCS) or direct upload URL (local).
- `getObjectEntityFile(objectPath)` — returns a file handle for reading.
- `downloadObject(file)` — streams file as a `Response` with correct Content-Type and Cache-Control.
- `canAccessObjectEntity()` — checks ACL policy (owner match).

---

### `src/lib/localObjectStorage.ts`
Local filesystem implementation of the storage interface.

- `LocalStoredFile` — wraps an absolute path. Stores metadata in a `.meta.json` sidecar file.
- `LocalObjectStorage` — root-anchored storage. `createUpload()` generates a UUID path. `file(objectPath)` returns a `LocalStoredFile`.
- `usesLocalObjectStorage()` — returns true if `PRIVATE_OBJECT_DIR` points to a local path or `data/objects` exists.

---

### `src/lib/datasetAdapters.ts`
Dataset configuration registry.

- Defines MARS, MSMT17, Market-1501 with their env variable names and access notes.
- `listDatasetAdapters(ownerId)` — returns dataset list with `configured` (env var set + dir exists), `importedItems` count from DB.
- `getDatasetRoot(datasetId)` — returns the configured root path or null.

---

### `src/routes/health.ts`
Single route: `GET /api/healthz` → `{status: "ok"}`.

---

### `src/routes/storage.ts`
File upload and download routes.

- `POST /api/storage/uploads/request-url` — validates file type/size, creates upload record in DB, returns upload URL.
- `PUT /api/storage/uploads/direct/objects/:kind/:id` — receives raw bytes, validates magic bytes match content type, writes to local storage.
- `GET /api/storage/objects/*path` — serves private files after ownership + ACL check.

**`getOwnedUpload(ownerId, objectPath, kind)`** — shared helper that validates the upload path pattern, checks DB ownership, validates content type and size, verifies magic bytes, sets ACL policy.

---

### `src/routes/visiontrace.ts`
Main API routes. All require authentication.

| Route | Purpose |
|---|---|
| `GET /visiontrace/runtime` | ML sidecar health + model status |
| `GET /visiontrace/overview` | Dashboard stats (video/frame/person counts, recent jobs) |
| `GET /visiontrace/videos` | List user's indexed videos |
| `POST /visiontrace/videos` | Register video → queue `video_index` job |
| `POST /visiontrace/videos/details` | Get single video |
| `POST /visiontrace/videos/frames` | List frames + detections for a video |
| `GET /visiontrace/cameras` | List distinct camera IDs |
| `POST /visiontrace/images/detections` | Detect people in an uploaded image |
| `POST /visiontrace/search/image` | Image-based person search across all videos |
| `POST /visiontrace/search/video` | Queue video search job |
| `POST /visiontrace/search/tracklet` | Search by tracklet ID |
| `POST /visiontrace/search/detection` | Search by stored detection embedding (gallery click) |
| `POST /visiontrace/jobs/status` | Poll job status |
| `POST /visiontrace/jobs/cancel` | Cancel queued/running job |
| `GET /visiontrace/gallery` | Paginated person detection gallery |
| `GET /visiontrace/gallery/videos` | Indexed videos for gallery filter |
| `POST /visiontrace/faiss/restore` | Rebuild FAISS from all DB embeddings |
| `GET /visiontrace/pipeline/status` | Live pipeline visualization data |
| `GET /visiontrace/models` | List model artifacts (admin) |
| `POST /visiontrace/models/activate` | Activate a verified model (admin) |
| `POST /visiontrace/models/rollback` | Rollback to previous model (admin) |
| `POST /visiontrace/videos/delete` | Delete video + remove FAISS vectors |
| `GET /visiontrace/datasets` | List dataset adapters |
| `POST /visiontrace/datasets/imports` | Queue dataset import job |

**Detection search** (`/search/detection`) uses temporal window collapse: groups results by `videoId:floor(timestamp/3)`, averages similarity scores within ±1.5s of the best hit per group.

---

## PART 3 — Database Schema (`lib/db/src/schema/`)

### `vision-videos.ts` → `vision_source_videos`
One row per uploaded video file.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `owner_id` | text | Clerk user ID |
| `name` | text | Display name |
| `object_path` | text | Storage path e.g. `/objects/uploads/uuid` |
| `camera_id` | text nullable | Optional camera label |
| `recorded_at` | timestamptz | Optional recording time |
| `status` | text | `queued → processing → indexed / failed` |
| `frame_count` | int | Set after indexing |
| `person_count` | int | Set after indexing |
| `sha256` | text | File hash |
| `fps` / `width` / `height` / `duration_seconds` | | From video probe |
| `sample_fps` | real | FPS used during indexing |

Indexes: `(owner_id, object_path)` unique, `(owner_id, created_at)`, `(owner_id, camera_id)`.

---

### `vision-frames.ts` → `vision_indexed_frames` + `vision_person_detections`

**`vision_indexed_frames`** — One row per sampled frame.
- `video_id` FK → cascade delete
- `frame_number` int, `timestamp_seconds` real
- `thumbnail_path` text nullable

**`vision_person_detections`** — One row per detected person per frame.
- `frame_id` FK → cascade delete
- `detection_index` int (0-based within frame)
- `bbox` real[] — `[x1, y1, x2, y2]`
- `detector_score` real — YOLO confidence
- `crop_path` text nullable
- `tracklet_id` text — `videoId:tN` format

Index on `tracklet_id` for tracklet search queries.

---

### `vision-embeddings.ts` → `vision_reid_embeddings`
One row per person detection (1:1 with detections).

| Column | Notes |
|---|---|
| `vector_id` | serial PK — used as FAISS ID |
| `detection_id` | UUID FK unique → cascade delete |
| `dimension` | 512 |
| `vector` | real[] — 512 floats |
| `model_version` | `osnet_x1_0` |
| `embedding_space` | `osnet_x1_0_imagenet_l2` |
| `index_status` | `pending → indexed` |

`vector_id` is a serial integer — this is the stable ID used in FAISS. PostgreSQL is the source of truth; FAISS can be rebuilt from these rows.

---

### `vision-jobs.ts` → `vision_processing_jobs`
Job queue table.

| Column | Notes |
|---|---|
| `kind` | `video_index`, `video_search`, `dataset_import` |
| `status` | `queued → running → completed / failed` |
| `progress` | 0–100 integer |
| `stage` | Human-readable stage name |
| `video_id` | FK nullable (for video jobs) |
| `dataset_id` | text nullable (for import jobs) |
| `result` | jsonb — stores search results for video_search jobs |
| `error` | text — error message on failure |

---

### `vision-uploads.ts` → `vision_private_uploads`
Tracks every upload request before the file is written.

- `object_path` unique — prevents duplicate registrations.
- `verified` bool — set to true after magic-byte validation on first access.
- `content_type` — validated against allowed MIME types.

---

### `vision-searches.ts` → `vision_completed_searches` + `vision_dataset_imports` + `vision_dataset_samples`

**`vision_completed_searches`** — Audit log of every search performed (mode, query path, result count).

**`vision_dataset_imports`** — One row per dataset import job with full validation stats.

**`vision_dataset_samples`** — One row per image in an imported dataset. ID is a SHA-256 of `datasetId:originalId:path` for deduplication.

---

### `vision-models.ts` → `vision_model_artifacts` + `vision_training_runs` + `vision_model_activation_history`

**`vision_model_artifacts`** — Registry of model checkpoints (detector or reid type). `verified` and `active` booleans control which model is in production.

**`vision_training_runs`** — Records of fine-tuning runs with config and metrics.

**`vision_model_activation_history`** — Audit trail of model activations and rollbacks with actor ID and previous model ID.

---

## PART 4 — Shared Libraries

### `lib/api-zod/`
Zod schemas for all API request/response types. Generated from `lib/api-spec/openapi.yaml` via Orval. Used by both the API server (validation) and the React client (type safety).

Key types: `PersonDetection`, `SearchHit`, `SearchResponse`, `ProcessingJob`, `SourceVideo`, `VisionRuntime`, `IndexedFrame`.

### `lib/api-client-react/`
Auto-generated React Query hooks for all API endpoints. Consumed by the frontend.

### `lib/db/`
Drizzle ORM client + all schema tables exported from `src/index.ts`. `drizzle.config.ts` reads `DATABASE_URL` from `.env` for migrations.

---

## PART 5 — Scripts

### `scripts/fetch_model_weights.py`
Downloads YOLO11n from Ultralytics GitHub releases and OSNet x1.0 from HuggingFace (with GitHub fallback). Also downloads a sample person image (`data/samples/authorized-person.jpg`). Prints SHA-256 of each downloaded file.

### `scripts/demo_setup.py`
Demo environment setup script.

### `start_all.sh`
One-command startup:
1. Loads `.env` (copies from `.env.example` if missing).
2. Optionally starts a PostgreSQL Docker container (`visiontrace_pg`).
3. Starts Express API backend on port 8080 (if not already running).
4. Starts React frontend on port 3000 (if not already running).
5. `wait` — keeps script alive until both processes exit.

---

---

## PART 6 — Complete Data Flow (End to End)

```
USER UPLOADS VIDEO
       │
       ▼
PUT /api/storage/uploads/direct/objects/uploads/{uuid}
  → magic-byte validation (ftyp header for MP4)
  → written to data/objects/uploads/{uuid}
  → vision_private_uploads row: verified=true

POST /api/visiontrace/videos
  → getOwnedUpload() validates ownership + content type
  → INSERT vision_source_videos (status=queued)
  → INSERT vision_processing_jobs (kind=video_index, status=queued)

JOB WORKER TICK (every 2s)
  → SELECT queued job → UPDATE status=running
  → getVisionRuntime() → GET http://127.0.0.1:18777/healthz
  → materializeObject() → download to /tmp/visiontrace-xxx/media.mp4

  POST /probe-video {path}
    → cv2.VideoCapture → fps, duration, width, height
    → UPDATE vision_source_videos (fps, duration, dimensions)

  LOOP until exhausted:
    POST /process-video-chunk {path, sampleFps=1, startFrame, maxFrames=8}
      → iter_sampled_frames() → every 30th frame (at 30fps native)
        → detect_people(frame)
            → YOLO11n.predict(classes=[0], conf=0.25, imgsz=640)
            → clip_bbox() → list of {bbox, detectorScore}
        → IoUTracker.update(detections, frameNumber)
            → IoU matching → assign trackletId (t1, t2, ...)
        → for each detection:
            embed_person(frame, detection)
              → crop_person() → osnet_tensor() → OSNet.forward()
              → l2_normalize() → 512-d float list
        → encode_jpeg(frame) → base64 thumbnail
      → return {frames: [{frameNumber, timestamp, thumbnail, detections+embeddings}]}

    for each frame:
      INSERT vision_indexed_frames (videoId, frameNumber, timestamp)
      for each detection:
        INSERT vision_person_detections (frameId, bbox, score, trackletId=videoId:tN)
        INSERT vision_reid_embeddings (detectionId, vector[512], indexStatus=pending)

    indexPendingEmbeddings():
      SELECT pending embeddings (up to 500)
      POST /vectors/add {vectorIds, vectors, embeddingSpace}
        → FaissVectorIndex.add() → index.add_with_ids() → save to disk
      UPDATE vision_reid_embeddings SET indexStatus=indexed

  UPDATE vision_source_videos (status=indexed, frameCount, personCount)
  UPDATE vision_processing_jobs (status=completed, progress=100)

USER SEARCHES BY IMAGE
  POST /api/visiontrace/search/image {objectPath, detectionIndex, filters}
    → getOwnedUpload() → download image bytes
    → POST /search-image (multipart: file + detection_index + top_k)
        → decode_image() → detect_people() → embed_person()
        → FaissVectorIndex.search(vector, topK*10)
        → return {matches: [{vectorId, similarity}]}
    → JOIN vision_reid_embeddings → vision_person_detections
         → vision_indexed_frames → vision_source_videos
    → filtersPass() (cameraId, date range, time range)
    → sort by similarity → slice to topK
    → INSERT vision_completed_searches
    → return {results: [{videoId, sourceName, cameraId, frameNumber,
                         timestampSeconds, similarity, thumbnailPath}]}
```

---

## PART 7 — Commands to Start All Services

### Prerequisites

```bash
# 1. Install Python 3.12
python3.12 --version

# 2. Install pnpm
npm install -g pnpm

# 3. Install PostgreSQL (or use Docker)
# Option A: local PostgreSQL
sudo apt install postgresql
sudo -u postgres createuser visiontrace --pwprompt  # password: visiontrace
sudo -u postgres createdb visiontrace --owner=visiontrace

# Option B: Docker (handled automatically by start_all.sh)
docker run -d --name visiontrace_pg \
  -e POSTGRES_USER=visiontrace \
  -e POSTGRES_PASSWORD=visiontrace \
  -e POSTGRES_DB=visiontrace \
  -p 5432:5432 postgres:16-alpine
```

---

### Step 1 — Environment Setup

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

# Copy env file
cp .env.example .env

# Edit .env — minimum required for local dev (defaults work as-is):
# DATABASE_URL=postgres://visiontrace:visiontrace@127.0.0.1:5432/visiontrace
# PORT=8080
# NODE_ENV=development
# VISIONTRACE_DEV_USER_ID=dev-user
# VITE_VISIONTRACE_DEV_USER_ID=dev-user
```

---

### Step 2 — Install Python Dependencies

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

python3.12 -m pip install -r requirements.txt
```

---

### Step 3 — Download Model Weights

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

python3.12 scripts/fetch_model_weights.py
# Downloads:
#   models/yolo11n.pt         (~6 MB)
#   models/osnet_x1_0.pth     (~2 MB)
#   data/samples/authorized-person.jpg
```

---

### Step 4 — Verify Models (required before search works)

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

PYTHONPATH=artifacts/api-server/ml \
python3.12 artifacts/api-server/ml/verify_models.py \
  --image data/samples/authorized-person.jpg

# Creates:
#   data/model-verification.json  (SHA-256 hashes + embedding space)
#   data/active-models.json       (active model registry)
```

---

### Step 5 — Install Node.js Dependencies

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

pnpm install
```

---

### Step 6 — Push Database Schema

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

pnpm --filter @workspace/db run push
# Runs drizzle-kit push → creates all vision_* tables in PostgreSQL
```

---

### Step 7 — Start All Services (One Command)

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

bash start_all.sh
```

This starts:
- Express API + ML sidecar on **http://localhost:8080**
- React frontend on **http://localhost:3000**

---

### Alternative: Start Services Individually

**Terminal 1 — ML Sidecar (Python FastAPI)**
```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

PYTHONPATH=artifacts/api-server/ml \
python3.12 -m uvicorn app:app \
  --app-dir artifacts/api-server/ml \
  --host 127.0.0.1 \
  --port 18777 \
  --workers 1
```

**Terminal 2 — Express API Server**
```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

pnpm --filter @workspace/api-server run dev
# Builds TypeScript → runs on http://localhost:8080
# Note: if VISIONTRACE_ML_URL is set, it won't spawn the sidecar again
```

**Terminal 3 — React Frontend**
```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

pnpm --filter @workspace/visiontrace run dev
# Vite dev server on http://localhost:3000
```

---

### Optional: Dataset Import (for Re-ID benchmarking)

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

# Inspect dataset layout (dry run, no writes)
PYTHONPATH=artifacts/api-server/ml \
python3.12 artifacts/api-server/ml/inspect_dataset.py \
  --dataset market1501 \
  --root data/dataset-fixtures/market1501

# Import dataset metadata
PYTHONPATH=artifacts/api-server/ml \
python3.12 artifacts/api-server/ml/import_dataset.py \
  --dataset market1501 \
  --root data/dataset-fixtures/market1501 \
  --purpose benchmark
```

---

### Optional: Fine-tune OSNet

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

PYTHONPATH=artifacts/api-server/ml \
python3.12 artifacts/api-server/ml/train_osnet.py \
  --dataset market1501 \
  --root data/dataset-fixtures/market1501 \
  --output data/training/osnet-finetune.pth \
  --epochs 5 \
  --batch-size 8

# After training, verify the new weights before activating:
PYTHONPATH=artifacts/api-server/ml \
python3.12 artifacts/api-server/ml/verify_models.py \
  --image data/samples/authorized-person.jpg \
  --output data/model-verification.json
```

---

### Optional: Evaluate Re-ID Model

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

PYTHONPATH=artifacts/api-server/ml \
python3.12 artifacts/api-server/ml/evaluate_reid.py \
  --dataset market1501 \
  --root data/dataset-fixtures/market1501 \
  --weights models/osnet_x1_0.pth
# Prints JSON: rank1, rank5, rank10, mAP
```

---

### Run Tests

```bash
cd /home/swapnil/BUG\ PROOF/Captain\ Up/VisionAI/VisionTrace-AI-Lab

# Python ML tests
PYTHONPATH=artifacts/api-server/ml \
python3.12 -m pytest artifacts/api-server/ml/tests -q

# TypeScript type check
pnpm run typecheck
```

---

### Restore FAISS Index (if lost or corrupted)

```bash
# Via API (requires running server):
curl -X POST http://localhost:8080/api/visiontrace/faiss/restore \
  -H "x-visiontrace-dev-user: dev-user"
```

---

## PART 8 — Service Ports & URLs

| Service | Port | URL |
|---|---|---|
| React Frontend | 3000 | http://localhost:3000 |
| Express API | 8080 | http://localhost:8080/api |
| Python ML Sidecar | 18777 | http://127.0.0.1:18777 |
| PostgreSQL | 5432 | postgres://visiontrace:visiontrace@127.0.0.1:5432/visiontrace |

**Health check endpoints:**
```bash
curl http://localhost:8080/api/healthz          # Node API
curl http://127.0.0.1:18777/healthz             # ML sidecar (full model status)
```

---

## PART 9 — Key Design Decisions

| Decision | Reason |
|---|---|
| Python sidecar for ML | PyTorch/YOLO/FAISS are Python-native; Node.js handles auth, DB, jobs |
| FAISS IndexFlatIP (exact search) | No approximation errors; dataset sizes don't require ANN |
| L2-normalize before IndexFlatIP | Inner product of unit vectors = cosine similarity |
| PostgreSQL as source of truth | FAISS can be rebuilt from `vision_reid_embeddings.vector` at any time |
| SHA-256 model verification | Prevents search with unverified or swapped weight files |
| Track IDs are video-scoped | IoU tracking cannot link identities across cameras — only embeddings can |
| Temporal collapse (±1.5s) | Prevents the same person appearing 5 times in results from adjacent frames |
| `indexStatus=pending` pattern | Decouples DB writes from FAISS writes; allows batch indexing and recovery |
| Serial `vector_id` as FAISS ID | Stable integer IDs that survive FAISS rebuilds |
