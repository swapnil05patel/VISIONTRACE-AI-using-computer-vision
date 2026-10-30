# VisionTrace AI

VisionTrace is a visual-search workspace for footage that the operator is authorized to process. PostgreSQL is the source of truth for metadata. A Python 3.12 FastAPI sidecar runs YOLO person detection, OSNet Re-ID, and FAISS cosine retrieval. Similarity scores are ranking signals, not identity probabilities.

## Run

```bash
python3.12 -m pip install -r requirements.txt
python3.12 scripts/fetch_model_weights.py
python3.12 artifacts/api-server/ml/verify_models.py --image data/samples/authorized-person.jpg
pnpm install
pnpm --filter @workspace/db run push
pnpm --filter @workspace/api-server run dev
pnpm --filter @workspace/visiontrace run dev
```

Copy `.env.example` and set Clerk keys for production sign-in. For local development, `VISIONTRACE_DEV_USER_ID` (and matching `VITE_VISIONTRACE_DEV_USER_ID`) enables a single owner-scoped user. Do not set that in production.

The API starts a FastAPI sidecar on `127.0.0.1:18777`. Set `VISIONTRACE_ML_URL` only for a separately managed compatible runtime. If that runtime is not localhost, set `VISIONTRACE_ML_SHARED_SECRET` on both sides.

## Model weights

Weights are not stored in git.

| Model | Default path | Source | License |
|-------|--------------|--------|---------|
| YOLO11n person detector (COCO class 0) | `models/yolo11n.pt` | [Ultralytics assets](https://github.com/ultralytics/assets/releases) | AGPL-3.0 |
| OSNet x1.0 (512-d) | `models/osnet_x1_0.pth` | Kaiyang Zhou deep-person-reid model zoo | MIT |

`scripts/fetch_model_weights.py` downloads those public checkpoints on demand. The application never downloads weights at startup. After files exist:

```bash
python3.12 artifacts/api-server/ml/verify_models.py --image /path/to/authorized-person-image.jpg
```

Search stays disabled until `data/model-verification.json` matches the configured weight hashes, preprocess id `osnet_256x128_imagenet_rgb`, and embedding space `osnet_x1_0_imagenet_l2`.

## Datasets

MARS, MSMT17, and Market-1501 are **not** downloaded. Place authorized local copies and set:

- `VISIONTRACE_MARS_ROOT` — `bbox_train/` and `bbox_test/`
- `VISIONTRACE_MSMT17_ROOT` — `list_train.txt` plus query/gallery lists, or `train/` and `test/` identity folders
- `VISIONTRACE_MARKET1501_ROOT` — `bounding_box_train/`, `bounding_box_test/`, `query/`

Layout-correct synthetic fixtures (not the licensed datasets) can be generated and imported for parser tests:

```bash
python3.12 artifacts/api-server/ml/tests/fixtures/make_fixtures.py
PYTHONPATH=artifacts/api-server/ml python3.12 artifacts/api-server/ml/import_dataset.py --dataset market1501 --root data/dataset-fixtures/market1501 --purpose benchmark
PYTHONPATH=artifacts/api-server/ml python3.12 artifacts/api-server/ml/import_dataset.py --dataset msmt17 --root data/dataset-fixtures/msmt17 --purpose benchmark
PYTHONPATH=artifacts/api-server/ml python3.12 artifacts/api-server/ml/import_dataset.py --dataset mars --root data/dataset-fixtures/mars --purpose benchmark
PYTHONPATH=artifacts/api-server/ml python3.12 artifacts/api-server/ml/inspect_dataset.py --dataset market1501 --root data/dataset-fixtures/market1501
```

Imports store **metadata only** in PostgreSQL (`vision_dataset_imports`, `vision_dataset_samples`). Media remains on the configured filesystem. Benchmark samples are never mixed into `vision_source_videos`.

## Training and evaluation

Training is explicit and never runs on upload or process start:

```bash
PYTHONPATH=artifacts/api-server/ml python3.12 artifacts/api-server/ml/train_osnet.py --dataset market1501 --root "$VISIONTRACE_MARKET1501_ROOT" --output data/training/osnet-finetune.pth
PYTHONPATH=artifacts/api-server/ml python3.12 artifacts/api-server/ml/evaluate_reid.py --dataset market1501 --root "$VISIONTRACE_MARKET1501_ROOT" --weights data/training/osnet-finetune.pth
```

A finished training run does not activate the production Re-ID model. Activation is an administrator action (`/models` or `POST /api/visiontrace/models/activate`) and requires verification. Changing embedding space requires FAISS rebuild.

YOLO fine-tuning is not included; the pretrained detector is sufficient for the search pipeline.

## Video ingestion and search

1. Upload a private video (library) — owner-scoped object storage (local `data/objects` or Replit GCS).
2. Registration queues a `video_index` job. The Node worker samples frames (default 1 FPS), detects people, tracks within that video, embeds crops, writes Postgres, then adds vectors to FAISS.
3. Image search detects people, embeds the selected crop, searches FAISS, and joins hits to owner-scoped source frames.
4. Video search queues `video_search` and aggregates nearby-frame cosine scores.
5. Tracklet search uses an indexed `trackletId` (`videoId:tN`). Track IDs are not identities and are not global.

Deleting a source video removes derived rows and FAISS ids.

## Tests

```bash
PYTHONPATH=artifacts/api-server/ml python3.12 -m pytest artifacts/api-server/ml/tests -q
pnpm run typecheck
```

Live inference tests run only when weights and `data/samples/authorized-person.jpg` exist. They are skipped otherwise; they never fabricate detections or metrics.

## Backup

Back up PostgreSQL, `data/objects`, `data/visiontrace.faiss`, `data/model-verification.json`, `data/active-models.json`, and `models/`. Restore the same embedding space before serving search.

## Security

- Clerk (or the development user header, non-production only) is required for private routes.
- Owner scoping applies to videos, jobs, embeddings, and search hydration.
- Visual similarity is not proof of identity.
- Process only public, synthetic, consented, or otherwise authorized footage.
