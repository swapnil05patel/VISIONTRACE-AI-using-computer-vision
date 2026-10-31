# VisionTrace AI Lab — Setup Guide

## Overview

VisionTrace is a person re-identification platform with three services:

| Service | Stack | Port |
|---|---|---|
| ML Sidecar | Python 3.12 + FastAPI + YOLO + OSNet + FAISS | 18777 |
| API Server | Node.js + Express + Drizzle ORM | 8080 |
| React Frontend | Vite + React | 3000 |

---

## Prerequisites

### Linux

```bash
# Python 3.12
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Node.js 20+
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# pnpm
npm install -g pnpm

# PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# ffmpeg (for video processing)
sudo apt install -y ffmpeg

# Build tools
sudo apt install -y build-essential git
```

### Windows

1. **Python 3.12** — https://www.python.org/downloads/ (check "Add to PATH")
2. **Node.js 20+** — https://nodejs.org/
3. **pnpm** — open PowerShell as Admin: `npm install -g pnpm`
4. **PostgreSQL** — https://www.postgresql.org/download/windows/
5. **ffmpeg** — https://ffmpeg.org/download.html → add `bin/` folder to PATH
6. **Git** — https://git-scm.com/download/win

---

## Step 1 — Clone the Repository

### Linux
```bash
git clone <repo-url> VisionTrace-AI-Lab
cd VisionTrace-AI-Lab
```

### Windows (PowerShell)
```powershell
git clone <repo-url> VisionTrace-AI-Lab
cd VisionTrace-AI-Lab
```

---

## Step 2 — PostgreSQL Database Setup

### Linux
```bash
sudo -u postgres psql -c "CREATE USER visiontrace WITH PASSWORD 'visiontrace';"
sudo -u postgres psql -c "CREATE DATABASE visiontrace OWNER visiontrace;"
```

### Windows (PowerShell — run as Admin)
```powershell
psql -U postgres -c "CREATE USER visiontrace WITH PASSWORD 'visiontrace';"
psql -U postgres -c "CREATE DATABASE visiontrace OWNER visiontrace;"
```

---

## Step 3 — Python Virtual Environment

### Linux
```bash
cd VisionTrace-AI-Lab
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
cd VisionTrace-AI-Lab
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU (optional):** Replace the torch install with the CUDA version:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step 4 — Node.js Dependencies

### Linux & Windows
```bash
pnpm install
```

---

## Step 5 — Environment Configuration

Copy the example env file:

### Linux
```bash
cp .env.example .env
```

### Windows (PowerShell)
```powershell
Copy-Item .env.example .env
```

Edit `.env` with your values. Key variables:

```env
# Database
DATABASE_URL=postgres://visiontrace:visiontrace@127.0.0.1:5432/visiontrace

# Ports
PORT=8080

# Dev auth (no Clerk needed in dev)
VISIONTRACE_DEV_USER_ID=dev-user
VITE_VISIONTRACE_DEV_USER_ID=dev-user
VISIONTRACE_ADMIN_USER_IDS=dev-user

# Storage
VISIONTRACE_OBJECT_BACKEND=fs
PRIVATE_OBJECT_DIR=data/objects

# ML Sidecar
VISIONTRACE_ML_URL=http://127.0.0.1:18777
VISIONTRACE_ML_SHARED_SECRET=

# Model paths (relative to project root)
VISIONTRACE_YOLO_WEIGHTS=models/yolo11n.pt
VISIONTRACE_OSNET_WEIGHTS=models/osnet_x1_0.pth
VISIONTRACE_YOLO_CONF=0.25
VISIONTRACE_YOLO_IMGSZ=640

# Data paths
VISIONTRACE_FAISS_PATH=artifacts/api-server/data/visiontrace.faiss
VISIONTRACE_MODEL_VERIFICATION_FILE=artifacts/api-server/data/model-verification.json
VISIONTRACE_ACTIVE_MODELS_FILE=artifacts/api-server/data/active-models.json
VISIONTRACE_SAMPLE_FPS=1
```

---

## Step 6 — Download Model Weights

Create the models directory and download weights:

### Linux
```bash
mkdir -p models

# YOLO11n (person detector)
wget -O models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# OSNet x1.0 (Re-ID embeddings)
wget -O models/osnet_x1_0.pth https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY
```

### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path models

# YOLO11n
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" -OutFile "models\yolo11n.pt"

# OSNet — download manually from:
# https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY
# Save as: models\osnet_x1_0.pth
```

Verify both files exist:
```
models/
  yolo11n.pt       (~5.5 MB)
  osnet_x1_0.pth   (~17 MB)
```

---

## Step 7 — Run Database Migrations

### Linux & Windows
```bash
pnpm --filter @workspace/db run migrate
```

---

## Step 8 — Model Verification

This step verifies the model weights work correctly on a sample image. Run once after downloading models.

### Linux
```bash
source venv/bin/activate
cd VisionTrace-AI-Lab   # project root

export $(grep -v '^#' .env | xargs)

python3.12 -m artifacts.api-server.ml.verify_models \
  --image data/samples/authorized-person.jpg
```

Or use the test images provided:
```bash
python3.12 artifacts/api-server/ml/verify_models.py \
  --image "data/test images/query_person_A.jpg"
```

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1

# Load env vars
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
  $parts = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
}

python artifacts\api-server\ml\verify_models.py --image "data\test images\query_person_A.jpg"
```

Expected output:
```json
{"success": true, "verifiedAt": "...", "detector": "yolo11n", "embeddingDimension": 512, ...}
```

---

## Step 9 — Create Symlink for Models (Linux only)

The API server expects models at `artifacts/api-server/models/`:

```bash
cd artifacts/api-server
ln -sf ../../models models
cd ../..
```

### Windows (PowerShell — run as Admin)
```powershell
cd artifacts\api-server
New-Item -ItemType Junction -Path "models" -Target "..\..\models"
cd ..\..
```

---

## Step 10 — Start All Services

Open **3 separate terminals** in the project root.

### Terminal 1 — ML Sidecar

#### Linux
```bash
cd VisionTrace-AI-Lab
export $(grep -v '^#' .env | xargs)
PYTHONPATH=artifacts/api-server/ml \
./venv/bin/python3.12 -m uvicorn app:app \
  --app-dir artifacts/api-server/ml \
  --host 127.0.0.1 --port 18777 --workers 1
```

#### Windows (PowerShell)
```powershell
cd VisionTrace-AI-Lab
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
  $parts = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
}
$env:PYTHONPATH = "artifacts/api-server/ml"
.\venv\Scripts\python.exe -m uvicorn app:app `
  --app-dir artifacts/api-server/ml `
  --host 127.0.0.1 --port 18777 --workers 1
```

Verify ML sidecar is ready:
```bash
curl http://127.0.0.1:18777/healthz
# Expected: {"status":"ok","runtime":{"status":"ready",...}}
```

---

### Terminal 2 — Express API Server

#### Linux
```bash
cd VisionTrace-AI-Lab
export $(grep -v '^#' .env | xargs)
PORT=8080 pnpm --filter @workspace/api-server run dev
```

#### Windows (PowerShell)
```powershell
cd VisionTrace-AI-Lab
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
  $parts = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
}
$env:PORT = "8080"
pnpm --filter @workspace/api-server run dev
```

---

### Terminal 3 — React Frontend

#### Linux
```bash
cd VisionTrace-AI-Lab
export $(grep -v '^#' .env | xargs)
PORT=3000 BASE_PATH=/ pnpm --filter @workspace/visiontrace run dev
```

#### Windows (PowerShell)
```powershell
cd VisionTrace-AI-Lab
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
  $parts = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
}
$env:PORT = "3000"
$env:BASE_PATH = "/"
pnpm --filter @workspace/visiontrace run dev
```

Open browser: **http://localhost:3000**

---

## Test Assets

All test images and videos are in `data/test images/`.

### Original assets (real-scene photos/videos)

| File | People detected | Use for |
|---|---|---|
| `query_person_A.jpg` | 2 | Image detection / search query |
| `query_person_B.jpg` | 3 | Image detection / search query |
| `query_person_C.jpg` | 1 | Image detection / search query |
| `query_crowd_scene.jpg` | 5 | Multi-person detection |
| `query_multi_person.jpg` | 10 | Multi-person detection |
| `test_video_person_A.mp4` | ~2/frame | Video indexing (6s, 5fps, 640×480) |
| `test_video_multi_person.mp4` | ~5/frame | Video indexing (6s, 5fps, 640×480) |
| `test_video_mixed_scenes.mp4` | ~2/frame | Video indexing (6s, 5fps, 640×480) |

### Generated multi-camera assets

Generated by `generate_test_assets.py` — simulates a 3-camera surveillance setup using real person crops from the original videos.

| File | People | Camera | Use for |
|---|---|---|---|
| `test_video_cam_A.mp4` | 2 | Cam A | Video indexing — 2 people walking opposite directions |
| `test_video_cam_B.mp4` | 3 | Cam B | Video indexing — 2 walking + 1 stationary |
| `test_video_cam_C.mp4` | 6 | Cam C | Video indexing — crowded scene |
| `query_cam_A_person.jpg` | 1 | Cam A | Query image — single person from Cam A |
| `query_cam_B_person.jpg` | 3–4 | Cam B | Query image — Cam B mid-scene |
| `query_reid_target.jpg` | 1 | Cam A | Re-ID query — upload this, search across Cam B/C footage |

To regenerate:
```powershell
venv\Scripts\python.exe generate_test_assets.py
```

---

## Demo Flow

### Basic demo (original assets)
1. Open **http://localhost:3000**
2. Upload `test_video_person_A.mp4` → Register → wait for indexing
3. Upload `test_video_multi_person.mp4` → Register → wait for indexing
4. Go to **Gallery** → browse indexed person detections
5. Click a detection → **Search Similar** → see ranked matches
6. Go to **Search** → upload `query_person_A.jpg` → select a person → run search

### Multi-camera Re-ID demo (generated assets)
1. Upload `test_video_cam_A.mp4` → Register as "Camera A" → wait for indexing
2. Upload `test_video_cam_B.mp4` → Register as "Camera B" → wait for indexing
3. Upload `test_video_cam_C.mp4` → Register as "Camera C" → wait for indexing
4. Go to **Gallery** → confirm detections appear from all 3 cameras
5. Go to **Search** → upload `query_reid_target.jpg` → select the person → run search
   - Expected: matches appear in Cam B and Cam C footage (same person crop used in both)
6. Click any gallery detection → **Search Similar** → verify cross-camera ranking

---

## Troubleshooting

**503 — weights not verified**
```bash
# Re-run verification
python3.12 artifacts/api-server/ml/verify_models.py \
  --image "data/test images/query_person_A.jpg"
# Then restart ML sidecar
```

**503 — ML sidecar not reachable**
```bash
curl http://127.0.0.1:18777/healthz
# If no response, ML sidecar is not running — start Terminal 1
```

**No person detected**
- Use full-scene images (640×480+), not pre-cropped patches
- Use images from `data/test images/query_*.jpg`

**Wrong venv / ModuleNotFoundError**
- Always use `./venv/bin/python3.12` (Linux) or `.\venv\Scripts\python.exe` (Windows)
- Never use system `python3.12` directly

**Database connection error**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql   # Linux
# Windows: check Services → postgresql
```

**pnpm not found**
```bash
npm install -g pnpm
```
### commands

```bash
cd "/home/swapnil/BUG PROOF/Captain Up/VisionAI/VisionTrace-AI-Lab"
export $(grep -v '^#' .env | xargs)
PYTHONPATH=artifacts/api-server/ml \
python3.12 -m uvicorn app:app \
  --app-dir artifacts/api-server/ml \
  --host 127.0.0.1 --port 18777 --workers 1

cd "/home/swapnil/BUG PROOF/Captain Up/VisionAI/VisionTrace-AI-Lab"
export $(grep -v '^#' .env | xargs)
PORT=8080 pnpm --filter @workspace/api-server run dev

cd "/home/swapnil/BUG PROOF/Captain Up/VisionAI/VisionTrace-AI-Lab"
export $(grep -v '^#' .env | xargs)
PORT=3000 BASE_PATH=/ pnpm --filter @workspace/visiontrace run dev


```
# terminal-1 ml sidecar
```
Get-Content .env | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $parts = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
}
$env:PYTHONPATH = "artifacts/api-server/ml"
python -m uvicorn app:app --app-dir artifacts/api-server/ml --host 127.0.0.1 --port 18777 --workers 1
```
# terminal-2 
```
Get-Content .env | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $parts = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
}

$env:NODE_ENV = "development"
$env:PORT = "8080"

pnpm --filter @workspace/api-server run build

pnpm --filter @workspace/api-server run start
```

# terminal-3
```
cd "C:\Users\Swapnil\Desktop\VisionAI\VISIONTRACE-AI-using-computer-vision"

$env:PORT = "3000"
$env:BASE_PATH = "/"

pnpm --filter @workspace/visiontrace run dev
```
# db
```
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" -h 127.0.0.1 -p 5432 -U visiontrace -d visiontrace
```