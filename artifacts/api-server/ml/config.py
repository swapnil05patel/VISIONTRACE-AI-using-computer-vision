from __future__ import annotations

import os
from pathlib import Path

# Get the project root directory (parent of ml/)
PROJECT_ROOT = Path(__file__).parent.parent

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
YOLO_WEIGHTS = Path(os.getenv("VISIONTRACE_YOLO_WEIGHTS", PROJECT_ROOT / "models" / "yolo11n.pt"))
OSNET_WEIGHTS = Path(os.getenv("VISIONTRACE_OSNET_WEIGHTS", PROJECT_ROOT / "models" / "osnet_x1_0.pth"))
VERIFICATION_FILE = Path(
    os.getenv("VISIONTRACE_MODEL_VERIFICATION_FILE", PROJECT_ROOT / "data" / "model-verification.json")
)
FAISS_PATH = Path(os.getenv("VISIONTRACE_FAISS_PATH", PROJECT_ROOT / "data" / "visiontrace.faiss"))
YOLO_CONF = float(os.getenv("VISIONTRACE_YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.getenv("VISIONTRACE_YOLO_IMGSZ", "640"))
YOLO_MODEL_LABEL = os.getenv("VISIONTRACE_YOLO_MODEL", "yolo11n")
OSNET_ARCHITECTURE = os.getenv("VISIONTRACE_OSNET_ARCHITECTURE", "osnet_x1_0")
SAMPLE_FPS = float(os.getenv("VISIONTRACE_SAMPLE_FPS", "1"))
EMBEDDING_SPACE = os.getenv("VISIONTRACE_EMBEDDING_SPACE", "osnet_x1_0_imagenet_l2")
PREPROCESS_ID = "osnet_256x128_imagenet_rgb"
PERSON_CLASS_INDEX = 0
OSNET_INPUT_HEIGHT = 256
OSNET_INPUT_WIDTH = 128
OSNET_EMBEDDING_DIM = 512
ML_SHARED_SECRET = os.getenv("VISIONTRACE_ML_SHARED_SECRET", "")
ACTIVE_MODEL_FILE = Path(os.getenv("VISIONTRACE_ACTIVE_MODELS_FILE", PROJECT_ROOT / "data" / "active-models.json"))
