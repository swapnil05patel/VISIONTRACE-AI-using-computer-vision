from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import torch

from config import (
    DEVICE,
    EMBEDDING_SPACE,
    OSNET_ARCHITECTURE,
    OSNET_EMBEDDING_DIM,
    OSNET_WEIGHTS,
    PERSON_CLASS_INDEX,
    PREPROCESS_ID,
    VERIFICATION_FILE,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL_LABEL,
    YOLO_WEIGHTS,
)
from hashing import sha256_file
from osnet import load_osnet
from preprocess import ImageError, clip_bbox, crop_person, decode_image, l2_normalize, osnet_tensor, preprocess_record

_detector: Any | None = None
_osnet: Any | None = None


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


def verification_record() -> dict[str, Any] | None:
    try:
        data = json.loads(VERIFICATION_FILE.read_text())
    except (OSError, ValueError, TypeError):
        return None
    if (
        data.get("detectorWeightsSha256") == sha256_file(YOLO_WEIGHTS)
        and data.get("reidWeightsSha256") == sha256_file(OSNET_WEIGHTS)
        and int(data.get("embeddingDimension", 0)) == OSNET_EMBEDDING_DIM
        and data.get("preprocessId") == PREPROCESS_ID
        and data.get("embeddingSpace") == EMBEDDING_SPACE
    ):
        return data
    return None


def runtime_status(vector_count: int = 0) -> dict[str, Any]:
    limitations: list[str] = []
    detector_imported = ultralytics_available()
    if not detector_imported:
        limitations.append("Ultralytics is not installed; YOLO detection is unavailable.")
    if not YOLO_WEIGHTS.is_file():
        limitations.append(f"YOLO weights are missing at {YOLO_WEIGHTS.as_posix()}.")
    if not OSNET_WEIGHTS.is_file():
        limitations.append(f"OSNet weights are missing at {OSNET_WEIGHTS.as_posix()}.")
    verified = verification_record()
    if verified is None:
        limitations.append(
            "Configured detector and Re-ID weights have not passed sample-image verification."
        )
    detector_ready = detector_imported and YOLO_WEIGHTS.is_file() and verified is not None
    reid_ready = OSNET_WEIGHTS.is_file() and verified is not None
    ready = detector_ready and reid_ready
    return {
        "status": "ready" if ready else ("partial" if vector_count else "unavailable"),
        "detector": f"{YOLO_MODEL_LABEL} ready" if detector_ready else "unavailable",
        "reid": "OSNet ready" if reid_ready else "unavailable",
        "vectorSearch": "FAISS ready" if vector_count else "FAISS empty",
        "device": DEVICE,
        "verifiedAt": verified.get("verifiedAt") if verified else None,
        "limitations": limitations,
        "embeddingSpace": verified.get("embeddingSpace") if verified else None,
        "embeddingDimension": verified.get("embeddingDimension") if verified else None,
        "preprocessId": verified.get("preprocessId") if verified else None,
    }


def load_models(*, require_verification: bool = True, vector_count: int = 0) -> tuple[Any, Any]:
    global _detector, _osnet
    if require_verification:
        status = runtime_status(vector_count)
        if status["status"] != "ready":
            raise ImageError("; ".join(status["limitations"]) or "Models are not ready.", status_code=503)
    else:
        if not ultralytics_available():
            raise ImageError("Ultralytics is not installed; YOLO detection is unavailable.", status_code=503)
        if not YOLO_WEIGHTS.is_file() or not OSNET_WEIGHTS.is_file():
            raise ImageError(
                f"Both model weight files must exist: YOLO={YOLO_WEIGHTS}, OSNet={OSNET_WEIGHTS}",
                status_code=503,
            )
    if _detector is None:
        from ultralytics import YOLO

        _detector = YOLO(str(YOLO_WEIGHTS))
    if _osnet is None:
        _osnet = load_osnet(OSNET_WEIGHTS, DEVICE)
    return _detector, _osnet


def detect_people(image: np.ndarray, *, require_verification: bool = True) -> list[dict[str, Any]]:
    detector, _ = load_models(require_verification=require_verification)
    predictions = detector.predict(
        image,
        classes=[PERSON_CLASS_INDEX],
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        device=DEVICE,
    )
    detections: list[dict[str, Any]] = []
    height, width = image.shape[:2]
    for result in predictions:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            coordinates = box.xyxy[0].detach().cpu().tolist()
            bbox = clip_bbox([float(value) for value in coordinates], width, height)
            if bbox is None:
                continue
            detections.append(
                {
                    "id": str(uuid.uuid4()),
                    "detectionIndex": len(detections),
                    "bbox": bbox,
                    "detectorScore": float(box.conf[0].detach().cpu().item()),
                    "cropPath": None,
                    "trackletId": None,
                    "classIndex": PERSON_CLASS_INDEX,
                    "className": "person",
                }
            )
    return detections


def embed_person(image: np.ndarray, detection: dict[str, Any], *, require_verification: bool = True) -> list[float]:
    _, model = load_models(require_verification=require_verification)
    crop = crop_person(image, detection["bbox"])
    tensor = osnet_tensor(crop).to(DEVICE)
    with torch.no_grad():
        features = model(tensor)
    vector = l2_normalize(features, OSNET_EMBEDDING_DIM)
    return vector.tolist()


def model_card() -> dict[str, Any]:
    return {
        "detector": YOLO_MODEL_LABEL,
        "detectorArchitecture": "ultralytics-yolo",
        "reidArchitecture": OSNET_ARCHITECTURE,
        "embeddingSpace": EMBEDDING_SPACE,
        "embeddingDimension": OSNET_EMBEDDING_DIM,
        "preprocess": preprocess_record(),
        "device": DEVICE,
        "detectorWeightsSha256": sha256_file(YOLO_WEIGHTS),
        "reidWeightsSha256": sha256_file(OSNET_WEIGHTS),
    }
