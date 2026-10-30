from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from config import OSNET_EMBEDDING_DIM, OSNET_INPUT_HEIGHT, OSNET_INPUT_WIDTH, PREPROCESS_ID

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def decode_image(data: bytes, max_bytes: int = 32 * 1024 * 1024) -> np.ndarray:
    if not data or len(data) > max_bytes:
        raise ImageError("Image is empty or exceeds 32 MB.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ImageError("Uploaded bytes are not a readable color image.")
    height, width = image.shape[:2]
    if height < 8 or width < 8:
        raise ImageError("Image dimensions are too small for person detection.")
    return image


def clip_bbox(bbox: list[float], width: int, height: int) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return [x1, y1, x2, y2]


def crop_person(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    height, width = image.shape[:2]
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        raise ImageError("Selected person crop is empty or invalid.")
    x1, y1, x2, y2 = [int(value) for value in clipped]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ImageError("Selected person crop is empty.")
    return crop


def osnet_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    if crop_bgr.size == 0:
        raise ImageError("Selected person crop is empty.")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (OSNET_INPUT_WIDTH, OSNET_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def l2_normalize(vector: np.ndarray | torch.Tensor, expected_dim: int = OSNET_EMBEDDING_DIM) -> np.ndarray:
    if isinstance(vector, torch.Tensor):
        tensor = vector.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = functional.normalize(tensor, p=2, dim=1)
        array = tensor[0].detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if norm <= 1e-12:
            raise ImageError("Zero-length embeddings cannot be indexed.", status_code=500)
        array = array / norm
    if array.size != expected_dim:
        raise ImageError(
            f"Embedding dimension mismatch: expected {expected_dim}, received {array.size}.",
            status_code=500,
        )
    if not np.isfinite(array).all() or array.size == 0:
        raise ImageError("The Re-ID model returned an invalid embedding.", status_code=500)
    return array.astype(np.float32)


def preprocess_record() -> dict[str, Any]:
    return {
        "id": PREPROCESS_ID,
        "inputHeight": OSNET_INPUT_HEIGHT,
        "inputWidth": OSNET_INPUT_WIDTH,
        "colorSpace": "rgb",
        "normalization": "imagenet",
        "embeddingMetric": "cosine_via_inner_product",
    }
