from __future__ import annotations

import numpy as np
import pytest

from preprocess import ImageError, clip_bbox, decode_image, l2_normalize, osnet_tensor, preprocess_record


def test_decode_rejects_empty() -> None:
    with pytest.raises(ImageError):
        decode_image(b"")


def test_decode_rejects_invalid_bytes() -> None:
    with pytest.raises(ImageError):
        decode_image(b"not-an-image")


def test_clip_bbox_rejects_tiny_boxes() -> None:
    assert clip_bbox([0, 0, 2, 2], 100, 100) is None
    assert clip_bbox([10, 10, 40, 80], 100, 100) == [10, 10, 40, 80]


def test_l2_normalize_rejects_nan_and_zero() -> None:
    with pytest.raises(ImageError):
        l2_normalize(np.zeros(512, dtype=np.float32))
    with pytest.raises(ImageError):
        l2_normalize(np.full(512, np.nan, dtype=np.float32))


def test_l2_normalize_unit_length() -> None:
    vector = l2_normalize(np.arange(512, dtype=np.float32) + 1)
    assert vector.shape == (512,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_osnet_tensor_shape() -> None:
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    tensor = osnet_tensor(crop)
    assert tuple(tensor.shape) == (1, 3, 256, 128)


def test_preprocess_record_id() -> None:
    record = preprocess_record()
    assert record["id"] == "osnet_256x128_imagenet_rgb"
    assert record["embeddingMetric"] == "cosine_via_inner_product"
