from __future__ import annotations

from pathlib import Path

import pytest

from config import OSNET_EMBEDDING_DIM, OSNET_WEIGHTS, YOLO_WEIGHTS
from inference import detect_people, embed_person
from preprocess import decode_image

SAMPLE = Path("data/samples/authorized-person.jpg")


@pytest.mark.skipif(not YOLO_WEIGHTS.is_file() or not OSNET_WEIGHTS.is_file() or not SAMPLE.is_file(), reason="Live weights or sample image are not present")
def test_live_detect_and_embed() -> None:
    image = decode_image(SAMPLE.read_bytes())
    detections = detect_people(image, require_verification=False)
    assert detections
    vector = embed_person(image, detections[0], require_verification=False)
    assert len(vector) == OSNET_EMBEDDING_DIM
