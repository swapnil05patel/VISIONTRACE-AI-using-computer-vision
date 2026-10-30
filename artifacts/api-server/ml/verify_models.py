from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EMBEDDING_SPACE,
    OSNET_ARCHITECTURE,
    OSNET_EMBEDDING_DIM,
    OSNET_WEIGHTS,
    PREPROCESS_ID,
    VERIFICATION_FILE,
    YOLO_MODEL_LABEL,
    YOLO_WEIGHTS,
)
from hashing import sha256_file
from inference import detect_people, embed_person
from preprocess import ImageError, decode_image, preprocess_record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load real YOLO and OSNet weights and verify them on an authorized sample image."
    )
    parser.add_argument("--image", required=True, help="Path to an authorized sample image containing a person.")
    parser.add_argument(
        "--output",
        default=str(VERIFICATION_FILE),
        help="Verification record location (default: the configured application path).",
    )
    args = parser.parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        parser.error("The sample image does not exist.")
    if not YOLO_WEIGHTS.is_file() or not OSNET_WEIGHTS.is_file():
        parser.error(
            "Both model weight files must exist before verification: "
            f"YOLO={YOLO_WEIGHTS}, OSNet={OSNET_WEIGHTS}"
        )

    try:
        image = decode_image(image_path.read_bytes())
        detections = detect_people(image, require_verification=False)
        if not detections:
            parser.error("The detector returned no people; choose a clear authorized sample.")
        vector = embed_person(image, detections[0], require_verification=False)
    except ImageError as error:
        parser.error(str(error))

    if len(vector) != OSNET_EMBEDDING_DIM:
        parser.error(f"Unexpected embedding dimension {len(vector)}; expected {OSNET_EMBEDDING_DIM}.")

    yolo_hash = sha256_file(YOLO_WEIGHTS)
    osnet_hash = sha256_file(OSNET_WEIGHTS)
    verification = {
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "detector": YOLO_MODEL_LABEL,
        "reidArchitecture": OSNET_ARCHITECTURE,
        "detectorWeightsSha256": yolo_hash,
        "reidWeightsSha256": osnet_hash,
        "embeddingDimension": len(vector),
        "embeddingSpace": EMBEDDING_SPACE,
        "preprocessId": PREPROCESS_ID,
        "preprocess": preprocess_record(),
        "sampleDetectionCount": len(detections),
        "sampleImage": str(image_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2) + "\n")
    active = Path(os.getenv("VISIONTRACE_ACTIVE_MODELS_FILE", "data/active-models.json"))
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(
        json.dumps(
            {
                "detector": {
                    "name": YOLO_MODEL_LABEL,
                    "type": "detector",
                    "status": "active",
                    "checksum": yolo_hash,
                    "path": str(YOLO_WEIGHTS),
                },
                "reid": {
                    "name": OSNET_ARCHITECTURE,
                    "type": "reid",
                    "status": "active",
                    "checksum": osnet_hash,
                    "path": str(OSNET_WEIGHTS),
                    "embeddingSpace": EMBEDDING_SPACE,
                    "embeddingDimension": len(vector),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"success": True, **verification}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
