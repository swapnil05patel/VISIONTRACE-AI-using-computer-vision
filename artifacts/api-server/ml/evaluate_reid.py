from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import DEVICE, EMBEDDING_SPACE, OSNET_EMBEDDING_DIM, OSNET_WEIGHTS, PREPROCESS_ID
from datasets import PARSERS
from hashing import sha256_file
from osnet import load_osnet
from preprocess import crop_person, decode_image, l2_normalize, osnet_tensor


def average_precision(ranks: np.ndarray) -> float:
    if ranks.size == 0:
        return 0.0
    hits = np.cumsum(ranks)
    precision = hits / (np.arange(len(ranks)) + 1)
    return float((precision * ranks).sum() / max(int(ranks.sum()), 1))


def evaluate(query: np.ndarray, gallery: np.ndarray, query_ids: list[str], gallery_ids: list[str]) -> dict:
    scores = query @ gallery.T
    rank1 = rank5 = rank10 = 0
    aps: list[float] = []
    for index, scores_row in enumerate(scores):
        order = np.argsort(-scores_row)
        ordered_ids = [gallery_ids[item] for item in order]
        matches = np.array([identity == query_ids[index] for identity in ordered_ids], dtype=np.float32)
        if matches.sum() == 0:
            continue
        rank1 += int(matches[:1].any())
        rank5 += int(matches[:5].any())
        rank10 += int(matches[:10].any())
        aps.append(average_precision(matches))
    count = max(len(aps), 1)
    return {
        "rank1": rank1 / count,
        "rank5": rank5 / count,
        "rank10": rank10 / count,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "queriesUsed": len(aps),
        "queriesSkipped": len(query_ids) - len(aps),
    }


def embed_paths(model, paths: list[str]) -> np.ndarray:
    vectors = []
    for path in paths:
        image = decode_image(Path(path).read_bytes())
        h, w = image.shape[:2]
        tensor = osnet_tensor(crop_person(image, [0, 0, w, h])).to(DEVICE)
        with torch.no_grad():
            vectors.append(l2_normalize(model(tensor), OSNET_EMBEDDING_DIM))
    return np.stack(vectors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an OSNet checkpoint on a local Re-ID dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(PARSERS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--weights", default=str(OSNET_WEIGHTS))
    parser.add_argument("--max-query", type=int, default=64)
    parser.add_argument("--max-gallery", type=int, default=256)
    args = parser.parse_args()
    report = PARSERS[args.dataset](Path(args.root))
    if not report.layout_ok:
        raise SystemExit("; ".join(report.errors) or "Dataset layout is invalid.")
    query = [sample for sample in report.samples if sample.split == "query"][: args.max_query]
    gallery = [sample for sample in report.samples if sample.split in {"gallery", "test"}][: args.max_gallery]
    if not query or not gallery:
        raise SystemExit("The dataset did not provide both query and gallery/test samples.")
    model = load_osnet(Path(args.weights), DEVICE)
    metrics = evaluate(
        embed_paths(model, [item.path for item in query]),
        embed_paths(model, [item.path for item in gallery]),
        [item.identity_id for item in query],
        [item.identity_id for item in gallery],
    )
    payload = {
        "dataset": args.dataset,
        "root": args.root,
        "checkpoint": str(Path(args.weights).resolve()),
        "checkpointSha256": sha256_file(Path(args.weights)),
        "embeddingSpace": EMBEDDING_SPACE,
        "preprocessId": PREPROCESS_ID,
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "queryCount": len(query),
        "galleryCount": len(gallery),
        "metrics": metrics,
        "protocolNote": "Subset evaluation on the provided local copy. Official dataset protocols may differ.",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
