from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import PARSERS, validate_and_list

ACCESS_NOTES = {
    "market1501": "Market-1501 is a research Re-ID dataset. Confirm Tsinghua/provider terms before import. Media stays on the configured filesystem.",
    "msmt17": "MSMT17 requires agreement with the dataset providers. Confirm access terms before import. Media stays on the configured filesystem.",
    "mars": "MARS is a video-based Re-ID dataset. Confirm provider terms before import. Media stays on the configured filesystem.",
}


def sample_key(dataset_id: str, original_id: str, path: str) -> str:
    digest = hashlib.sha256(f"{dataset_id}:{original_id}:{path}".encode()).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and import metadata for an authorized local Re-ID dataset. Does not download data."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(PARSERS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--purpose", default="benchmark", choices=["benchmark", "index"])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="data/dataset-imports")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    report = validate_and_list(
        args.dataset,
        root,
        PARSERS[args.dataset],
        dry_run=False,
        split=args.split,
        max_items=args.max_items,
    )
    payload = {
        **report.as_dict(),
        "purpose": args.purpose,
        "accessNote": ACCESS_NOTES[args.dataset],
        "importedAt": datetime.now(timezone.utc).isoformat(),
        "dryRun": args.dry_run,
        "samples": [] if args.dry_run else [
            {
                "id": sample_key(sample.dataset_id, sample.original_id, sample.path),
                "datasetId": sample.dataset_id,
                "split": sample.split,
                "identityId": sample.identity_id,
                "cameraId": sample.camera_id,
                "path": sample.path,
                "originalId": sample.original_id,
            }
            for sample in report.samples
        ],
    }
    if args.dry_run:
        payload["validSamples"] = report.valid_samples
        print(json.dumps({k: v for k, v in payload.items() if k != "samples"}, indent=2))
        return 0 if report.layout_ok else 2
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.dataset}-{args.split or 'all'}-{args.purpose}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"success": report.layout_ok, "output": str(output), **report.as_dict()}, indent=2))
    return 0 if report.layout_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
