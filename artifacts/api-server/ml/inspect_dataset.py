from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import PARSERS, validate_and_list


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an authorized local Re-ID dataset without importing media.")
    parser.add_argument("--dataset", required=True, choices=sorted(PARSERS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    report = validate_and_list(
        args.dataset,
        Path(args.root),
        PARSERS[args.dataset],
        dry_run=True,
        split=args.split,
        max_items=args.max_items,
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.layout_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
