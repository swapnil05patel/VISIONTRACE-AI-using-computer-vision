from __future__ import annotations

import re
from pathlib import Path

from .base import DatasetSample, ValidationReport, iter_images

# Market-1501 filenames: 0001_c1s1_000151_00.jpg  (id_camera_sequence_frame)
FILENAME = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)s\d+_.*\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_market1501(root: Path) -> ValidationReport:
    report = ValidationReport(dataset_id="market1501", root=str(root), layout_ok=False)
    split_dirs = {
        "train": root / "bounding_box_train",
        "gallery": root / "bounding_box_test",
        "query": root / "query",
    }
    missing = [name for name, path in split_dirs.items() if not path.is_dir()]
    if missing:
        report.errors.append(
            "Market-1501 layout was not recognized. Expected bounding_box_train, bounding_box_test, and query. "
            f"Missing: {', '.join(missing)}"
        )
        return report
    report.layout_ok = True
    seen_files: set[str] = set()
    identities: set[str] = set()
    cameras: set[str] = set()
    for split, directory in split_dirs.items():
        files = iter_images(directory)
        report.discovered_files += len(files)
        for path in files:
            if path.name in seen_files:
                report.skipped += 1
                report.errors.append(f"Duplicate filename skipped: {path.name}")
                continue
            seen_files.add(path.name)
            match = FILENAME.match(path.name)
            if not match:
                report.skipped += 1
                continue
            pid = match.group("pid")
            if pid in {"-1", "0000"}:
                report.skipped += 1
                continue
            camera = match.group("cam")
            identities.add(pid)
            cameras.add(camera)
            report.samples.append(
                DatasetSample(
                    dataset_id="market1501",
                    split=split,
                    identity_id=pid,
                    camera_id=camera,
                    path=str(path),
                    original_id=path.name,
                )
            )
    report.valid_samples = len(report.samples)
    report.identities = len(identities)
    report.cameras = len(cameras)
    for sample in report.samples:
        report.splits[sample.split] = report.splits.get(sample.split, 0) + 1
    if report.valid_samples == 0:
        report.errors.append("No valid Market-1501 identity samples were parsed from filenames.")
    return report
