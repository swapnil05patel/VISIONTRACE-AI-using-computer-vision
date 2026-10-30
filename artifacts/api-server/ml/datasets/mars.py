from __future__ import annotations

from pathlib import Path

from .base import DatasetSample, ValidationReport, iter_images


def parse_mars(root: Path) -> ValidationReport:
    report = ValidationReport(dataset_id="mars", root=str(root), layout_ok=False)
    bbox_train = root / "bbox_train"
    bbox_test = root / "bbox_test"
    if not bbox_train.is_dir() or not bbox_test.is_dir():
        report.errors.append("MARS layout was not recognized. Expected bbox_train/ and bbox_test/ identity folders.")
        return report
    report.layout_ok = True
    identities: set[str] = set()
    cameras: set[str] = set()
    for split, directory in (("train", bbox_train), ("test", bbox_test)):
        identity_dirs = [path for path in directory.iterdir() if path.is_dir()]
        if not identity_dirs:
            files = iter_images(directory)
            report.discovered_files += len(files)
            for image in files:
                pid = image.stem.split("C")[0].split("c")[0][:4] if image.stem else "unknown"
                camera = None
                upper = image.stem.upper()
                if "C" in upper:
                    camera = upper.split("C")[-1][:1]
                identities.add(pid)
                if camera:
                    cameras.add(camera)
                report.samples.append(
                    DatasetSample(
                        dataset_id="mars",
                        split=split,
                        identity_id=pid,
                        camera_id=camera,
                        path=str(image),
                        original_id=image.name,
                    )
                )
            continue
        for identity_dir in identity_dirs:
            identities.add(identity_dir.name)
            files = iter_images(identity_dir)
            report.discovered_files += len(files)
            for image in files:
                camera = None
                upper = image.stem.upper()
                if "C" in upper:
                    camera = upper.split("C")[-1][:1]
                if camera:
                    cameras.add(camera)
                report.samples.append(
                    DatasetSample(
                        dataset_id="mars",
                        split=split,
                        identity_id=identity_dir.name,
                        camera_id=camera,
                        path=str(image),
                        original_id=image.name,
                    )
                )
    report.valid_samples = len(report.samples)
    report.identities = len(identities)
    report.cameras = len(cameras)
    for sample in report.samples:
        report.splits[sample.split] = report.splits.get(sample.split, 0) + 1
    if report.valid_samples == 0:
        report.errors.append("No MARS image samples were found under bbox_train/bbox_test.")
    return report
