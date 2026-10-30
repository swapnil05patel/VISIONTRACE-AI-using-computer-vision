from __future__ import annotations

from pathlib import Path

from .base import DatasetSample, ValidationReport, iter_images


def _parse_list(path: Path, split: str, report: ValidationReport, identities: set[str], cameras: set[str]) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        relative = parts[0]
        image_path = path.parent / relative
        if not image_path.exists():
            image_path = path.parent.parent / relative if (path.parent.parent / relative).exists() else Path(relative)
        if not image_path.is_file():
            report.skipped += 1
            report.errors.append(f"Missing listed file: {relative}")
            continue
        report.discovered_files += 1
        identity = parts[1] if len(parts) > 1 else image_path.parent.name
        camera = None
        name = image_path.stem
        if "_c" in name:
            camera = name.split("_c")[-1].split("_")[0]
        identities.add(identity)
        if camera:
            cameras.add(camera)
        report.samples.append(
            DatasetSample(
                dataset_id="msmt17",
                split=split,
                identity_id=identity,
                camera_id=camera,
                path=str(image_path.resolve()),
                original_id=relative,
            )
        )


def parse_msmt17(root: Path) -> ValidationReport:
    report = ValidationReport(dataset_id="msmt17", root=str(root), layout_ok=False)
    list_dir = root / "list_train.txt"
    if list_dir.exists():
        lists = {
            "train": root / "list_train.txt",
            "query": root / "list_query.txt",
            "gallery": root / "list_gallery.txt",
            "test": root / "list_val.txt" if (root / "list_val.txt").exists() else root / "list_gallery.txt",
        }
        if not lists["query"].exists() or not lists["gallery"].exists():
            report.errors.append("MSMT17 list_query.txt and list_gallery.txt are required alongside list_train.txt.")
            return report
        report.layout_ok = True
        identities: set[str] = set()
        cameras: set[str] = set()
        for split, path in lists.items():
            if path.exists():
                _parse_list(path, split if split != "test" else "gallery", report, identities, cameras)
        report.valid_samples = len(report.samples)
        report.identities = len(identities)
        report.cameras = len(cameras)
        for sample in report.samples:
            report.splits[sample.split] = report.splits.get(sample.split, 0) + 1
        return report

    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        report.errors.append(
            "MSMT17 layout was not recognized. Expected list_train.txt or train/ and test/ identity folders."
        )
        return report
    report.layout_ok = True
    identities: set[str] = set()
    cameras: set[str] = set()
    for split, directory in (("train", train_dir), ("gallery", test_dir)):
        for identity_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            identities.add(identity_dir.name)
            files = iter_images(identity_dir)
            report.discovered_files += len(files)
            for image in files:
                camera = None
                if "_c" in image.stem:
                    camera = image.stem.split("_c")[-1].split("_")[0]
                if camera:
                    cameras.add(camera)
                report.samples.append(
                    DatasetSample(
                        dataset_id="msmt17",
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
    return report
