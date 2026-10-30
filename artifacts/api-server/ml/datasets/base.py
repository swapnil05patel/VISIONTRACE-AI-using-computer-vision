from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class DatasetSample:
    dataset_id: str
    split: str
    identity_id: str
    camera_id: str | None
    path: str
    original_id: str


@dataclass
class ValidationReport:
    dataset_id: str
    root: str
    layout_ok: bool
    discovered_files: int = 0
    valid_samples: int = 0
    identities: int = 0
    cameras: int = 0
    skipped: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    samples: list[DatasetSample] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "datasetId": self.dataset_id,
            "root": self.root,
            "layoutOk": self.layout_ok,
            "discoveredFiles": self.discovered_files,
            "validSamples": self.valid_samples,
            "identities": self.identities,
            "cameras": self.cameras,
            "skipped": self.skipped,
            "splits": self.splits,
            "errors": self.errors,
        }


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def validate_and_list(
    dataset_id: str,
    root: Path,
    parser: Callable[[Path], ValidationReport],
    dry_run: bool = True,
    split: str | None = None,
    max_items: int | None = None,
) -> ValidationReport:
    if not root.is_dir():
        report = ValidationReport(dataset_id=dataset_id, root=str(root), layout_ok=False)
        report.errors.append(f"Dataset root does not exist or is not a directory: {root}")
        return report
    report = parser(root)
    if split:
        report.samples = [sample for sample in report.samples if sample.split == split]
        report.valid_samples = len(report.samples)
        report.splits = {split: len(report.samples)}
    if max_items is not None:
        report.samples = report.samples[: max(0, max_items)]
        report.valid_samples = len(report.samples)
    if dry_run:
        report.samples = []
    return report
