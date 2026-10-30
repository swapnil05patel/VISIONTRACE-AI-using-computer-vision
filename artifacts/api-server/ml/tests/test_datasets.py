from __future__ import annotations

from pathlib import Path

from datasets.market1501 import parse_market1501
from datasets.mars import parse_mars
from datasets.msmt17 import parse_msmt17


def _jpeg(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffd8ffd9"
        )
    )


def test_market1501_parser(tmp_path: Path) -> None:
    (tmp_path / "bounding_box_train").mkdir()
    (tmp_path / "bounding_box_test").mkdir()
    (tmp_path / "query").mkdir()
    _jpeg(tmp_path / "bounding_box_train" / "0001_c1s1_000151_00.jpg")
    _jpeg(tmp_path / "bounding_box_test" / "0001_c2s1_000301_00.jpg")
    _jpeg(tmp_path / "query" / "0001_c3s1_000151_00.jpg")
    _jpeg(tmp_path / "bounding_box_test" / "-1_c1s1_000001_00.jpg")
    report = parse_market1501(tmp_path)
    assert report.layout_ok
    assert report.valid_samples == 3
    assert report.identities == 1
    assert report.cameras == 3
    assert report.splits["query"] == 1


def test_msmt17_identity_folders(tmp_path: Path) -> None:
    (tmp_path / "train" / "0001").mkdir(parents=True)
    (tmp_path / "test" / "0002").mkdir(parents=True)
    _jpeg(tmp_path / "train" / "0001" / "0001_c1_001.jpg")
    _jpeg(tmp_path / "test" / "0002" / "0002_c2_001.jpg")
    report = parse_msmt17(tmp_path)
    assert report.layout_ok
    assert report.valid_samples == 2
    assert report.identities == 2


def test_mars_parser(tmp_path: Path) -> None:
    (tmp_path / "bbox_train" / "0001").mkdir(parents=True)
    (tmp_path / "bbox_test" / "0002").mkdir(parents=True)
    _jpeg(tmp_path / "bbox_train" / "0001" / "0001C1T0001F001.jpg")
    _jpeg(tmp_path / "bbox_test" / "0002" / "0002C2T0001F001.jpg")
    report = parse_mars(tmp_path)
    assert report.layout_ok
    assert report.valid_samples == 2
    assert "train" in report.splits


def test_market_reports_unsupported_layout(tmp_path: Path) -> None:
    report = parse_market1501(tmp_path)
    assert not report.layout_ok
    assert report.errors
