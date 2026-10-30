from __future__ import annotations

from pathlib import Path

JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffd9"
)


def write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(JPEG)


def build_market1501(root: Path) -> None:
    write_jpeg(root / "bounding_box_train" / "0001_c1s1_000151_00.jpg")
    write_jpeg(root / "bounding_box_train" / "0001_c2s1_000201_00.jpg")
    write_jpeg(root / "bounding_box_test" / "0001_c3s1_000301_00.jpg")
    write_jpeg(root / "query" / "0001_c1s1_000401_00.jpg")


def build_msmt17(root: Path) -> None:
    write_jpeg(root / "train" / "0001" / "0001_c1_001.jpg")
    write_jpeg(root / "train" / "0001" / "0001_c2_002.jpg")
    write_jpeg(root / "test" / "0002" / "0002_c3_001.jpg")
    (root / "list_train.txt").write_text("train/0001/0001_c1_001.jpg 0001\ntrain/0001/0001_c2_002.jpg 0001\n")
    (root / "list_query.txt").write_text("test/0002/0002_c3_001.jpg 0002\n")
    (root / "list_gallery.txt").write_text("test/0002/0002_c3_001.jpg 0002\n")


def build_mars(root: Path) -> None:
    write_jpeg(root / "bbox_train" / "0001" / "0001C1T0001F001.jpg")
    write_jpeg(root / "bbox_train" / "0001" / "0001C2T0001F002.jpg")
    write_jpeg(root / "bbox_test" / "0002" / "0002C1T0001F001.jpg")


def build_all(base: Path) -> dict[str, Path]:
    paths = {
        "market1501": base / "market1501",
        "msmt17": base / "msmt17",
        "mars": base / "mars",
    }
    build_market1501(paths["market1501"])
    build_msmt17(paths["msmt17"])
    build_mars(paths["mars"])
    return paths


if __name__ == "__main__":
    created = build_all(Path("data/dataset-fixtures"))
    for name, path in created.items():
        print(f"{name}={path.resolve()}")
