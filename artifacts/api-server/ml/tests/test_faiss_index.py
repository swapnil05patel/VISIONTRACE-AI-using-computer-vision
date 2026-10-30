from __future__ import annotations

import numpy as np

from faiss_index import FaissVectorIndex


def _unit(index: int) -> list[float]:
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1.0
    return vector.tolist()


def test_add_search_rebuild_remove(tmp_path) -> None:
    index = FaissVectorIndex(str(tmp_path / "test.faiss"))
    added = index.add([1, 2], [_unit(0), _unit(1)], "space-a")
    assert added == 2
    assert index.size == 2
    hits = index.search(_unit(0), top_k=1)
    assert hits[0]["vectorId"] == 1
    assert hits[0]["similarity"] > 0.99
    assert index.add([1], [_unit(0)], "space-a") == 0
    rebuilt = index.rebuild([3], [_unit(3)], "space-a")
    assert rebuilt == 1
    assert index.size == 1
    removed = index.remove([3])
    assert removed == 1
    assert index.size == 0
    assert not (tmp_path / "test.faiss").exists()


def test_rejects_incompatible_space(tmp_path) -> None:
    index = FaissVectorIndex(str(tmp_path / "space.faiss"))
    index.add([1], [_unit(0)], "space-a")
    try:
        index.add([2], [_unit(1)], "space-b")
        raise AssertionError("mixed spaces should fail")
    except ValueError as error:
        assert "incompatible" in str(error)
