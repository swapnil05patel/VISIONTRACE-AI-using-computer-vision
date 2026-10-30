from __future__ import annotations

import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np

from config import EMBEDDING_SPACE, FAISS_PATH, OSNET_EMBEDDING_DIM


class FaissVectorIndex:
    """Persistent cosine-similarity index over L2-normalized embeddings.

    PostgreSQL remains the metadata source of truth. FAISS IDs are stable
    `vision_reid_embeddings.vector_id` values and can be rebuilt from rows.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or FAISS_PATH)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.lock = RLock()
        self.index: Any | None = None
        self.dimension: int | None = None
        self.embedding_space: str | None = None
        self._load()

    def _load(self) -> None:
        import json

        with self.lock:
            if self.path.exists():
                try:
                    self.index = faiss.read_index(str(self.path))
                    self.dimension = int(self.index.d)
                except Exception:
                    self.index = None
                    self.dimension = None
                    self.embedding_space = None
                    return
            else:
                self.index = None
                self.dimension = None
            if self.meta_path.exists():
                try:
                    meta = json.loads(self.meta_path.read_text())
                    self.embedding_space = meta.get("embeddingSpace")
                except (OSError, ValueError, TypeError):
                    self.embedding_space = None
            else:
                self.embedding_space = EMBEDDING_SPACE if self.index is not None else None

    @property
    def size(self) -> int:
        with self.lock:
            return int(self.index.ntotal) if self.index is not None else 0

    def _new_index(self, dimension: int) -> Any:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def _write_meta(self) -> None:
        import json

        payload = {
            "embeddingSpace": self.embedding_space,
            "dimension": self.dimension,
            "size": int(self.index.ntotal) if self.index is not None else 0,
        }
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(payload, indent=2) + "\n")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            if self.path.exists():
                self.path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()
            return
        fd, temp_name = tempfile.mkstemp(prefix="visiontrace-", suffix=".faiss", dir=self.path.parent)
        os.close(fd)
        try:
            faiss.write_index(self.index, temp_name)
            os.replace(temp_name, self.path)
            self._write_meta()
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def _validated_vectors(vectors: list[list[float]], dimension: int | None = None) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("Embedding batch must be a non-empty two-dimensional array")
        expected = dimension or OSNET_EMBEDDING_DIM
        if array.shape[1] != expected:
            raise ValueError(f"Embedding dimension mismatch: expected {expected}, received {array.shape[1]}")
        if not np.isfinite(array).all():
            raise ValueError("Embeddings must contain only finite values")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Zero-length embeddings cannot be indexed")
        return np.ascontiguousarray(array / norms, dtype=np.float32)

    def _known_ids(self) -> set[int]:
        if self.index is None or self.index.ntotal == 0:
            return set()
        return set(faiss.vector_to_array(self.index.id_map).astype(np.int64).tolist())

    def add(self, vector_ids: list[int], vectors: list[list[float]], embedding_space: str | None = None) -> int:
        if len(vector_ids) != len(vectors):
            raise ValueError("The vector ID and embedding counts must match")
        if not vector_ids:
            return 0
        if len(set(vector_ids)) != len(vector_ids) or any(value <= 0 for value in vector_ids):
            raise ValueError("Vector IDs must be unique positive integers")
        space = embedding_space or EMBEDDING_SPACE

        with self.lock:
            matrix = self._validated_vectors(vectors, self.dimension or OSNET_EMBEDDING_DIM)
            if self.index is None:
                self.dimension = int(matrix.shape[1])
                self.index = self._new_index(self.dimension)
                self.embedding_space = space
            elif self.embedding_space and self.embedding_space != space:
                raise ValueError(
                    f"Index embedding space {self.embedding_space} is incompatible with {space}"
                )
            known = self._known_ids()
            fresh = [(identifier, row) for identifier, row in zip(vector_ids, matrix) if identifier not in known]
            if not fresh:
                return 0
            ids = np.asarray([item[0] for item in fresh], dtype=np.int64)
            batch = np.ascontiguousarray([item[1] for item in fresh], dtype=np.float32)
            self.index.add_with_ids(batch, ids)
            self._save()
            return len(fresh)

    def search(self, vector: list[float], top_k: int = 10) -> list[dict[str, float | int]]:
        if not 1 <= top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000")
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            query = self._validated_vectors([vector], self.dimension)
            scores, ids = self.index.search(query, min(top_k, int(self.index.ntotal)))
            results: list[dict[str, float | int]] = []
            for score, vector_id in zip(scores[0], ids[0]):
                if vector_id < 0 or not np.isfinite(score):
                    continue
                results.append({"vectorId": int(vector_id), "similarity": float(score)})
            return results

    def remove(self, vector_ids: list[int]) -> int:
        if not vector_ids:
            return 0
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return 0
            selector = faiss.IDSelectorBatch(np.asarray(vector_ids, dtype=np.int64))
            removed = int(self.index.remove_ids(selector))
            if self.index.ntotal == 0:
                self.index = None
                self.dimension = None
                self.embedding_space = None
            self._save()
            return removed

    def rebuild(
        self,
        vector_ids: list[int],
        vectors: list[list[float]],
        embedding_space: str | None = None,
    ) -> int:
        if len(vector_ids) != len(vectors):
            raise ValueError("The vector ID and embedding counts must match")
        if len(set(vector_ids)) != len(vector_ids):
            raise ValueError("Vector IDs must be unique")
        if any(value <= 0 for value in vector_ids):
            raise ValueError("Vector IDs must be unique positive integers")
        space = embedding_space or EMBEDDING_SPACE
        with self.lock:
            self.index = None
            self.dimension = None
            self.embedding_space = None
            if vector_ids:
                matrix = self._validated_vectors(vectors, OSNET_EMBEDDING_DIM)
                self.dimension = int(matrix.shape[1])
                self.index = self._new_index(self.dimension)
                self.embedding_space = space
                self.index.add_with_ids(matrix, np.asarray(vector_ids, dtype=np.int64))
            self._save()
            return len(vector_ids)


vector_index = FaissVectorIndex()
