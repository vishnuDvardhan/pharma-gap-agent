"""Vector database layer: persistent, incremental, metadata-filtered semantic search.

Two interchangeable backends behind one small interface (`VectorStore`):

* LocalVectorStore  - built in, numpy + JSON on disk, exact cosine search. No extra dependency,
                      so CI and offline demos use it. `path=None` keeps it in memory only.
* ChromaVectorStore - ChromaDB (`pip install -e ".[vectordb]"`), HNSW index, persistent client.

Both store one record per text: id, embedding, metadata, and a content hash. `upsert` only
re-embeds records whose text changed (incremental indexing), and `sync` also deletes records
that disappeared from the source (for example a removed dossier file). Each collection records
which embedder built it; opening it with a different embedder rebuilds it instead of silently
mixing incompatible vectors.

Filters use a small subset of the Chroma `where` language, so both backends accept the same
filters:
    {"dossier_id": "zelvorin_global", "module": "m1"}        # equality, AND-ed
    {"chunk_id": {"$in": [...]}}, {"region": {"$ne": "eu"}}  # $in / $nin / $eq / $ne
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NONE = ""   # metadata cannot hold None in Chroma; store missing values as ""


# ---------------------------------------------------------------- embedders
class HashingEmbedder:
    """Stateless char-n-gram embedder (hashing trick). Needs no download and no fitting.

    A fitted TF-IDF model changes its vectors whenever the corpus changes, so it cannot back a
    persistent index. Hashed n-grams give the same vector for the same text in every process.
    On the retrieval eval it matches the TF-IDF embedder exactly (see README).
    """

    def __init__(self, dim: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.dim = dim
        self.id = f"hash-char3-5-{dim}"
        self._v = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=dim,
                                    alternate_sign=False, norm=None)

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self._v.transform(texts)
        m.data = 1.0 + np.log(m.data)            # sublinear tf
        return _l2(m.toarray().astype(np.float32))


class SentenceTransformerEmbedder:
    """Neural embeddings, e.g. `st:BAAI/bge-small-en-v1.5` (needs sentence-transformers)."""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self._m = SentenceTransformer(model)
        self.id = f"st-{model}"
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float32)


def make_embedder(name: str):
    if name == "hash" or name.startswith("hash:"):
        return HashingEmbedder(int(name.split(":")[1]) if ":" in name else 1024)
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    raise ValueError(f"embedder {name!r} cannot back a vector store; use 'hash' or 'st:<model>' "
                     "('tfidf' must be re-fitted per corpus, so it only works with vector_store: none)")


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- filters
def _clean(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: (NONE if v is None else v) for k, v in meta.items()}


_OPS = {"$eq": lambda v, a: v == a, "$ne": lambda v, a: v != a,
        "$in": lambda v, a: v in a, "$nin": lambda v, a: v not in a}


def matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a where-filter against one metadata dict (used by the local backend)."""
    for key, cond in (where or {}).items():
        if key == "$and":
            if not all(matches(meta, c) for c in cond):
                return False
            continue
        val = meta.get(key, NONE)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op not in _OPS:
                    raise ValueError(f"unsupported filter operator {op}")
                if not _OPS[op](val, arg):
                    return False
        elif val != cond:
            return False
    return True


def to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma needs an explicit $and when more than one field is filtered."""
    if not where:
        return None
    clauses = [{k: v} for k, v in where.items() if k != "$and"] + list(where.get("$and", []))
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ---------------------------------------------------------------- interface
@dataclass
class VectorHit:
    id: str
    score: float                  # cosine similarity, higher is better
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpsertStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0

    def __iadd__(self, o: UpsertStats) -> UpsertStats:
        self.added += o.added
        self.updated += o.updated
        self.unchanged += o.unchanged
        self.deleted += o.deleted
        return self


class VectorStore(ABC):
    backend = "base"

    def __init__(self, collection: str, embedder):
        self.collection, self.embedder = collection, embedder

    # -- to implement
    @abstractmethod
    def _hashes(self, ids: list[str]) -> dict[str, str]:
        """Stored content hash for each id that exists."""

    @abstractmethod
    def _write(self, ids: list[str], vecs: np.ndarray, texts: list[str], metas: list[dict]) -> None: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int: ...

    @abstractmethod
    def ids(self, where: dict[str, Any] | None = None) -> list[str]: ...

    @abstractmethod
    def query_vector(self, vec: np.ndarray, k: int, where: dict[str, Any] | None = None) -> list[VectorHit]: ...

    @abstractmethod
    def count(self) -> int: ...

    # -- shared logic
    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> UpsertStats:
        """Insert or update records. Records whose text did not change are not re-embedded."""
        metadatas = metadatas or [{} for _ in ids]
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("ids, texts and metadatas must have the same length")
        hashes = [content_hash(t) for t in texts]
        existing = self._hashes(ids)
        todo = [i for i, (rid, h) in enumerate(zip(ids, hashes, strict=True)) if existing.get(rid) != h]
        stats = UpsertStats(unchanged=len(ids) - len(todo))
        if todo:
            stats.added = sum(1 for i in todo if ids[i] not in existing)
            stats.updated = len(todo) - stats.added
            vecs = self.embedder.encode([texts[i] for i in todo])
            self._write([ids[i] for i in todo], vecs, [texts[i] for i in todo],
                        [_clean({**metadatas[i], "content_hash": hashes[i]}) for i in todo])
        return stats

    def sync(self, ids: list[str], texts: list[str], metadatas: list[dict], scope: dict[str, Any]) -> UpsertStats:
        """Make the records matching `scope` equal to exactly this set (upsert + delete stale)."""
        stats = self.upsert(ids, texts, metadatas)
        stale = sorted(set(self.ids(scope)) - set(ids))
        stats.deleted = self.delete(stale) if stale else 0
        return stats

    def query(self, text: str, k: int = 5, where: dict[str, Any] | None = None) -> list[VectorHit]:
        return self.query_vector(self.embedder.encode([text])[0], k, where)


# ---------------------------------------------------------------- built-in backend
class LocalVectorStore(VectorStore):
    """Exact cosine search over a numpy matrix, persisted as `<path>/<collection>/{vectors.npy,records.json}`.

    Writes go to temp files and are swapped in with os.replace, so a crash mid-write never leaves a
    half-written index. A lock makes writes safe when several agents share one store.
    """
    backend = "local"

    def __init__(self, collection: str, embedder, path: str | os.PathLike | None = None):
        super().__init__(collection, embedder)
        self.dir = Path(path) / collection if path else None
        self._lock = threading.RLock()
        self.vecs = np.zeros((0, embedder.dim), dtype=np.float32)
        self.records: list[dict[str, Any]] = []      # {"id", "text", "metadata"}
        self.pos: dict[str, int] = {}
        if self.dir and (self.dir / "records.json").exists():
            info = json.loads((self.dir / "records.json").read_text())
            if info.get("embedder") == embedder.id:
                self.records = info["records"]
                self.vecs = np.load(self.dir / "vectors.npy")
                self.pos = {r["id"]: i for i, r in enumerate(self.records)}
            # else: built with another embedder -> start empty, next upsert rebuilds it

    def _persist(self) -> None:
        if not self.dir:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_v, tmp_r = self.dir / "vectors.tmp.npy", self.dir / "records.tmp.json"
        np.save(tmp_v, self.vecs)
        tmp_r.write_text(json.dumps({"embedder": self.embedder.id, "records": self.records}))
        os.replace(tmp_v, self.dir / "vectors.npy")
        os.replace(tmp_r, self.dir / "records.json")

    def _hashes(self, ids):
        return {i: self.records[self.pos[i]]["metadata"].get("content_hash") for i in ids if i in self.pos}

    def _write(self, ids, vecs, texts, metas):
        with self._lock:
            new_rows = []
            for rid, v, t, m in zip(ids, vecs, texts, metas, strict=True):
                rec = {"id": rid, "text": t, "metadata": m}
                if rid in self.pos:
                    self.vecs[self.pos[rid]] = v
                    self.records[self.pos[rid]] = rec
                else:
                    self.pos[rid] = len(self.records) + len(new_rows)
                    new_rows.append((rec, v))
            if new_rows:
                self.records.extend(r for r, _ in new_rows)
                self.vecs = np.vstack([self.vecs, np.stack([v for _, v in new_rows])])
            self._persist()

    def delete(self, ids):
        with self._lock:
            drop = {i for i in ids if i in self.pos}
            if not drop:
                return 0
            keep = [i for i, r in enumerate(self.records) if r["id"] not in drop]
            self.records = [self.records[i] for i in keep]
            self.vecs = self.vecs[keep]
            self.pos = {r["id"]: i for i, r in enumerate(self.records)}
            self._persist()
            return len(drop)

    def ids(self, where=None):
        return [r["id"] for r in self.records if matches(r["metadata"], where)]

    def count(self):
        return len(self.records)

    def query_vector(self, vec, k, where=None):
        rows = [i for i, r in enumerate(self.records) if matches(r["metadata"], where)]
        if not rows or k <= 0:
            return []
        scores = self.vecs[rows] @ vec
        order = np.argsort(-scores, kind="stable")[:k]
        return [VectorHit(self.records[rows[j]]["id"], float(scores[j]), self.records[rows[j]]["text"],
                          self.records[rows[j]]["metadata"]) for j in order]


# ---------------------------------------------------------------- ChromaDB backend
class ChromaVectorStore(VectorStore):
    """ChromaDB persistent collection (HNSW, cosine space). We pass our own embeddings so both
    backends use the same embedder and give comparable scores."""
    backend = "chroma"

    def __init__(self, collection: str, embedder, path: str | os.PathLike | None = None):
        super().__init__(collection, embedder)
        import chromadb  # optional dependency: pip install -e ".[vectordb]"
        from chromadb.config import Settings
        settings = Settings(anonymized_telemetry=False)
        self.client = (chromadb.PersistentClient(path=str(path), settings=settings) if path
                       else chromadb.EphemeralClient(settings=settings))
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine", "embedder": embedder.id}, embedding_function=None)
        if (self.col.metadata or {}).get("embedder") != embedder.id:   # built with another embedder: rebuild
            self.client.delete_collection(collection)
            self.col = self.client.create_collection(
                collection, metadata={"hnsw:space": "cosine", "embedder": embedder.id}, embedding_function=None)

    def _hashes(self, ids):
        if not ids:
            return {}
        got = self.col.get(ids=ids, include=["metadatas"])
        return {i: (m or {}).get("content_hash") for i, m in zip(got["ids"], got["metadatas"], strict=True)}

    def _write(self, ids, vecs, texts, metas):
        self.col.upsert(ids=ids, embeddings=[v.tolist() for v in vecs], documents=texts, metadatas=metas)

    def delete(self, ids):
        if ids:
            self.col.delete(ids=ids)
        return len(ids)

    def ids(self, where=None):
        return self.col.get(where=to_chroma_where(where), include=[])["ids"]

    def count(self):
        return self.col.count()

    def query_vector(self, vec, k, where=None):
        n = len(self.ids(where)) if where else self.count()
        if n == 0 or k <= 0:
            return []
        r = self.col.query(query_embeddings=[vec.tolist()], n_results=min(k, n), where=to_chroma_where(where),
                           include=["documents", "metadatas", "distances"])
        return [VectorHit(i, 1.0 - float(d), doc, meta or {})
                for i, d, doc, meta in zip(r["ids"][0], r["distances"][0], r["documents"][0], r["metadatas"][0], strict=True)]


# ---------------------------------------------------------------- factory + indexing helpers
BACKENDS = {"local": LocalVectorStore, "chroma": ChromaVectorStore}
_CACHE: dict[tuple, VectorStore] = {}
_CACHE_LOCK = threading.Lock()


def make_vector_store(cfg: dict, collection: str) -> VectorStore | None:
    """Store configured under `retrieval:`; None when `vector_store: none` (in-memory retrieval).

    Instances are cached per (backend, path, collection, embedder) so the API, the pipeline and
    every agent share one open index instead of reloading it per request.
    """
    rc = cfg["retrieval"]
    kind = rc.get("vector_store", "none")
    if kind == "none":
        return None
    if kind not in BACKENDS:
        raise ValueError(f"unknown vector_store {kind!r}; choose none | {' | '.join(BACKENDS)}")
    path = rc.get("vector_path")
    key = (kind, str(path), collection, rc.get("embedder", "hash"))
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = BACKENDS[kind](collection, make_embedder(rc.get("embedder", "hash")), path)
        return _CACHE[key]


DOSSIER_COLLECTION = "dossier_chunks"
REQUIREMENT_COLLECTION = "requirements"


def chunk_text(c) -> str:
    return f"{c.heading}\n{c.text}"


def index_dossier(store: VectorStore, dossier) -> UpsertStats:
    """Sync one dossier's chunks into the shared chunk collection (one collection, many dossiers)."""
    cs = dossier.chunks
    metas = [{"chunk_id": c.chunk_id, "dossier_id": c.dossier_id, "file": c.file, "path": c.path, "module": c.module, "region": c.region,
              "section": c.section, "heading": c.heading, "line_start": c.line_start, "line_end": c.line_end}
             for c in cs]
    return store.sync([c.chunk_id for c in cs], [chunk_text(c) for c in cs], metas,
                      scope={"dossier_id": dossier.dossier_id})


def requirement_text(r) -> str:
    return f"{r.id} {r.title}. {r.text} Evidence: {r.evidence_criteria} Source: {r.source.doc}, {r.source.clause}"


def index_requirements(store: VectorStore, reqs: list) -> UpsertStats:
    metas = [{"market": r.market, "ctd_section": r.ctd_section, "severity": r.severity, "check_type": r.check_type,
              "source": f"{r.source.doc}, {r.source.clause}", "verified": r.verified} for r in reqs]
    return store.sync([r.id for r in reqs], [requirement_text(r) for r in reqs], metas, scope={})
