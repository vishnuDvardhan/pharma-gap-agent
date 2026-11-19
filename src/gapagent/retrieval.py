"""Retrieval: BM25 (lexical), embeddings (TF-IDF fallback or sentence-transformers), hybrid via RRF.

Requirement -> evidence retrieval runs per requirement, with a CTD section + region filter
applied BEFORE scoring. That filter is the main precision lever; eval/eval_retrieval.py
measures it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .loader import section_matches
from .schema import Chunk, Requirement

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.dl = np.array([len(t) for t in self.toks], dtype=float)
        self.avgdl = float(self.dl.mean()) if len(docs) else 0.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            out[i] = s
        return out


class Embedder:
    """`tfidf` works offline with no model download. `st:<model>` uses sentence-transformers."""

    def __init__(self, name: str = "tfidf"):
        self.name = name
        self._st = None
        self._vec = None
        if name.startswith("st:"):
            from sentence_transformers import SentenceTransformer  # optional dependency
            self._st = SentenceTransformer(name[3:])

    def fit(self, docs: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(docs, normalize_embeddings=True))
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        m = self._vec.fit_transform(docs)
        return _l2(m.toarray())

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(texts, normalize_embeddings=True))
        return _l2(self._vec.transform(texts).toarray())


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    """Hybrid retriever. The dense half is either an in-memory matrix (`vector_store=None`) or a
    persistent vector DB (`vectorstore.py`), queried with a metadata filter that restricts it to
    the same candidate chunks the section/region filter allowed."""

    def __init__(self, chunks: list[Chunk], method: str = "hybrid", embedder: str = "tfidf", rrf_k: int = 60,
                 vector_store=None):
        if method not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"unknown retrieval method {method}")
        self.chunks, self.method, self.rrf_k = chunks, method, rrf_k
        docs = [f"{c.heading}\n{c.text}" for c in chunks]
        self.bm25 = BM25(docs) if method in {"bm25", "hybrid"} else None
        self.store = vector_store if method in {"dense", "hybrid"} else None
        self.embedder = Embedder(embedder) if method in {"dense", "hybrid"} and self.store is None else None
        self.emb = self.embedder.fit(docs) if self.embedder else None

    def _dense_scores(self, query: str, idx: np.ndarray) -> np.ndarray:
        if self.store is None:
            return self.emb[idx] @ self.embedder.encode([query])[0]
        ids = [self.chunks[i].chunk_id for i in idx]
        hits = self.store.query(query, k=len(ids), where={"chunk_id": {"$in": ids}})
        by_id = {h.id: h.score for h in hits}
        return np.array([by_id.get(cid, -1.0) for cid in ids])  # not returned => ranked last

    def search(self, query: str, k: int = 5, candidates: list[int] | None = None) -> list[Hit]:
        idx = np.arange(len(self.chunks)) if candidates is None else np.asarray(candidates, dtype=int)
        if len(idx) == 0:
            return []
        rankings = []
        if self.bm25 is not None:
            s = self.bm25.scores(query)[idx]
            rankings.append(s)
        if self.emb is not None or self.store is not None:
            rankings.append(self._dense_scores(query, idx))
        if len(rankings) == 1:
            final = rankings[0]
        else:  # reciprocal rank fusion
            final = np.zeros(len(idx))
            for s in rankings:
                ranks = np.empty(len(s))
                ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
                final += 1.0 / (self.rrf_k + ranks)
        order = np.argsort(-final, kind="stable")[:k]
        return [Hit(self.chunks[idx[i]], float(final[i])) for i in order]


def requirement_query(req: Requirement) -> str:
    return f"{req.title}. {req.text} {req.evidence_criteria}"


def candidate_indices(chunks: list[Chunk], req: Requirement, section_filter: bool = True) -> tuple[list[int], bool]:
    """Indices of chunks that may hold evidence for `req`.

    Returns (indices, used_fallback). Module 1 of OTHER regions is never a candidate when the
    requirement is regional: an EU application form must not satisfy the US Form 356h check.
    """
    def region_ok(c: Chunk) -> bool:
        if c.module != "m1":
            return True
        if req.region is None:
            return True
        return c.region == req.region

    if not section_filter:  # ablation baseline: no metadata filtering at all
        return list(range(len(chunks))), False
    allowed = [i for i, c in enumerate(chunks) if region_ok(c)]
    filtered = [i for i in allowed if section_matches(req.ctd_section, chunks[i].section)]
    if filtered:
        return filtered, False
    if req.region is not None:
        # regional requirement: search the whole regional Module 1 before giving up
        regional = [i for i in allowed if chunks[i].module == "m1" and chunks[i].region == req.region]
        return regional, True
    return allowed, True
