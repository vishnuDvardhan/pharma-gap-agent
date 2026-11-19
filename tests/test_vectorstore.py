"""Vector DB: one contract, tested against every backend (ChromaDB only when installed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gapagent.config import load_config
from gapagent.loader import load_dossier, load_requirements
from gapagent.pipeline import GapChecker, run_gap_check
from gapagent.retrieval import Retriever, candidate_indices, requirement_query
from gapagent.vectorstore import (
    BACKENDS,
    DOSSIER_COLLECTION,
    HashingEmbedder,
    LocalVectorStore,
    index_dossier,
    make_embedder,
    make_vector_store,
    matches,
)

KINDS = ["local"] + (["chroma"] if importlib.util.find_spec("chromadb") else [])
DOCS = {"a": ("Long-term stability at 30°C/75% RH for three batches", {"module": "m3", "region": None}),
        "b": ("Form FDA 356h signed by the applicant", {"module": "m1", "region": "us"}),
        "c": ("EU application form, Module 1", {"module": "m1", "region": "eu"})}


def _store(kind, path, embedder=None):
    return BACKENDS[kind]("test_col", embedder or HashingEmbedder(), path)


def _fill(store):
    return store.upsert(list(DOCS), [t for t, _ in DOCS.values()], [m for _, m in DOCS.values()])


@pytest.mark.parametrize("kind", KINDS)
def test_upsert_query_and_metadata_filters(kind, tmp_path):
    s = _store(kind, tmp_path)
    assert vars(_fill(s)) == {"added": 3, "updated": 0, "unchanged": 0, "deleted": 0}
    assert s.query("stability 30 degrees 75% RH", 1)[0].id == "a"
    # region filter: a US query must never return the EU form, even when it is the closest text
    assert [h.id for h in s.query("application form", 3, {"module": "m1", "region": "us"})] == ["b"]
    assert {h.id for h in s.query("form", 3, {"region": {"$in": ["us", "eu"]}})} == {"b", "c"}
    assert {h.id for h in s.query("form", 3, {"module": "m1", "region": {"$ne": "eu"}})} == {"b"}
    assert s.query("x", 3, {"module": "m5"}) == []


@pytest.mark.parametrize("kind", KINDS)
def test_incremental_upsert_persistence_and_sync(kind, tmp_path):
    s = _store(kind, tmp_path)
    _fill(s)
    st = s.upsert(["a", "b"], ["Long-term stability at 30°C/75% RH for three batches", "Form 356h, now updated"],
                  [{"module": "m3"}, {"module": "m1", "region": "us"}])
    assert (st.unchanged, st.updated, st.added) == (1, 1, 0)       # only changed text is re-embedded
    reopened = _store(kind, tmp_path)                               # new process / client, same path
    assert reopened.count() == 3
    assert "updated" in reopened.query("356h", 1)[0].text
    st = reopened.sync(["a"], [DOCS["a"][0]], [DOCS["a"][1]], scope={})   # b and c disappeared from source
    assert st.deleted == 2 and reopened.ids() == ["a"]


@pytest.mark.parametrize("kind", KINDS)
def test_embedder_change_rebuilds_collection(kind, tmp_path):
    _fill(_store(kind, tmp_path, HashingEmbedder(1024)))
    s = _store(kind, tmp_path, HashingEmbedder(512))     # incompatible vectors must not be mixed
    assert s.count() == 0
    assert _fill(s).added == 3


def test_local_store_without_path_is_memory_only(tmp_path):
    s = LocalVectorStore("t", HashingEmbedder(), None)
    _fill(s)
    assert s.count() == 3 and not any(tmp_path.iterdir())


def test_filter_language():
    m = {"module": "m1", "region": "us"}
    assert matches(m, {"module": "m1", "region": "us"}) and not matches(m, {"region": "eu"})
    assert matches(m, {"$and": [{"module": "m1"}, {"region": {"$nin": ["eu", "jp"]}}]})
    with pytest.raises(ValueError):
        matches(m, {"region": {"$regex": "u"}})


def test_tfidf_cannot_back_a_persistent_store():
    with pytest.raises(ValueError, match="cannot back a vector store"):
        make_embedder("tfidf")


@pytest.mark.parametrize("kind", KINDS)
def test_vector_db_retrieval_matches_in_memory_and_keeps_region_filter(kind, tmp_path):
    cfg = load_config(vector_store=kind, vector_path=str(tmp_path), embedder="hash")
    d = load_dossier(Path(cfg["data_dir"]) / "dossiers" / "zelvorin_ref")
    store = make_vector_store(cfg, DOSSIER_COLLECTION)
    assert index_dossier(store, d).added == len(d.chunks)
    assert index_dossier(store, d).unchanged == len(d.chunks)
    mem = Retriever(d.chunks, "hybrid", "tfidf")
    vdb = Retriever(d.chunks, "hybrid", vector_store=store)
    for r in [x for rs in load_requirements(Path(cfg["data_dir"])).values() for x in rs]:
        idx, _ = candidate_indices(d.chunks, r)
        got = vdb.search(requirement_query(r), 3, idx)
        assert {h.chunk.chunk_id for h in got} <= {d.chunks[i].chunk_id for i in idx}   # filter respected
        assert got[0].chunk.file == mem.search(requirement_query(r), 3, idx)[0].chunk.file if idx else not got
        if r.id.startswith("US-M1"):
            assert all(h.chunk.region in (None, "us") for h in got)


def test_pipeline_uses_vector_db_and_reindexes_incrementally(cfg):
    g = GapChecker(cfg, "zelvorin_global")
    assert g.vector_store is not None and g.index_stats.added > 0
    g2 = GapChecker(cfg, "zelvorin_global")
    assert g2.index_stats.added == g2.index_stats.updated == 0
    # same verdicts with the in-memory index
    mem = dict(cfg, retrieval={**cfg["retrieval"], "vector_store": "none", "embedder": "tfidf"})
    a = {(f.market, f.req_id): f.verdict for f in run_gap_check(cfg, "zelvorin_global", ["US", "IN", "JP"]).findings}
    b = {(f.market, f.req_id): f.verdict for f in run_gap_check(mem, "zelvorin_global", ["US", "IN", "JP"]).findings}
    assert a == b
