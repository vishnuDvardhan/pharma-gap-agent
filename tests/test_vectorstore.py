"""Vector DB: one contract, tested against every backend (ChromaDB only when installed)."""
from __future__ import annotations

import importlib.util

import pytest

from gapagent.vectorstore import (
    BACKENDS,
    HashingEmbedder,
    LocalVectorStore,
    make_embedder,
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
