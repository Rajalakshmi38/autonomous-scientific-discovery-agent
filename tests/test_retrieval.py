"""Retrieval tests. Papers below are FICTIONAL TEST DATA."""
import numpy as np
import pytest

from config.settings import ConfigError, load_settings
from src.ingestion.paper_store import PaperStore
from src.models import Paper
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore, VectorStoreError


def paper(pid, title, abstract):
    return Paper(source="arxiv", paper_id=pid, title=title, abstract=abstract)


CORPUS = [
    paper("1", "Graph neural networks for molecules", "We study message passing graph neural networks on molecular property prediction."),
    paper("2", "Bayesian optimisation of hyperparameters", "Gaussian process surrogate models guide hyperparameter search efficiently."),
    paper("3", "Protein folding with transformers", "Attention based transformer models predict protein structure."),
]


def test_hashing_embedder_shape_determinism_and_norm():
    emb = HashingEmbedder(dim=256)
    a, b = emb.embed(["graph neural networks"]), emb.embed(["graph neural networks"])
    assert a.shape == (1, 256)
    assert np.array_equal(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5
    assert emb.embed([]).shape == (0, 256)


def test_empty_text_gives_zero_vector_not_nan():
    vec = HashingEmbedder(dim=64).embed([""])[0]
    assert not np.isnan(vec).any() and np.linalg.norm(vec) == 0


def test_vector_store_search_upsert_and_dimension_check():
    store = NumpyVectorStore()
    store.add(["a", "b"], np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert store.search(np.array([1, 0]), k=1)[0][0] == "a"
    store.add(["a"], np.array([[0, 1]], dtype=np.float32))          # replace "a"
    assert store.count() == 2
    with pytest.raises(VectorStoreError):
        store.add(["c"], np.array([[1, 0, 0]], dtype=np.float32))   # wrong dim
    assert NumpyVectorStore().search(np.array([1, 0])) == []         # empty store


def test_vector_store_persistence_and_embedder_mismatch(tmp_path):
    s1 = NumpyVectorStore(tmp_path, embedder_name="x")
    s1.add(["a"], np.array([[1, 0]], dtype=np.float32))
    assert NumpyVectorStore(tmp_path, embedder_name="x").count() == 1
    with pytest.raises(VectorStoreError, match="reindex"):
        NumpyVectorStore(tmp_path, embedder_name="different")


@pytest.fixture
def retriever(tmp_path):
    store = PaperStore(tmp_path / "p.db")
    store.upsert_many(CORPUS)
    emb = HashingEmbedder()
    r = PaperRetriever(store, emb, NumpyVectorStore(tmp_path / "v", emb.name))
    r.index_papers(CORPUS)
    yield r
    store.close()


def test_retriever_ranks_relevant_paper_first(retriever):
    results = retriever.retrieve("graph neural networks molecular", k=3)
    assert results[0].paper.paper_id == "1"
    assert results[0].score > 0


def test_retriever_drops_unrelated_query(retriever):
    assert retriever.retrieve("quantum chromodynamics lattice", k=3) == []


def test_retriever_rejects_blank_query(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_reindex_rebuilds_from_store(retriever):
    assert retriever.reindex_all() == 3
    assert retriever.vector_store.count() == 3


def test_settings_validation():
    with pytest.raises(ConfigError):
        load_settings(env={"LLM_PROVIDER": "nonsense"}, load_env_file=False)
    with pytest.raises(ConfigError):
        load_settings(env={"ARXIV_RETRIES": "abc"}, load_env_file=False)
    assert load_settings(env={}, load_env_file=False).is_test_mode
