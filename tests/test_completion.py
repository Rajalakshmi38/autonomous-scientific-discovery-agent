"""Published/preprint label, the strategy bandit and the Milvus adapter. All data here is TEST DATA."""
import copy
import json
import random
import sys

import numpy as np
import pytest

from config.settings import ConfigError, load_settings
from src.agents.base import format_evidence
from src.dashboard.data import citations
from src.draft.generator import build_draft, to_markdown
from src.ingestion.paper_store import PaperStore
from src.main import build_retriever, main
from src.models import Paper, RetrievedPaper, publication_status
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.milvus_store import MilvusVectorStore, collection_name
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore, VectorStoreError
from src.rl.bandit import StrategyBandit, reward_from_record
from tests.test_dashboard import FULL


def paper(pid, source="arxiv", title="T", abstract=None, **kw):
    return Paper(source=source, paper_id=pid, title=title, abstract=abstract or f"abstract about {title}",
                 published="2026-11-01T00:00:00Z", url=f"https://example.org/{pid}", **kw)


# ------------------------------------------------------------------ published or preprint
def test_publication_status_is_a_label_for_published_or_preprint():
    assert publication_status(paper("1")) == "preprint"
    assert publication_status(paper("2", journal_ref="Nature 1 (2026)")) == "published"
    assert publication_status(paper("3", doi="10.1000/x")) == "published"
    assert publication_status(paper("4", source="pubmed")) == "published"


def test_the_agents_see_the_status_of_every_paper():
    text = format_evidence([RetrievedPaper(paper("1", title="A"), 0.9), RetrievedPaper(paper("2", source="pubmed", title="B"), 0.8)])
    assert "Status: preprint" in text and "Status: published" in text


def make_retriever(tmp_path, papers, embedder=None):
    embedder = embedder or HashingEmbedder()
    store = PaperStore(tmp_path / "p.db")
    store.upsert_many(papers)
    retriever = PaperRetriever(store, embedder, NumpyVectorStore(None, embedder.name))
    retriever.index_papers(papers)
    return store, retriever


def test_published_only_retrieval_skips_preprints_but_still_finds_enough_papers(tmp_path):
    papers = [paper("1", title="graph networks graph"), paper("2", source="pubmed", title="graph networks medicine"),
              paper("3", title="graph networks theory", doi="10.1/x"), paper("4", title="unrelated cooking")]
    store, retriever = make_retriever(tmp_path, papers)
    both = [r.paper.uid for r in retriever.retrieve("graph networks", k=2)]
    only = [r.paper.uid for r in retriever.retrieve("graph networks", k=2, published_only=True)]
    store.close()
    assert "arxiv:1" in both                                           # the preprint is the closest paper
    assert sorted(only) == ["arxiv:3", "pubmed:2"]                     # deeper search found two published ones


def test_dashboard_and_draft_show_the_status_when_the_record_has_it():
    record = copy.deepcopy(FULL)
    record["evidence"][0]["status"] = "preprint"
    assert citations(record)["rows"][0]["status"] == "preprint" and citations(record)["rows"][1]["status"] == "not recorded"
    md = to_markdown(build_draft(record))
    assert "(preprint)" in md


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    return monkeypatch


@pytest.fixture()
def indexed(env):
    settings = load_settings()
    store, retriever = build_retriever(settings)
    papers = [paper("1", title="graph neural networks learning"), paper("2", source="pubmed", title="graph neural networks in medicine"),
              paper("3", title="decision tree forest ensemble")]
    store.upsert_many(papers)
    retriever.index_papers(papers)
    store.close()
    return settings


def test_search_prints_the_status_and_can_skip_preprints(indexed, capsys):
    assert main(["search", "--query", "graph neural networks", "-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "(preprint)" in out and "(published)" in out
    assert main(["search", "--query", "graph neural networks", "-k", "3", "--published-only"]) == 0
    out = capsys.readouterr().out
    assert "(published)" in out and "(preprint)" not in out


def test_run_records_the_status_of_each_paper(indexed, monkeypatch):
    monkeypatch.setattr("src.main.make_sandbox", lambda s: None)
    assert main(["run", "--topic", "graph neural networks", "--no-memory", "--no-execute"]) == 0
    record = json.loads(next(indexed.runs_dir.glob("run_*.json")).read_text(encoding="utf-8"))
    assert {e["status"] for e in record["evidence"]} <= {"preprint", "published"} and record["evidence"]


def test_new_settings_are_read_and_checked(env):
    env.setenv("RETRIEVAL_PUBLISHED_ONLY", "true")
    env.setenv("BANDIT_EPSILON", "0.5")
    s = load_settings()
    assert s.retrieval_published_only is True and s.bandit_epsilon == 0.5 and s.vector_backend == "local"
    for name, value in (("VECTOR_BACKEND", "pinecone"), ("BANDIT_EPSILON", "2")):
        env.setenv(name, value)
        with pytest.raises(ConfigError):
            load_settings()
        env.delenv(name)
        env.setenv("BANDIT_EPSILON", "0.5")
    env.setenv("VECTOR_BACKEND", "milvus")
    with pytest.raises(ConfigError, match="MILVUS_URI"):
        load_settings()


# ------------------------------------------------------------------ the bandit
GOOD = {"execution_status": "SUCCESS", "hypothesis_grounding": {"cited_ids": ["arxiv:1"], "unknown_ids": []},
        "critic": {"verdict": "INCONCLUSIVE"}, "experiment": {"note": ""}, "warnings": []}


def test_reward_comes_from_automatic_checks_only():
    assert reward_from_record(GOOD) == 1.0
    assert reward_from_record(dict(GOOD, execution_status="TIMEOUT", critic={"verdict": "NO_RESULT"})) == pytest.approx(1 / 3)
    assert reward_from_record(dict(GOOD, hypothesis_grounding={"cited_ids": ["a"], "unknown_ids": ["x"]})) == pytest.approx(2 / 3)
    assert reward_from_record(dict(GOOD, warnings=["Experiment design: Possible mismatch with the hypothesis: x"])) == pytest.approx(2 / 3)
    assert reward_from_record({}) == 0.0


def test_every_arm_is_tried_once_then_the_best_is_used(tmp_path):
    b = StrategyBandit(tmp_path / "b.json", epsilon=0.0, rng=random.Random(1))
    first = []
    for reward in (0.2, 0.9, 0.5):
        arm, _ = b.choose()
        first.append(arm)
        b.update(arm, reward)
    assert first == [2, 3, 5] and b.best_arm() == 3
    assert b.choose() == (3, "best average reward so far")


def test_the_bandit_learns_on_simulated_rewards_and_beats_a_random_strategy(tmp_path):
    truth = {2: 0.3, 3: 0.5, 5: 0.8}                    # simulated chance of a good run for each retrieval depth
    sim = random.Random(7)
    bandit = StrategyBandit(tmp_path / "b.json", epsilon=0.1, rng=random.Random(3))
    chosen, total = [], 0.0
    for _ in range(400):
        arm, _ = bandit.choose()
        reward = 1.0 if sim.random() < truth[arm] else 0.0
        bandit.update(arm, reward)
        chosen.append(arm)
        total += reward
    rnd = random.Random(11)
    random_total = sum(1.0 if sim.random() < truth[rnd.choice([2, 3, 5])] else 0.0 for _ in range(400))
    assert bandit.best_arm() == 5 and chosen[-100:].count(5) >= 70
    assert total > random_total + 30                    # simulation only: this says nothing about real runs


def test_the_bandit_is_saved_reloaded_and_checks_its_input(tmp_path):
    path = tmp_path / "b.json"
    b = StrategyBandit(path, epsilon=0.0)
    b.update(3, 0.5)
    b.update(3, 1.0)
    assert not list(tmp_path.glob("*.tmp"))
    again = StrategyBandit(path, epsilon=0.0)
    assert again.stats["3"] == {"n": 2, "reward": 1.5} and again.total_runs() == 2 and "average reward 0.75" in "\n".join(again.describe())
    for bad in (lambda: b.update(9, 0.5), lambda: b.update(3, 2.0), lambda: StrategyBandit(path, arms=[]),
                lambda: StrategyBandit(path, epsilon=3)):
        with pytest.raises(ValueError):
            bad()
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="damaged"):
        StrategyBandit(path)


def test_adaptive_runs_choose_a_strategy_learn_from_the_run_and_can_be_reset(indexed, monkeypatch, capsys):
    monkeypatch.setattr("src.main.make_sandbox", lambda s: None)
    tried = []
    for _ in range(3):
        assert main(["run", "--topic", "graph neural networks", "--no-memory", "--no-execute", "--adaptive"]) == 0
        tried.append([line for line in capsys.readouterr().out.splitlines() if line.startswith("Adaptive strategy: retrieve")][0])
    assert [t.split()[3] for t in tried] == ["2", "3", "5"]                # each arm is tried once first
    assert main(["bandit"]) == 0
    out = capsys.readouterr().out
    assert "3 run(s) recorded" in out and "Best so far" in out and "not proof" in out
    assert main(["bandit", "--reset"]) == 0 and main(["bandit"]) == 0
    assert "0 run(s) recorded" in capsys.readouterr().out


# ------------------------------------------------------------------ the Milvus adapter
class FakeMilvus:
    """Just enough of MilvusClient to test the adapter's logic on any platform."""

    def __init__(self):
        self.data = {}
        self.fail = None

    def _check(self):
        if self.fail:
            raise RuntimeError(self.fail)

    def has_collection(self, name):
        self._check()
        return name in self.data

    def create_collection(self, name, dimension=None, **kw):
        self.data[name] = {"dim": dimension, "rows": {}, "kw": kw}

    def upsert(self, name, rows):
        for r in rows:
            assert len(r["vector"]) == self.data[name]["dim"]
            self.data[name]["rows"][r["id"]] = np.asarray(r["vector"], dtype=float)

    def search(self, name, data, limit, output_fields):
        q = np.asarray(data[0], dtype=float)
        scored = sorted(((float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q))), i) for i, v in self.data[name]["rows"].items()), reverse=True)
        return [[{"id": i, "distance": s, "entity": {}} for s, i in scored[:limit]]]

    def query(self, name, filter, output_fields):
        return [{"count(*)": len(self.data[name]["rows"])}]

    def drop_collection(self, name):
        del self.data[name]


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_milvus_adapter_adds_searches_counts_upserts_and_clears():
    fake = FakeMilvus()
    store = MilvusVectorStore("fake://", "hashing-1024", client=fake)
    assert store.count() == 0 and store.search(unit(1, 0), 3) == []
    store.add(["a", "b"], np.stack([unit(1, 0), unit(0, 1)]))
    assert store.count() == 2
    top = store.search(unit(1, 0.1), 2)
    assert [i for i, _ in top] == ["a", "b"] and top[0][1] > top[1][1]
    store.add(["a"], np.stack([unit(0, 1)]))                     # same id again: replaced, not duplicated
    assert store.count() == 2 and store.search(unit(0, 1), 1)[0][0] in ("a", "b")
    assert fake.data[store.collection]["kw"]["metric_type"] == "COSINE"
    store.clear()
    assert store.count() == 0 and store.collection not in fake.data


def test_each_embedder_gets_its_own_collection_with_a_safe_name():
    assert collection_name("ollama-nomic-embed-text") == "asda_papers_ollama_nomic_embed_text"
    fake = FakeMilvus()
    a, b = MilvusVectorStore("x", "hashing-1024", client=fake), MilvusVectorStore("x", "ollama-nomic", client=fake)
    a.add(["1"], np.stack([unit(1, 0)]))
    assert a.count() == 1 and b.count() == 0


def test_milvus_errors_become_vector_store_errors():
    fake = FakeMilvus()
    store = MilvusVectorStore("x", "e", client=fake)
    with pytest.raises(VectorStoreError, match="matching lengths"):
        store.add(["a", "b"], np.stack([unit(1, 0)]))
    fake.fail = "server unreachable"
    with pytest.raises(VectorStoreError, match="server unreachable"):
        store.count()


def test_a_missing_milvus_package_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(VectorStoreError, match="requirements-milvus.txt"):
        MilvusVectorStore("http://localhost:19530", "e")


def real_milvus_lite(path):
    pytest.importorskip("pymilvus")
    try:
        from pymilvus import MilvusClient
        MilvusClient(str(path))
    except Exception as exc:         # Milvus Lite is not available on Windows
        pytest.skip(f"Milvus Lite is not available here: {exc}")


def test_the_real_milvus_client_works_through_the_whole_retrieval_path(tmp_path, env):
    db = tmp_path / "milvus.db"
    real_milvus_lite(db)
    env.setenv("VECTOR_BACKEND", "milvus")
    env.setenv("MILVUS_URI", str(db))
    settings = load_settings()
    store, retriever = build_retriever(settings)
    papers = [paper("1", title="graph neural networks learning"), paper("2", title="decision tree forest ensemble")]
    store.upsert_many(papers)
    assert retriever.index_papers(papers) == 2 and retriever.vector_store.count() == 2
    assert retriever.retrieve("graph neural networks", k=1)[0].paper.uid == "arxiv:1"
    retriever.reindex_all()
    assert retriever.vector_store.count() == 2
    store.close()
