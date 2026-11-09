"""Holdout evaluation. Papers, abstracts and records are TEST DATA written by hand, not real research."""
import csv
import json
from pathlib import Path

import pytest

from config.settings import load_settings
from src.evaluation.holdout import (RATING_COLUMNS, SHEET_COLUMNS, auto_checks, evaluate_heldout, pick_heldout,
                                    similarity_matrix, summarize_ratings)
from src.ingestion.paper_store import PaperStore
from src.main import build_retriever, main
from src.models import Paper
from src.retrieval.embeddings import EmbeddingError, HashingEmbedder
from tests.test_arxiv_client import SAMPLE_FEED, make_client
from tests.test_dashboard import FULL

EMBED = HashingEmbedder().embed


def paper(pid, published, title="T", abstract="An abstract about graphs."):
    return Paper(source="arxiv", paper_id=pid, title=title, abstract=abstract, published=published, url=f"https://arxiv.org/abs/{pid}")


# ------------------------------------------------------------------ choosing the held-out papers
def test_only_papers_newer_than_the_whole_index_and_not_in_it_are_held_out():
    indexed = [paper("1", "2026-10-01T00:00:00Z"), paper("2", "2026-10-30T00:00:00Z")]
    fetched = [paper("9", "2026-11-05T00:00:00Z"),                       # new: kept
               paper("2", "2026-10-30T00:00:00Z"),                       # already indexed
               paper("8", "2026-10-30T00:00:00Z"),                       # not newer than the newest indexed paper
               paper("7", "2026-11-04T00:00:00Z", abstract="   "),       # no abstract
               paper("9", "2026-11-05T00:00:00Z"),                       # duplicate
               paper("6", "2026-11-03T00:00:00Z"),
               paper("5", "2026-11-02T00:00:00Z")]
    chosen, cutoff = pick_heldout(fetched, indexed, n=2)
    assert [p.paper_id for p in chosen] == ["9", "6"] and cutoff == "2026-10-30T00:00:00Z"


def test_nothing_is_held_out_when_nothing_is_newer():
    assert pick_heldout([paper("1", "2026-01-01T00:00:00Z")], [paper("2", "2026-02-01T00:00:00Z")], 3)[0] == []


# ------------------------------------------------------------------ automatic checks and similarity
def test_auto_checks_read_the_saved_record():
    c = auto_checks(FULL)
    assert c["hypothesis_cites_retrieved_paper"] is True and c["invalid_citations"] == 1
    assert c["execution"] == "SUCCESS" and c["critic_verdict"] == "INCONCLUSIVE" and c["design_mismatch"] is False
    assert c["hypothesis"].startswith("A random forest")
    noisy = dict(FULL, warnings=["Experiment design: Possible mismatch with the hypothesis: x"])
    assert auto_checks(noisy)["design_mismatch"] is True


def test_a_hypothesis_is_more_similar_to_its_own_abstract_than_to_another():
    sims = similarity_matrix(["graph neural networks for molecules", "random forest tree depth"],
                             ["graph neural networks molecules and chemistry", "random forest decision tree depth study"], EMBED)
    assert sims[0][0] > sims[0][1] and sims[1][1] > sims[1][0]


# ------------------------------------------------------------------ the evaluation run
PAPERS = [paper("100", "2026-11-05T00:00:00Z", "Graph neural networks for molecules", "graph neural networks molecules chemistry"),
          paper("101", "2026-11-06T00:00:00Z", "Random forest tree depth", "random forest decision tree depth study"),
          paper("102", "2026-11-07T00:00:00Z", "A third paper", "something else entirely")]


def make_runner(tmp_path, fail_on=None):
    seen = []

    def run_topic(topic):
        seen.append(topic)
        if topic == fail_on:
            raise RuntimeError("model crashed")
        record = json.loads(json.dumps(FULL))
        record["hypothesis"]["statement"] = topic + " study"
        path = tmp_path / f"run_{len(seen):03d}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path
    return run_topic, seen


def test_evaluation_writes_results_a_rating_sheet_and_a_summary_with_limits(tmp_path):
    run_topic, seen = make_runner(tmp_path)
    out = evaluate_heldout(PAPERS, run_topic, EMBED, tmp_path / "out", cutoff="2026-11-01")
    assert seen == [p.title for p in PAPERS]
    rows = out["rows"]
    assert all(r["similarity_to_own_abstract"] is not None and r["own_abstract_rank"] >= 1 for r in rows)
    assert rows[0]["own_abstract_rank"] == 1 and rows[1]["own_abstract_rank"] == 1
    assert out["summary"]["runs_finished"] == 3 and out["summary"]["own_abstract_ranked_first"] >= 2
    sheet = list(csv.DictReader((tmp_path / "out" / "rating_sheet.csv").open(encoding="utf-8-sig")))
    assert list(sheet[0].keys()) == SHEET_COLUMNS and len(sheet) == 3
    assert all(r[c] == "" for r in sheet for c in RATING_COLUMNS)           # left blank for a human
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "Limits of this evaluation" in summary and "tiny" in summary and "rough proxy" in summary
    saved = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert [h["id"] for h in saved["held_out"]] == ["arxiv:100", "arxiv:101", "arxiv:102"] and "abstract" in saved["held_out"][0]


def test_one_failed_run_does_not_stop_the_evaluation(tmp_path):
    run_topic, _ = make_runner(tmp_path, fail_on="Random forest tree depth")
    out = evaluate_heldout(PAPERS, run_topic, EMBED, tmp_path / "out")
    assert out["summary"]["runs_finished"] == 2 and out["summary"]["runs_failed"] == 1
    assert "RuntimeError: model crashed" in out["rows"][1]["error"] and "similarity_to_own_abstract" not in out["rows"][1]


def test_results_are_saved_after_every_run(tmp_path):
    run_topic, _ = make_runner(tmp_path)
    counts = []

    def spying(topic):
        results = tmp_path / "out" / "results.json"
        counts.append(len(json.loads(results.read_text(encoding="utf-8"))["rows"]) if results.exists() else 0)
        return run_topic(topic)
    evaluate_heldout(PAPERS, spying, EMBED, tmp_path / "out")
    assert counts == [0, 1, 2]


def test_an_embedding_failure_is_recorded_not_fatal(tmp_path):
    def broken(_texts):
        raise EmbeddingError("embedding model missing")
    run_topic, _ = make_runner(tmp_path)
    out = evaluate_heldout(PAPERS, run_topic, broken, tmp_path / "out")
    assert out["summary"]["runs_finished"] == 3 and out["summary"]["mean_similarity_own"] is None
    assert "similarity not computed" in out["rows"][0]["error"] and (tmp_path / "out" / "rating_sheet.csv").exists()


# ------------------------------------------------------------------ the human ratings
def test_ratings_are_averaged_and_bad_entries_are_reported(tmp_path):
    sheet = tmp_path / "sheet.csv"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["held_out_id"] + RATING_COLUMNS)
        w.writeheader()
        w.writerow({"held_out_id": "a", "relevance_1to5": "4", "testability_1to5": "2", "groundedness_1to5": ""})
        w.writerow({"held_out_id": "b", "relevance_1to5": "2", "testability_1to5": "7", "groundedness_1to5": "abc"})
    text = summarize_ratings(sheet)
    assert "relevance_1to5: 2 rated, mean 3.00" in text and "testability_1to5: 1 rated, mean 2.00" in text
    assert "'7'" in text and "'abc'" in text and "groundedness_1to5: 0 rated, no ratings yet" in text


# ------------------------------------------------------------------ the arXiv call and the command
def test_latest_asks_arxiv_for_a_category_newest_first_and_checks_the_category():
    client = make_client([__import__("tests.test_arxiv_client", fromlist=["FakeResponse"]).FakeResponse(SAMPLE_FEED)])
    papers = client.latest("cs.LG", max_results=5)
    params = client.session.calls[0]["params"] if isinstance(client.session.calls[0], dict) else client.session.calls[0][1]
    assert len(papers) == 2 and params["search_query"] == "cat:cs.LG" and params["sortOrder"] == "descending"
    with pytest.raises(ValueError):
        client.latest("cs.LG; drop", 5)


class FakeArxiv:
    def __init__(self, *a, **k):
        pass

    def latest(self, category, max_results=30):
        return [paper("900", "2026-11-09T00:00:00Z", "Graph learning today", "graph learning methods today"),
                paper("901", "2026-11-10T00:00:00Z", "Tree methods today", "tree ensemble methods today")]


@pytest.fixture()
def indexed_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setattr("src.main.ArxivClient", FakeArxiv)
    settings = load_settings()
    store, retriever = build_retriever(settings)
    old = [paper("1", "2026-10-01T00:00:00Z", "Graph neural networks", "graph neural networks learning"),
           paper("2", "2026-10-02T00:00:00Z", "Decision trees", "decision tree forest ensemble")]
    store.upsert_many(old)
    retriever.index_papers(old)
    store.close()
    return settings


def test_evaluate_with_an_empty_index_explains_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    assert main(["evaluate", "--dry-run"]) == 1 and "index is empty" in capsys.readouterr().out


def test_evaluate_dry_run_lists_the_papers_and_adds_nothing_to_the_index(indexed_env, capsys):
    assert main(["evaluate", "--dry-run", "--n", "2"]) == 0
    out = capsys.readouterr().out
    assert "arxiv:900" in out and "arxiv:901" in out and "Dry run" in out
    store = PaperStore(indexed_env.papers_db_path)
    assert store.count() == 2 and store.get("arxiv:900") is None            # held-out papers never enter the index
    store.close()


def test_evaluate_runs_end_to_end_with_the_mock_model_and_keeps_papers_out_of_the_index(indexed_env, monkeypatch, capsys):
    monkeypatch.setattr("src.main.make_sandbox", lambda settings: None)     # no Docker in the test
    assert main(["evaluate", "--n", "2", "-k", "2"]) == 0
    out = capsys.readouterr().out
    folders = list((indexed_env.storage_dir / "evaluation").glob("holdout_*"))
    assert len(folders) == 1 and "Finished 2 of 2 run(s)" in out
    rows = list(csv.DictReader((folders[0] / "rating_sheet.csv").open(encoding="utf-8-sig")))
    assert [r["held_out_id"] for r in rows] == ["arxiv:900", "arxiv:901"] and all(r["run_file"] for r in rows)
    assert len(list(indexed_env.runs_dir.glob("run_*.json"))) == 2
    store = PaperStore(indexed_env.papers_db_path)
    assert store.count() == 2
    store.close()
    assert main(["evaluate", "--ratings", str(folders[0] / "rating_sheet.csv")]) == 0
    assert "0 rated" in capsys.readouterr().out


def test_two_runs_saved_in_the_same_second_do_not_overwrite_each_other(indexed_env, monkeypatch):
    monkeypatch.setattr("src.workflow.time.strftime", lambda fmt: "20260101_000000")
    assert main(["run", "--topic", "graph learning", "--no-execute", "--no-memory"]) == 0
    assert main(["run", "--topic", "decision trees", "--no-execute", "--no-memory"]) == 0
    names = sorted(f.name for f in indexed_env.runs_dir.glob("run_*.json"))
    assert names == ["run_20260101_000000.json", "run_20260101_000000_2.json"]
