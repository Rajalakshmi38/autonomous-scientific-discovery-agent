"""Safe deletion of saved runs, topic validation, the real progress events and the CLI. All data here is TEST DATA."""
import json

import pytest

from src.dashboard.manage import DeleteResult, delete_all_runs, delete_run, draft_files_for, is_run_file_name
from src.dashboard.runner import format_event, friendly_error, validate_topic
from src.main import main
from src.workflow import PIPELINE_STEPS, ProgressEvent


# ------------------------------------------------------------------ deletion
@pytest.fixture()
def storage(tmp_path):
    """A storage folder in the real layout: two runs with drafts, plus every shared item."""
    s = tmp_path / "storage"
    for sub in ("runs", "drafts", "logs", "vectors", "evaluation/holdout_1"):
        (s / sub).mkdir(parents=True)
    for stem in ("run_20260101_000001", "run_20260101_000001_2", "run_20260102_000002"):
        (s / "runs" / f"{stem}.json").write_text(json.dumps({"topic": stem}), encoding="utf-8")
    for name in ("run_20260101_000001_draft.md", "run_20260101_000001_draft.tex",
                 "run_20260102_000002_failed_run_report.md", "run_20260101_000001_2_draft.md"):
        (s / "drafts" / name).write_text("draft", encoding="utf-8")
    shared = {"papers.db": "db", "memory.jsonl": "{}", "bandit.json": "{}", "watchlist.json": "[]", "logs/app.log": "log",
              "vectors/vectors.npy": "v", "vectors/meta.json": "{}", "evaluation/holdout_1/results.json": "{}"}
    for rel, text in shared.items():
        (s / rel).write_text(text, encoding="utf-8")
    return s


def shared_files(s):
    return sorted(str(p.relative_to(s)) for p in s.rglob("*") if p.is_file() and p.parts[len(s.parts)] not in ("runs", "drafts"))


def test_deleting_one_run_removes_only_its_record_and_drafts(storage):
    before_shared = shared_files(storage)
    result = delete_run(storage / "runs", storage / "drafts", "run_20260101_000001.json")
    assert result.ok and sorted(result.deleted) == ["run_20260101_000001.json", "run_20260101_000001_draft.md", "run_20260101_000001_draft.tex"]
    assert sorted(p.name for p in (storage / "runs").iterdir()) == ["run_20260101_000001_2.json", "run_20260102_000002.json"]
    # the neighbour with a similar name (_2) and the other run keep their drafts
    assert sorted(p.name for p in (storage / "drafts").iterdir()) == ["run_20260101_000001_2_draft.md", "run_20260102_000002_failed_run_report.md"]
    assert shared_files(storage) == before_shared                      # papers, vectors, memory, log, bandit, evaluation untouched


def test_deleting_a_run_that_is_already_gone_is_not_an_error(storage):
    assert delete_run(storage / "runs", storage / "drafts", "run_20260102_000002.json").ok
    again = delete_run(storage / "runs", storage / "drafts", "run_20260102_000002.json")
    assert again.ok and again.missing == ["run_20260102_000002.json"] and not again.deleted


@pytest.mark.parametrize("bad", ["../storage/papers.db", "run_1.json", "runs/run_20260101_000001.json", "run_20260101_000001.json/..",
                                 "papers.db", "", "run_20260101_000001.json\x00", "run_20260101_000001.json\n", "*", "run_*.json"])
def test_a_name_that_is_not_a_run_file_deletes_nothing(storage, bad):
    before = sorted(str(p) for p in storage.rglob("*"))
    result = delete_run(storage / "runs", storage / "drafts", bad)
    assert not result.ok and not result.deleted and sorted(str(p) for p in storage.rglob("*")) == before
    assert not is_run_file_name(bad)


def test_a_draft_that_cannot_be_deleted_keeps_the_record(storage, monkeypatch):
    from pathlib import Path
    real = Path.unlink

    def flaky(self, *a, **k):
        if self.name.endswith("_draft.tex"):
            raise PermissionError(13, "Permission denied")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", flaky)
    result = delete_run(storage / "runs", storage / "drafts", "run_20260101_000001.json")
    assert not result.ok and any("Permission denied" in e for e in result.errors) and any("kept" in e for e in result.errors)
    assert (storage / "runs" / "run_20260101_000001.json").exists()    # still there, so the person can retry


def test_draft_files_for_uses_exact_names(storage):
    names = sorted(p.name for p in draft_files_for(storage / "drafts", "run_20260101_000001.json"))
    assert names == ["run_20260101_000001_draft.md", "run_20260101_000001_draft.tex"]
    assert draft_files_for(storage / "drafts", "not-a-run") == []


def test_clear_all_removes_every_run_record_and_draft_and_nothing_else(storage):
    (storage / "runs" / "run_20260103_000003.json").write_text("{broken", encoding="utf-8")     # unreadable record
    (storage / "drafts" / "run_20250101_000000_draft.md").write_text("orphan", encoding="utf-8")  # its record is already gone
    (storage / "runs" / "notes.txt").write_text("mine", encoding="utf-8")                        # not a run file
    (storage / "drafts" / "my_own_notes.md").write_text("mine", encoding="utf-8")
    before_shared = shared_files(storage)
    result = delete_all_runs(storage / "runs", storage / "drafts")
    assert result.ok and result.runs_deleted == 4
    assert [p.name for p in (storage / "runs").iterdir()] == ["notes.txt"]
    assert [p.name for p in (storage / "drafts").iterdir()] == ["my_own_notes.md"]
    assert shared_files(storage) == before_shared
    assert delete_all_runs(storage / "runs", storage / "drafts").ok     # a second click finds nothing and is harmless


def test_clear_all_with_missing_folders_is_harmless(tmp_path):
    assert delete_all_runs(tmp_path / "no_runs", tmp_path / "no_drafts") == DeleteResult()


def test_clear_all_reports_a_problem_and_still_deletes_what_it_can(storage, monkeypatch):
    from pathlib import Path
    real = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError(13, "Permission denied"))
                        if self.name == "run_20260102_000002.json" else real(self, *a, **k))
    result = delete_all_runs(storage / "runs", storage / "drafts")
    assert not result.ok and (storage / "runs" / "run_20260102_000002.json").exists()
    assert not (storage / "runs" / "run_20260101_000001.json").exists()


# ------------------------------------------------------------------ topic validation and messages
@pytest.mark.parametrize("raw", ["", "   ", None, "abc", "12345 678", "!!!! ????", "x" * 301, "bad\x00topic"])
def test_invalid_topics_are_rejected_with_a_reason(raw):
    check = validate_topic(raw)
    assert not check.ok and check.error and check.topic == ""


def test_a_valid_topic_is_cleaned_not_rewritten():
    check = validate_topic("  Diabetes   prediction\nusing machine learning ")
    assert check.ok and check.topic == "Diabetes prediction using machine learning"


def test_progress_lines_say_exactly_what_happened():
    started = format_event(ProgressEvent("review", "running"))
    assert "Step 2 of 6" in started and "running" in started and "finished" not in started
    assert "finished in 82.2 s" in format_event(ProgressEvent("review", "done", 82.2, "Cites 2 retrieved paper(s)."))
    assert "failed" in format_event(ProgressEvent("execute", "failed", 1.0, "boom")) and "boom" in format_event(ProgressEvent("execute", "failed", 1.0, "boom"))
    assert "not run" in format_event(ProgressEvent("critic", "skipped"))
    assert "with a warning" in format_event(ProgressEvent("hypothesis", "warning", 3))


def test_friendly_errors_name_the_step_and_the_likely_cause():
    assert "Ollama" in friendly_error(ConnectionError("Connection refused"), "Literature review")
    assert "Literature review" in friendly_error(RuntimeError("x"), "Literature review")
    assert "LLM_TIMEOUT_SECONDS" in friendly_error(TimeoutError("timed out"), None)
    assert "RuntimeError: odd" in friendly_error(RuntimeError("odd"))


# ------------------------------------------------------------------ real progress events from the workflow
def test_step_titles_match_the_dashboard():
    from src.dashboard.data import STEPS
    assert PIPELINE_STEPS == STEPS


@pytest.fixture()
def wf_factory(tmp_path):
    from src.agents.code_executor import CodeExecutorAgent
    from src.agents.hypothesis_generator import HypothesisGeneratorAgent
    from src.agents.literature_reviewer import LiteratureReviewerAgent
    from src.ingestion.paper_store import PaperStore
    from src.llm.stub_model import RuleBasedStubChatModel
    from src.models import Paper
    from src.retrieval.embeddings import HashingEmbedder
    from src.retrieval.retriever import PaperRetriever
    from src.retrieval.vector_store import NumpyVectorStore
    from src.workflow import ResearchWorkflow
    store = PaperStore(tmp_path / "p.db")
    corpus = [Paper("arxiv", "1", "Graph neural networks", "Message passing on graph neural networks.", url="u1")]
    store.upsert_many(corpus)
    retriever = PaperRetriever(store, HashingEmbedder(), NumpyVectorStore())
    retriever.index_papers(corpus)
    llm = RuleBasedStubChatModel()

    def make(**kw):
        return ResearchWorkflow(retriever, LiteratureReviewerAgent(llm), HypothesisGeneratorAgent(llm), CodeExecutorAgent(llm),
                                llm_mode="TEST MODE", top_k=3, **kw)
    yield make
    store.close()


def test_progress_events_follow_the_six_steps_in_order(wf_factory):
    events = []
    wf_factory(on_progress=events.append).run("graph neural networks")
    keys = [k for k, _ in PIPELINE_STEPS]
    assert [e.step for e in events] == [k for k in keys for _ in (0, 1)]                  # each step: one start, one end
    assert [e.phase for e in events][::2] == ["running"] * 6
    ends = {e.step: e for e in events if e.phase != "running"}
    assert ends["retrieve"].phase == "done" and "1 paper" in ends["retrieve"].message
    assert ends["execute"].phase == "skipped" and "disabled" in ends["execute"].message   # no sandbox was given: it did NOT run
    assert ends["critic"].phase == "skipped"                                              # nothing executed, so nothing to check
    assert all(e.seconds is not None for e in ends.values())
    # a step is never reported finished before it started
    seen = []
    for e in events:
        assert e.phase == "running" or e.step in seen
        seen.append(e.step)


def test_without_evidence_the_later_steps_are_reported_as_skipped(wf_factory):
    events = []
    result = wf_factory(on_progress=events.append).run("quantum chromodynamics lattice")
    ends = {e.step: e.phase for e in events if e.phase != "running"}
    assert result.status == "NO_EVIDENCE" and ends["retrieve"] == "warning"
    assert [ends[k] for k, _ in PIPELINE_STEPS[1:]] == ["skipped"] * 5


def test_a_failing_step_is_reported_failed_and_the_error_still_reaches_the_caller(wf_factory):
    events = []
    wf = wf_factory(on_progress=events.append)
    wf.reviewer.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model exploded"))
    with pytest.raises(RuntimeError, match="model exploded"):
        wf.run("graph neural networks")
    assert events[-1].step == "review" and events[-1].phase == "failed" and "model exploded" in events[-1].message
    assert not any(e.step in ("hypothesis", "execute") for e in events)                   # nothing after the failure was announced


def test_a_broken_progress_listener_never_breaks_the_run(wf_factory):
    def listener(_event):
        raise ValueError("ui bug")
    assert wf_factory(on_progress=listener).run("graph neural networks").status == "COMPLETED"


def test_the_workflow_without_a_listener_behaves_as_before(wf_factory):
    result = wf_factory().run("graph neural networks")
    assert result.status == "COMPLETED" and set(result.step_seconds) == {k for k, _ in PIPELINE_STEPS}


# ------------------------------------------------------------------ the CLI
@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setattr("src.main.make_sandbox", lambda s: None)
    return tmp_path / "s" / "runs"


def test_cli_run_accepts_the_topic_both_ways(cli_env, capsys):
    assert main(["run", "--topic", "graph neural networks", "--no-execute", "--no-memory"]) == 0       # the original form
    assert main(["run", "graph neural networks", "--no-execute", "--no-memory"]) == 0                  # the form in the brief
    assert main(["run", "graph neural networks", "--topic", "graph neural networks", "--no-execute", "--no-memory"]) == 0
    assert len(list(cli_env.glob("run_*.json"))) == 3
    assert "Saved run record" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["run"], ["run", "--topic", "  "], ["run", "topic one", "--topic", "topic two"]])
def test_cli_run_rejects_a_missing_or_ambiguous_topic(cli_env, argv, capsys):
    assert main(argv) == 2 and "input error" in capsys.readouterr().err and not list(cli_env.glob("run_*.json"))


def test_cli_and_dashboard_share_one_pipeline_function(cli_env, monkeypatch):
    calls = []
    import src.main as m
    real = m.run_research
    monkeypatch.setattr(m, "run_research", lambda settings, topic, **kw: calls.append((topic, kw)) or real(settings, topic, **kw))
    assert main(["run", "graph neural networks", "--no-execute", "--no-memory", "-k", "3"]) == 0
    assert calls and calls[0][0] == "graph neural networks" and calls[0][1]["k"] == 3 and calls[0][1]["execute"] is False
    assert calls[0][1]["use_memory"] is False


# ------------------------------------------------------------------ saving ratings and resetting the bandit
SHEET = ("held_out_id,held_out_title,relevance_1to5,testability_1to5,groundedness_1to5,rater,comments\r\n"
         "a1,Paper One,,,,,\r\na2,Paper Two,2,2,2,AI assistant,old comment\r\n")


def sheet(tmp_path):
    path = tmp_path / "rating_sheet.csv"
    path.write_bytes(SHEET.encode("utf-8-sig"))          # raw bytes: write_text would turn \r\n into \r\r\n on Windows
    return path


def read_rows(path):
    import csv
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def test_save_ratings_changes_only_the_rows_that_were_edited(tmp_path):
    from src.dashboard.insights import save_ratings
    path = sheet(tmp_path)
    scores = [{"relevance": 5, "testability": 4, "groundedness": 3, "comments": "good"},
              {"relevance": 2, "testability": 2, "groundedness": 2, "comments": "old comment"}]
    assert save_ratings(path, scores, "  Anu   K ") == 1
    rows = read_rows(path)
    assert (rows[0]["relevance_1to5"], rows[0]["rater"], rows[0]["comments"]) == ("5", "Anu K", "good")
    assert (rows[1]["rater"], rows[1]["comments"]) == ("AI assistant", "old comment")        # an earlier rating keeps its own rater
    assert (tmp_path / "rating_sheet.previous.csv").read_bytes() == SHEET.encode("utf-8-sig")      # the old sheet is kept exactly
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("bad", [0, 6, 12, "x"])
def test_save_ratings_rejects_values_outside_one_to_five(tmp_path, bad):
    from src.dashboard.insights import RatingError, save_ratings
    path = sheet(tmp_path)
    before = path.read_bytes()
    with pytest.raises(RatingError):
        save_ratings(path, [{"relevance": bad, "testability": 3, "groundedness": 3}, {"relevance": 2, "testability": 2, "groundedness": 2}], "Anu")
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_save_ratings_needs_a_rater_a_matching_sheet_and_the_expected_columns(tmp_path):
    from src.dashboard.insights import RatingError, save_ratings
    path = sheet(tmp_path)
    two = [{"relevance": 5, "testability": 4, "groundedness": 3}, {"relevance": 2, "testability": 2, "groundedness": 2}]
    with pytest.raises(RatingError, match="rater"):
        save_ratings(path, two, "   ")
    with pytest.raises(RatingError, match="changed while"):
        save_ratings(path, two[:1], "Anu")
    with pytest.raises(RatingError, match="no rating sheet"):
        save_ratings(tmp_path / "missing.csv", two, "Anu")
    (tmp_path / "odd.csv").write_text("a,b\r\n1,2\r\n", encoding="utf-8")
    with pytest.raises(RatingError, match="expected columns"):
        save_ratings(tmp_path / "odd.csv", [{}], "Anu")
    assert save_ratings(path, [{"relevance": None, "testability": None, "groundedness": None},
                               {"relevance": 2, "testability": 2, "groundedness": 2, "comments": "old comment"}], "") == 0   # nothing changed: no name needed


def test_reset_bandit_keeps_a_copy_and_is_harmless_when_there_is_nothing(tmp_path):
    from src.dashboard.insights import reset_bandit
    path = tmp_path / "bandit.json"
    assert reset_bandit(path) is False and not (tmp_path / "bandit.json.bak").exists()
    path.write_text('{"arms": {}}', encoding="utf-8")
    assert reset_bandit(path) is True and not path.exists() and (tmp_path / "bandit.json.bak").read_text(encoding="utf-8") == '{"arms": {}}'


# ------------------------------------------------------------------ search words for fetching papers
def test_fallback_queries_start_with_the_whole_topic_then_use_its_main_words():
    from src.dashboard.runner import fallback_queries
    assert fallback_queries("decision tree depth 1 versus depth 5 accuracy on the wine dataset") == [
        "decision tree depth 1 versus depth 5 accuracy on the wine dataset", "decision tree depth wine", "decision tree"]     # "depth" once
    assert fallback_queries("support vector machine with and without feature scaling on the wine dataset") == [
        "support vector machine with and without feature scaling on the wine dataset", "support vector machine feature", "support vector"]
    assert fallback_queries("  k-nearest   neighbors  ") == ["k-nearest neighbors"]                      # nothing shorter to try
    assert fallback_queries("graph neural networks") == ["graph neural networks"]
    assert fallback_queries("") == [""] and len(fallback_queries("a b c d e f g h i j k l")) == 1          # no words of 3+ letters: no invented queries
