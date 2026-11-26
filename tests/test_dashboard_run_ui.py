"""The dashboard page driven like a user: start research, watch progress, delete runs, clear all, write a draft.

The slow pipeline (Ollama + Docker) is replaced by a stand-in that emits the same progress events and writes a run record
in the real shape. TEST DATA only: these tests prove the page's wiring, not the quality of any research.
"""
import json
from pathlib import Path

import pytest

from src.workflow import PIPELINE_STEPS, ProgressEvent
from tests.test_dashboard import full

APP = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")
NEW_RUN = "run_20991231_235959.json"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    s = tmp_path / "s"
    for sub in ("runs", "drafts", "logs", "vectors"):
        (s / sub).mkdir(parents=True)
    for rel in ("papers.db", "memory.jsonl", "bandit.json", "logs/app.log", "vectors/vectors.npy"):
        (s / rel).write_text("shared", encoding="utf-8")
    return s


def add_run(storage, name, **changes):
    (storage / "runs" / name).write_text(json.dumps(full(**changes)), encoding="utf-8")


@pytest.fixture()
def two_runs(env):
    add_run(env, "run_20260101_000001.json", topic="first topic")
    add_run(env, "run_20260102_000002.json", topic="second topic")
    return env


class FakePipeline:
    """Stands in for src.main.run_research and records how often it was called."""
    def __init__(self, storage, monkeypatch, fail_at=None, mock_model=False):
        self.calls, self.kwargs, self.storage, self.fail_at, self.mock_model = [], [], storage, fail_at, mock_model
        import src.main as m
        monkeypatch.setattr(m, "run_research", self)
        monkeypatch.setattr(m, "llm_problem", lambda settings: None)

    def __call__(self, settings, topic, **kw):
        self.calls.append(topic)
        self.kwargs.append(kw)
        on_progress = kw["on_progress"]
        if kw.get("adaptive"):
            kw["notify"]("Adaptive strategy: retrieve 2 papers (exploring an untried option).")
        for key, _ in PIPELINE_STEPS:
            on_progress(ProgressEvent(key, "running"))
            if key == self.fail_at:
                on_progress(ProgressEvent(key, "failed", 1.5, "RuntimeError: model exploded"))
                raise RuntimeError("model exploded")
            on_progress(ProgressEvent(key, "done", 1.0, "ok"))
        record = full(topic=topic, created_at="2099-12-31T23:59:59")
        if self.mock_model:
            record["llm_mode"] = "TEST MODE (mock model)"
        (self.storage / "runs" / NEW_RUN).write_text(json.dumps(record), encoding="utf-8")
        from types import SimpleNamespace
        return SimpleNamespace(path=self.storage / "runs" / NEW_RUN, k=2 if kw.get("adaptive") else kw.get("k"),
                               reward=1.0 if kw.get("adaptive") else None,
                               result=SimpleNamespace(status="COMPLETED", execution_status="SUCCESS", llm_mode=record["llm_mode"],
                                                      warnings=["Code check: a fake warning"]))


def open_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.fixture()
def table(monkeypatch):
    """Tells the page which row of the Runs table is selected (AppTest cannot click a row). table.pick(0) = the first row."""
    import streamlit as st
    from types import SimpleNamespace
    state = {"rows": []}
    real = st.dataframe

    def dataframe(*args, **kwargs):
        shown = real(*args, **kwargs)
        if kwargs.get("on_select"):
            return SimpleNamespace(selection=SimpleNamespace(rows=list(state["rows"]), columns=[], cells=[]))
        return shown
    monkeypatch.setattr(st, "dataframe", dataframe)
    return SimpleNamespace(pick=lambda *rows: state.update(rows=list(rows)))


def go(at, section):
    """Choose a section in the sidebar, like clicking it."""
    next(b for b in at.sidebar.button if b.label == section).click().run(timeout=60)         # a menu item in the sidebar
    return at


def open_all_runs():
    return go(open_app(), "Runs")


def button(at, label=None, key=None):
    return next(b for b in at.button if (b.label == label if label else b.key == key))


def run_files(storage):
    return sorted(p.name for p in (storage / "runs").glob("run_*.json"))


def shared_intact(storage):
    return all((storage / rel).read_text(encoding="utf-8") == "shared"
               for rel in ("papers.db", "memory.jsonl", "bandit.json", "logs/app.log", "vectors/vectors.npy"))


def submit(at, topic):
    at.text_input(key="topic_input").set_value(topic)
    button(at, "Start Research").click().run(timeout=60)
    return at


# ------------------------------------------------------------------ start new research
def test_the_page_shows_the_new_subtitle_and_no_toy_dataset_wording(two_runs):
    at = open_app()
    assert not at.exception
    assert any("autonomous AI research platform" in c.value for c in at.caption)
    page_text = " ".join(str(e.value) for e in list(at.caption) + list(at.markdown) + list(at.subheader))
    assert "toy dataset" not in page_text.lower() and "read-only" not in page_text.lower()
    assert at.session_state["section"] == "Start New Research"                      # the app opens on the page that starts research
    assert any(s.value == "Start New Research" for s in at.subheader)
    go(at, "Runs")
    assert not any(s.value == "Start New Research" for s in at.subheader)             # other pages do not repeat the form
    assert [b.label for b in at.sidebar.button] == ["Start New Research", "Runs", "Logic tree", "Literature Citations",
                                                    "Execution Performance & Costs", "Draft", "Evaluation", "Learning (bandit)"]
    assert not at.sidebar.selectbox and not at.sidebar.checkbox and not at.sidebar.radio              # the sidebar is only a menu
    assert not any(b.key.startswith("pick_") for b in at.sidebar.button)       # no separate saved-runs list: All Runs manages runs


@pytest.mark.parametrize("topic", ["", "   ", "ab", "1234 5678"])
def test_an_invalid_topic_shows_an_error_and_starts_nothing(two_runs, monkeypatch, topic):
    fake = FakePipeline(two_runs, monkeypatch)
    at = submit(open_app(), topic)
    assert not at.exception and at.error and fake.calls == [] and run_files(two_runs) == ["run_20260101_000001.json", "run_20260102_000002.json"]


def test_a_valid_topic_runs_the_pipeline_once_saves_and_selects_the_new_run(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert not at.exception
    assert fake.calls == ["Diabetes Prediction Using Machine Learning"]
    assert NEW_RUN in run_files(two_runs)
    assert at.selectbox(key="selected_run").value == NEW_RUN                              # the new run is selected...
    assert any("Diabetes Prediction" in s.value for s in at.subheader)                    # ...and its results are shown
    assert any("Research finished" in s.value and NEW_RUN in s.value for s in at.success)
    assert any("a fake warning" in w.value for w in at.warning)                           # pipeline warnings are not hidden
    assert at.session_state["section"] == "Logic tree"                                  # the new run's steps are shown straight away
    assert any("Diabetes Prediction" in o for o in at.selectbox(key="selected_run").options)   # the run list is refreshed
    # the progress box lists all six real steps, each reported by the pipeline
    last = at.session_state["last_job"]
    assert last["ok"] and len(last["lines"]) == 6 and all("finished" in line for line in last["lines"])
    assert NEW_RUN in list(go(at, "Runs").dataframe[0].value["file"])                     # the Runs table is refreshed too


def test_reruns_selecting_and_tab_use_never_start_another_run(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    for _ in range(3):
        at.run()                                                                          # plain reruns
    at.selectbox(key="selected_run").select("run_20260101_000001.json").run()            # choosing an old run
    go(at, "Runs")
    go(at, "Draft")
    button(at, "Write draft").click().run()
    assert not at.exception and fake.calls == ["Diabetes Prediction Using Machine Learning"]


def test_double_submission_while_a_run_is_pending_is_ignored(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    at = open_app()
    at.session_state["pending_topic"] = "Diabetes Prediction Using Machine Learning"     # a run is already starting
    at.text_input(key="topic_input").set_value("Another topic about genes")
    button(at, "Start Research").click()
    # run the script once more by hand: the first run of the script consumed the pending topic exactly once
    at.run(timeout=60)
    assert fake.calls.count("Diabetes Prediction Using Machine Learning") == 1 and len(fake.calls) <= 2


def test_a_pipeline_failure_is_shown_names_the_step_and_saves_nothing(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch, fail_at="review")
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert not at.exception and fake.calls
    assert any("Literature review" in e.value and "model exploded" in e.value for e in at.error)
    assert NEW_RUN not in run_files(two_runs) and len(run_files(two_runs)) == 2
    assert at.session_state["last_job"]["ok"] is False
    assert not button(at, "Start Research").disabled                                           # the form is usable again
    assert "pending_topic" not in at.session_state                                                   # nothing left pending


def test_when_the_model_is_unavailable_nothing_runs_and_the_reason_is_shown(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    import src.main as m
    monkeypatch.setattr(m, "llm_problem", lambda settings: "Ollama is not running at http://localhost:11434.")
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert fake.calls == [] and any("Ollama is not running" in e.value for e in at.error) and len(run_files(two_runs)) == 2


def test_a_mock_model_run_is_still_visible_after_it_finishes(two_runs, monkeypatch):
    FakePipeline(two_runs, monkeypatch, mock_model=True)
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert at.selectbox(key="selected_run").value == NEW_RUN and any("TEST MODE" in w.value for w in at.warning)


def test_a_run_that_was_cut_off_is_reported_not_hidden(two_runs):
    at = open_app()
    at.session_state["running_topic"] = "Some topic that was running"
    at.run()
    assert any("did not finish" in w.value and "Some topic" in w.value for w in at.warning)
    at.run()
    assert not any("did not finish" in w.value for w in at.warning)                       # shown once


def test_an_empty_store_still_lets_you_start_research(env, monkeypatch):
    fake = FakePipeline(env, monkeypatch)
    at = open_app()
    assert not at.exception and any("No run records yet" in i.value for i in at.info)
    assert any(b.label == "Start Research" for b in at.button)
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert fake.calls and at.selectbox(key="selected_run").value == NEW_RUN


# ------------------------------------------------------------------ deleting one run
def test_deleting_from_the_runs_table_asks_first_then_deletes_only_that_run(two_runs, monkeypatch, table):
    fake = FakePipeline(two_runs, monkeypatch)
    (two_runs / "drafts" / "run_20260102_000002_draft.md").write_text("d", encoding="utf-8")
    at = open_all_runs()
    assert button(at, key="runs_delete").disabled and button(at, key="runs_open").disabled        # nothing selected yet
    table.pick(0)                                                                                  # the newest run (run_20260102...)
    at.run()
    assert any("run_20260102_000002.json" in c.value for c in at.caption if c.value.startswith("Selected"))
    button(at, key="runs_delete").click().run()
    assert run_files(two_runs) == ["run_20260101_000001.json", "run_20260102_000002.json"]    # nothing deleted by the first click
    assert any("Delete this research run?" in w.value for w in at.warning)
    button(at, key="confirm_delete_yes").click().run()
    assert not at.exception and run_files(two_runs) == ["run_20260101_000001.json"]
    assert not (two_runs / "drafts" / "run_20260102_000002_draft.md").exists() and shared_intact(two_runs)
    assert any("Deleted run run_20260102_000002.json" in s.value for s in at.success)
    assert at.session_state["current_run"] == "run_20260101_000001.json"                  # another run is selected instead
    assert list(at.dataframe[0].value["file"]) == ["run_20260101_000001.json"]            # gone from the table
    assert go(at, "Logic tree").selectbox(key="selected_run").value == "run_20260101_000001.json"
    assert fake.calls == []                                                               # deleting never researches


def test_cancel_keeps_the_run(two_runs, table):
    at = open_all_runs()
    table.pick(0)
    at.run()
    button(at, key="runs_delete").click().run()
    button(at, key="confirm_delete_no").click().run()
    assert len(run_files(two_runs)) == 2 and not any("Delete this research run?" in w.value for w in at.warning)
    at.run()
    assert len(run_files(two_runs)) == 2


def test_a_reload_after_confirming_does_not_delete_a_second_run(two_runs, table):
    at = open_all_runs()
    table.pick(0)
    at.run()
    button(at, key="runs_delete").click().run()
    button(at, key="confirm_delete_yes").click().run()
    for _ in range(3):
        at.run()
    assert run_files(two_runs) == ["run_20260101_000001.json"]                            # one click, one deletion


def test_the_runs_table_has_every_column_and_deleting_keeps_the_other_run(two_runs, monkeypatch, table):
    fake = FakePipeline(two_runs, monkeypatch)
    at = open_all_runs()
    frame = at.dataframe[0].value
    assert [c for c in frame.columns if c in ("time", "topic", "mode", "execution", "verdict", "seconds", "file")] == \
        ["time", "topic", "mode", "execution", "verdict", "seconds", "file"]
    table.pick(1)                                                                              # the older run
    at.run()
    button(at, key="runs_delete").click().run()
    button(at, key="confirm_delete_yes").click().run()
    assert run_files(two_runs) == ["run_20260102_000002.json"] and shared_intact(two_runs) and fake.calls == []
    assert list(at.dataframe[0].value["file"]) == ["run_20260102_000002.json"]


def test_a_selection_never_points_at_another_run_after_a_deletion(two_runs, table):
    at = open_all_runs()
    table.pick(0)
    at.run()
    button(at, key="runs_delete").click().run()
    button(at, key="confirm_delete_yes").click().run()
    table.pick()                                                                               # the new table starts with no row selected
    at.run()
    assert button(at, key="runs_delete").disabled and button(at, key="runs_open").disabled


def test_deleting_a_run_that_vanished_meanwhile_is_reported_calmly(two_runs, table):
    at = open_all_runs()
    table.pick(0)
    at.run()
    button(at, key="runs_delete").click().run()
    (two_runs / "runs" / "run_20260102_000002.json").unlink()                              # removed outside the dashboard
    button(at, key="confirm_delete_yes").click().run()
    assert not at.exception and any("already deleted" in i.value for i in at.info)


def test_deleting_the_last_run_shows_the_empty_state(env, table):
    add_run(env, "run_20260101_000001.json")
    at = open_all_runs()
    table.pick(0)
    at.run()
    button(at, key="runs_delete").click().run()
    button(at, key="confirm_delete_yes").click().run()
    assert not at.exception and run_files(env) == [] and any("No run records yet" in i.value for i in at.info)
    assert any(b.label == "Start Research" for b in go(at, "Start New Research").button) and shared_intact(env)


# ------------------------------------------------------------------ clear all
def test_clear_all_needs_explicit_confirmation_and_keeps_shared_data(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    (two_runs / "drafts" / "run_20260102_000002_draft.md").write_text("d", encoding="utf-8")
    at = open_all_runs()
    button(at, key="clear_all_tab").click().run()
    assert len(run_files(two_runs)) == 2                                                  # asking deletes nothing
    assert any("all 2 saved research run" in w.value and "cannot be undone" in w.value for w in at.warning)
    button(at, key="confirm_clear_yes").click().run()
    assert not at.exception and run_files(two_runs) == [] and list((two_runs / "drafts").iterdir()) == []
    assert shared_intact(two_runs) and fake.calls == []
    assert any("Deleted all saved research runs (2)" in s.value for s in at.success)
    assert any("No run records yet" in i.value for i in at.info)
    at.run()
    assert run_files(two_runs) == []


def test_clear_all_can_be_cancelled_and_is_also_offered_above_the_all_runs_table(two_runs):
    at = open_all_runs()
    button(at, key="clear_all_tab").click().run()
    button(at, key="confirm_clear_no").click().run()
    assert len(run_files(two_runs)) == 2 and not any("Clear all saved research runs?" in w.value for w in at.warning)


# ------------------------------------------------------------------ the existing tabs, improved
def test_the_citations_table_keeps_every_column(two_runs):
    at = go(open_app(), "Literature Citations")
    frame = at.dataframe[0].value
    assert list(frame.columns) == ["id", "title", "similarity", "status", "cited by review", "cited by hypothesis", "link"]
    assert any("Similarity is the search score" in c.value for c in at.caption)


def test_performance_tab_keeps_the_numbers_and_the_free_local_model_note(two_runs):
    at = go(open_app(), "Execution Performance & Costs")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total time (s)"] == "197.2" and metrics["Sandbox attempts"] == "1" and metrics["Money"] == "None"
    assert "Token counts are not recorded" in next(m for m in at.metric if m.label == "Money").help      # now in the Money tooltip
    assert "Execution Performance & Costs" in [b.label for b in at.sidebar.button]


def test_draft_is_written_from_the_record_shown_and_stays_after_a_rerun(two_runs):
    at = go(open_app(), "Draft")
    button(at, "Write draft").click().run()
    assert not at.exception and any("written" in s.value for s in at.success)
    assert list((two_runs / "drafts").glob("run_20260102_000002_draft.*"))                # the existing generator wrote the files
    assert any("without a language model" in m.value for m in at.markdown)
    at.run()
    assert any("written" in s.value for s in at.success)                                  # not lost on the next rerun


def test_a_failed_run_gets_a_failed_run_report(env):
    add_run(env, "run_20260101_000001.json", evidence=[], status="NO_EVIDENCE", execution=None, critic=None, hypothesis=None, experiment=None)
    at = go(open_app(), "Draft")
    button(at, "Write draft").click().run()
    assert not at.exception and any("Failed run report written" in s.value for s in at.success)
    assert list((env / "drafts").glob("*failed_run_report*"))


def test_the_page_survives_an_unreadable_run_file(two_runs):
    (two_runs / "runs" / "run_20260103_000003.json").write_text("{broken", encoding="utf-8")
    at = go(open_app(), "Runs")
    assert not at.exception and any("1 unreadable run file(s) are not listed" in c.value for c in at.caption)


# ------------------------------------------------------------------ the REAL pipeline behind the page (mock model, no Docker)
@pytest.fixture()
def real_env(tmp_path, monkeypatch):
    """An empty storage folder with a real, indexed paper library (no placeholder files)."""
    pytest.importorskip("streamlit")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    from config.settings import load_settings
    from src.main import build_retriever
    from src.models import Paper
    store, retriever = build_retriever(load_settings())
    papers = [Paper("arxiv", "1", "Graph neural networks", "Message passing on graph neural networks.", url="https://arxiv.org/abs/1"),
              Paper("arxiv", "2", "Graph learning", "Learning on graph structured data with neural networks.", url="https://arxiv.org/abs/2")]
    store.upsert_many(papers)
    retriever.index_papers(papers)
    store.close()
    return tmp_path / "s"


def paper_count(storage):
    from src.ingestion.paper_store import PaperStore
    store = PaperStore(storage / "papers.db")
    try:
        return store.count()
    finally:
        store.close()


def test_the_real_pipeline_runs_from_the_page_and_the_run_is_saved_and_shown(real_env, monkeypatch):
    """Nothing is faked except the language model (the project's own mock) and Docker (reported as unavailable)."""
    from types import SimpleNamespace
    monkeypatch.setattr("src.main.make_sandbox", lambda s: SimpleNamespace(
        check_available=lambda: SimpleNamespace(ok=False, message="Docker is not running (test).")))
    at = submit(open_app(), "graph neural networks")
    assert not at.exception
    saved = run_files(real_env)
    assert len(saved) == 1                                                       # exactly one real run record was written
    record = json.loads((real_env / "runs" / saved[0]).read_text(encoding="utf-8"))
    assert record["topic"] == "graph neural networks" and record["status"] == "COMPLETED"
    assert set(record["step_seconds"]) == {k for k, _ in PIPELINE_STEPS}
    assert at.selectbox(key="selected_run").value == saved[0] and any("TEST MODE" in w.value for w in at.warning)
    lines = at.session_state["last_job"]["lines"]
    assert len(lines) == 6 and "finished" in lines[0] and "not run" in lines[4] and "Docker is not running" in lines[4]   # honest: nothing was executed
    assert any("Docker is not running" in w.value for w in at.warning)                                                    # and the warning is shown
    assert not button(go(at, "Start New Research"), "Start Research").disabled
    for _ in range(2):
        at.run()
    assert run_files(real_env) == saved                                           # reruns never created another run
    go(at, "Draft")
    button(at, "Write draft").click().run()
    assert not at.exception and any("written" in s.value for s in at.success)
    # deleting that run removes it and its drafts but not the paper library
    go(at, "Runs")
    at.session_state["_runs_ver"] = 0
    import streamlit as st
    from types import SimpleNamespace
    real = st.dataframe
    monkeypatch.setattr(st, "dataframe", lambda *a, **k: SimpleNamespace(selection=SimpleNamespace(rows=[0], columns=[], cells=[])) if k.get("on_select") else real(*a, **k))
    at.run()
    button(at, key="runs_delete").click().run()
    button(at, key="confirm_delete_yes").click().run()
    assert run_files(real_env) == [] and list((real_env / "drafts").glob("run_*")) == []
    assert paper_count(real_env) == 2 and (real_env / "vectors").exists()


def test_the_cli_run_still_writes_a_record_the_dashboard_can_show(real_env, monkeypatch):
    from src.main import main
    monkeypatch.setattr("src.main.make_sandbox", lambda s: None)
    assert main(["run", "graph neural networks", "--no-execute", "--no-memory"]) == 0
    at = go(open_app(), "Logic tree")
    assert not at.exception and any("graph neural networks" in s.value for s in at.subheader)


def test_a_second_run_is_refused_while_another_is_in_progress(two_runs, monkeypatch):
    """The lock is shared by every browser tab on the server: one research run at a time."""
    import streamlit as st
    fake = FakePipeline(two_runs, monkeypatch)

    class HeldLock:                                   # another session is already running research
        def acquire(self, blocking=True):
            return False

        def release(self):
            raise AssertionError("a lock that was never acquired must not be released")
    import sys
    import threading
    real_lock = threading.Lock
    # only the dashboard's own lock is replaced; Streamlit's internal locks stay real
    monkeypatch.setattr(threading, "Lock", lambda: HeldLock() if sys._getframe(1).f_code.co_filename.replace("\\", "/").endswith("dashboard/app.py") else real_lock())
    st.cache_resource.clear()
    try:
        at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
        assert not at.exception and fake.calls == [] and any("already in progress" in w.value for w in at.warning)
        assert len(run_files(two_runs)) == 2 and not button(at, "Start Research").disabled
    finally:
        st.cache_resource.clear()


# ------------------------------------------------------------------ the options: papers to retrieve, bandit, fetch first
class FakeIngest:
    def __init__(self, monkeypatch, papers=3, error=None, empty_first=0):
        self.calls, self.papers, self.error, self.empty_first = [], papers, error, empty_first
        import src.main as m
        monkeypatch.setattr(m, "ingest_papers", self)

    def __call__(self, settings, query, **kw):
        self.calls.append((query, kw))
        if self.error:
            raise self.error
        return [] if len(self.calls) <= self.empty_first else [object()] * self.papers


def test_defaults_match_the_terminal_run_and_fetch_nothing(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch)
    submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert fake.kwargs[0]["k"] == 5 and fake.kwargs[0]["adaptive"] is False and ingest.calls == []


def test_the_number_of_papers_you_choose_reaches_the_pipeline(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    at = open_app()
    at.number_input(key="opt_k").set_value(3)
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert not at.exception and fake.kwargs[0]["k"] == 3 and fake.kwargs[0]["adaptive"] is False


def test_the_bandit_option_is_passed_on_and_its_choice_is_reported(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    at = open_app()
    at.checkbox(key="opt_adaptive").check()
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert not at.exception and fake.kwargs[0]["adaptive"] is True and len(fake.calls) == 1
    assert any("bandit chose to retrieve 2 paper(s)" in i.value and "1.00" in i.value for i in at.info)
    assert any("Adaptive strategy" in line for line in at.session_state["last_job"]["lines"])


def test_fetching_first_uses_the_topic_as_the_query_and_then_runs(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch, papers=4)
    at = open_app()
    at.checkbox(key="opt_fetch").check()
    at.number_input(key="opt_fetch_n").set_value(15)
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert not at.exception
    assert ingest.calls == [("Diabetes Prediction Using Machine Learning", {"source": "arxiv", "max_results": 15, "category": "cs.LG"})]
    assert len(fake.calls) == 1                                                   # ingest, then exactly one run
    assert any("Fetched, stored and indexed 4 paper(s)" in line for line in at.session_state["last_job"]["lines"])
    lines = at.session_state["last_job"]["lines"]
    assert len(lines) == 7 and "Fetched, stored and indexed 4 paper(s)" in lines[0]   # the fetch line comes FIRST, then the six steps


def test_fetch_keywords_and_pubmed_are_respected(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch)
    at = open_app()
    at.checkbox(key="opt_fetch").check()
    at.selectbox(key="opt_source").select("pubmed")
    at.text_input(key="opt_keywords").set_value("  diabetes   prediction  ")
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert ingest.calls == [("diabetes prediction", {"source": "pubmed", "max_results": 30, "category": None})]   # category is arXiv only


def test_a_failed_fetch_stops_before_any_research_and_says_so(two_runs, monkeypatch):
    fake = FakePipeline(two_runs, monkeypatch)
    FakeIngest(monkeypatch, error=ConnectionError("Connection refused"))
    at = open_app()
    at.checkbox(key="opt_fetch").check()
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert not at.exception and fake.calls == [] and len(run_files(two_runs)) == 2
    assert any("Fetch papers" in e.value for e in at.error) and not button(at, "Start Research").disabled


def test_when_nothing_is_found_the_run_continues_with_the_existing_index(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch, papers=0)
    at = open_app()
    at.checkbox(key="opt_fetch").check()
    submit(at, "Diabetes Prediction Using Machine Learning")
    assert [c[0] for c in ingest.calls] == ["Diabetes Prediction Using Machine Learning", "Diabetes Prediction Machine Learning", "Diabetes Prediction"]
    assert len(fake.calls) == 1
    assert any("No papers were returned" in line for line in at.session_state["last_job"]["lines"])


def test_choosing_options_never_starts_anything_by_itself(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch)
    at = open_app()
    at.number_input(key="opt_k").set_value(7).run()
    at.checkbox(key="opt_fetch").check().run()
    at.checkbox(key="opt_adaptive").check().run()
    assert fake.calls == [] and ingest.calls == []


def test_the_ingest_command_and_the_dashboard_share_one_fetch_function(tmp_path, monkeypatch, capsys):
    import src.main as m
    seen = []
    from src.models import Paper
    paper = Paper("arxiv", "1", "A title", "An abstract about trees.", url="u")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setattr(m, "_make_source", lambda settings, source: object())
    monkeypatch.setattr(m, "_search_source", lambda client, source, query, n, category: seen.append((query, n, category)) or [paper])
    assert m.main(["ingest", "--query", "decision tree depth", "--max-results", "7", "--category", "cs.LG"]) == 0
    assert seen == [("decision tree depth", 7, "cs.LG")] and "Stored and indexed 1 paper(s)" in capsys.readouterr().out
    assert m.main(["ingest", "--source", "pubmed", "--query", "x", "--category", "cs.LG"]) == 1
    with pytest.raises(ValueError):
        m.ingest_papers(m.load_settings(), "x", source="pubmed", category="cs.LG")


def test_the_real_bandit_learns_from_a_run_started_in_the_dashboard(real_env, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("src.main.make_sandbox", lambda s: SimpleNamespace(
        check_available=lambda: SimpleNamespace(ok=False, message="Docker is not running (test).")))
    at = open_app()
    at.checkbox(key="opt_adaptive").check()
    submit(at, "graph neural networks")
    assert not at.exception and len(run_files(real_env)) == 1
    bandit = json.loads((real_env / "bandit.json").read_text(encoding="utf-8"))
    assert "bandit chose to retrieve" in " ".join(i.value for i in at.info)
    assert "2" in json.dumps(bandit) or "3" in json.dumps(bandit) or "5" in json.dumps(bandit)    # the file now holds a learning event
    first = json.dumps(bandit, sort_keys=True)
    at.run()
    assert json.dumps(json.loads((real_env / "bandit.json").read_text(encoding="utf-8")), sort_keys=True) == first   # a rerun learns nothing new


def test_real_run_with_a_chosen_number_of_papers(real_env, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("src.main.make_sandbox", lambda s: None)
    at = open_app()
    at.number_input(key="opt_k").set_value(1)
    submit(at, "graph neural networks")
    record = json.loads((real_env / "runs" / run_files(real_env)[0]).read_text(encoding="utf-8"))
    assert not at.exception and len(record["evidence"]) == 1                              # exactly the one paper that was asked for


# ------------------------------------------------------------------ Evaluation and Learning (bandit): read-only sections
def write_evaluation(storage, rater="AI assistant; to be checked by the project owner", rated=True, name="holdout_20261110_112803"):
    folder = storage / "evaluation" / name
    folder.mkdir(parents=True)
    (folder / "summary.md").write_bytes("# Holdout evaluation\r\n\r\nHeld-out papers: 2.\r\n\r\n## Automatic checks\r\n\r\n- Runs finished: 2 of 2\r\n".encode("utf-8"))
    scores = ("3", "2", "4") if rated else ("", "", "")
    (folder / "rating_sheet.csv").write_bytes((
        "held_out_id,held_out_title,hypothesis_cites_retrieved_paper,execution,critic_verdict,similarity_to_own_abstract,own_abstract_rank,"
        "relevance_1to5,testability_1to5,groundedness_1to5,rater,comments\r\n"
        f"arxiv:1,Paper One,TRUE,SUCCESS,INCONCLUSIVE,0.5727,2,{scores[0]},{scores[1]},{scores[2]},{rater},first\r\n"
        f"arxiv:2,Paper Two,FALSE,SUCCESS,SIGNIFICANT,0.7,1,,,,,\r\n").encode("utf-8-sig"))
    return folder


def test_the_evaluation_section_shows_the_saved_results_and_who_rated(two_runs):
    write_evaluation(two_runs)
    at = go(open_app(), "Evaluation")
    assert not at.exception
    frame = at.dataframe[0].value
    assert list(frame["paper"]) == ["Paper One", "Paper Two"] and list(frame["verdict"]) == ["INCONCLUSIVE", "SIGNIFICANT"]
    assert list(frame["cites"]) == ["yes", "no"] and list(frame["rank"]) == [2, 1]
    assert frame["rated_by"][0].startswith("AI assistant") and frame["ratings"][1] == "not rated yet"
    assert any("Runs finished: 2 of 2" in m.value for m in at.markdown)
    assert any("Ratings filled in: 1 of 2" in c.value and "not a human rating" in c.value for c in at.caption)


def test_the_evaluation_section_without_any_evaluation_says_how_to_start_one(two_runs):
    at = go(open_app(), "Evaluation")
    assert not at.exception and any("python -m src.main evaluate" in i.value for i in at.info)


def test_a_damaged_evaluation_file_shows_a_message_not_a_crash(two_runs):
    folder = write_evaluation(two_runs)
    (folder / "rating_sheet.csv").write_bytes(b"\xff\xfe\x00 not a csv \x81")
    at = go(open_app(), "Evaluation")
    assert not at.exception and any("could not be read" in e.value for e in at.error)


def test_the_learning_section_shows_what_the_bandit_learned(two_runs):
    (two_runs / "bandit.json").write_text(json.dumps({"arms": {"2": {"n": 3, "reward": 2.5}, "3": {"n": 1, "reward": 1.0}, "5": {"n": 0, "reward": 0.0}}}), encoding="utf-8")
    at = go(open_app(), "Learning (bandit)")
    assert not at.exception
    assert at.dataframe[0].value.to_dict("records") == [{"papers": 2, "runs": 3, "average": "0.83"}, {"papers": 3, "runs": 1, "average": "1.00"},
                                                         {"papers": 5, "runs": 0, "average": "not tried yet"}]
    assert at.metric[0].value == "4" and any("not evidence that the strategy helps" in c.value for c in at.caption)
    assert (two_runs / "bandit.json").read_text(encoding="utf-8").count("2.5") == 1        # looking never changes what was learned


def test_the_learning_section_before_any_adaptive_run_and_with_a_damaged_file(two_runs):
    (two_runs / "bandit.json").unlink()                                      # the fixture's placeholder file is not a real bandit file
    at = go(open_app(), "Learning (bandit)")
    assert not at.exception and any("has not learned anything yet" in i.value for i in at.info)
    (two_runs / "bandit.json").write_text("{not json", encoding="utf-8")
    at = go(open_app(), "Learning (bandit)")
    assert not at.exception and any("could not be read" in e.value for e in at.error)


def test_the_new_sections_work_even_when_there_are_no_runs(env):
    write_evaluation(env)
    at = go(open_app(), "Evaluation")
    assert not at.exception and len(at.dataframe) == 1
    (env / "bandit.json").unlink()
    at = go(at, "Learning (bandit)")
    assert not at.exception and any("has not learned anything yet" in i.value for i in at.info)


def test_each_sidebar_item_is_its_own_page(two_runs):
    at = open_app()
    start_only = {"Start Research"}
    assert {b.label for b in at.button} >= start_only                                  # Start New Research has the form
    for section in ("Runs", "Logic tree", "Literature Citations", "Execution Performance & Costs", "Draft", "Evaluation", "Learning (bandit)"):
        go(at, section)
        assert not at.exception and not (start_only & {b.label for b in at.button}), section   # no form on the other pages
        assert not any(s.value == "Start New Research" for s in at.subheader), section
    go(at, "Start New Research")
    assert start_only <= {b.label for b in at.button}


def test_a_failed_run_stays_on_the_start_page_so_you_can_read_why(two_runs, monkeypatch):
    FakePipeline(two_runs, monkeypatch, fail_at="review")
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert at.session_state["section"] == "Start New Research" and any("model exploded" in e.value for e in at.error)


def test_a_finished_run_opens_its_logic_tree(two_runs, monkeypatch):
    FakePipeline(two_runs, monkeypatch)
    at = submit(open_app(), "Diabetes Prediction Using Machine Learning")
    assert at.session_state["section"] == "Logic tree" and len(at.expander) == 6


# ------------------------------------------------------------------ Evaluation and Learning pages are no longer read-only
FOLDER = "holdout_20261110_112803"
NEW_FOLDER = "holdout_20261201_101010"


def sheet_rows(storage, folder=FOLDER):
    import csv
    with (storage / "evaluation" / folder / "rating_sheet.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rate(at, i, rel, test, ground, folder=FOLDER):
    at.number_input(key=f"rate_{folder}_{i}_rel").set_value(rel)
    at.number_input(key=f"rate_{folder}_{i}_test").set_value(test)
    at.number_input(key=f"rate_{folder}_{i}_ground").set_value(ground)


def test_ratings_are_saved_with_the_name_of_the_person_who_rated(two_runs):
    write_evaluation(two_runs, rated=False)
    at = go(open_app(), "Evaluation")
    rate(at, 0, 4, 3, 5)
    at.text_input(key="rate_rater").set_value("Anu")
    button(at, "Save ratings").click().run()
    assert not at.exception and any("Saved the ratings of 1 paper(s)" in s.value for s in at.success)
    rows = sheet_rows(two_runs)
    assert [rows[0][c] for c in ("relevance_1to5", "testability_1to5", "groundedness_1to5", "rater")] == ["4", "3", "5", "Anu"]
    assert rows[1]["relevance_1to5"] == "" and rows[1]["rater"] == ""                        # the unrated paper is untouched
    assert rows[0]["held_out_title"] == "Paper One" and rows[0]["comments"] == "first"       # nothing else in the sheet changed
    assert (two_runs / "evaluation" / FOLDER / "rating_sheet.previous.csv").exists()
    assert list(at.dataframe[0].value["rated_by"])[0] == "Anu"


def test_ratings_without_a_rater_name_are_refused_and_change_nothing(two_runs):
    write_evaluation(two_runs, rated=False)
    before = (two_runs / "evaluation" / FOLDER / "rating_sheet.csv").read_bytes()
    at = go(open_app(), "Evaluation")
    rate(at, 0, 4, 3, 5)
    button(at, "Save ratings").click().run()
    assert any("Type your name as the rater" in e.value for e in at.error)
    assert (two_runs / "evaluation" / FOLDER / "rating_sheet.csv").read_bytes() == before
    assert not (two_runs / "evaluation" / FOLDER / "rating_sheet.previous.csv").exists()


def test_saving_without_changes_saves_nothing(two_runs):
    write_evaluation(two_runs)
    before = (two_runs / "evaluation" / FOLDER / "rating_sheet.csv").read_bytes()
    at = go(open_app(), "Evaluation")
    button(at, "Save ratings").click().run()
    assert any("Nothing changed" in i.value for i in at.info)
    assert (two_runs / "evaluation" / FOLDER / "rating_sheet.csv").read_bytes() == before


class FakeEvaluation:
    """Stands in for src.main.pick_evaluation_papers and run_evaluation (the real ones need arXiv, Ollama and Docker)."""
    def __init__(self, storage, monkeypatch, pick_error=None, run_error=None, found=2):
        self.storage, self.pick_error, self.run_error, self.found = storage, pick_error, run_error, found
        self.picks, self.runs = [], []
        import src.main as m
        monkeypatch.setattr(m, "pick_evaluation_papers", self.pick)
        monkeypatch.setattr(m, "run_evaluation", self.run)
        monkeypatch.setattr(m, "llm_problem", lambda settings: None)

    def pick(self, settings, **kw):
        from src.models import Paper
        self.picks.append(kw)
        if self.pick_error:
            raise self.pick_error
        return [Paper("arxiv", str(i), f"Recent paper {i}", "abstract", url="u", published="2026-11-02T00:00:00") for i in range(1, self.found + 1)], "2026-11-01"

    def run(self, settings, chosen, cutoff, **kw):
        self.runs.append((len(chosen), kw))
        for i, p in enumerate(chosen, 1):
            kw["on_paper"](i, len(chosen), p.title)
            kw["on_progress"](ProgressEvent("review", "running"))
        if self.run_error:
            raise self.run_error
        folder = write_evaluation(self.storage, name=NEW_FOLDER, rated=False)
        return folder, {"summary": {"runs_finished": len(chosen), "papers": len(chosen)}}


def test_preview_lists_the_papers_and_runs_nothing(two_runs, monkeypatch):
    fake = FakeEvaluation(two_runs, monkeypatch)
    at = go(open_app(), "Evaluation")
    button(at, "Preview papers").click().run()
    assert not at.exception and fake.runs == [] and fake.picks == [{"category": "cs.LG", "n": 5, "pool": 30}]
    assert [r["title"] for r in at.session_state["eval_preview"]["papers"]] == ["Recent paper 1", "Recent paper 2"]
    assert any("Nothing was run" in c.value for c in at.caption)


def test_run_evaluation_passes_your_choices_and_opens_the_new_results(two_runs, monkeypatch):
    fake = FakeEvaluation(two_runs, monkeypatch)
    at = go(open_app(), "Evaluation")
    at.text_input(key="ev_category").set_value("cs.CV")
    at.number_input(key="ev_n").set_value(2)
    at.number_input(key="ev_k").set_value(4)
    button(at, "Run evaluation").click().run(timeout=60)
    assert not at.exception and fake.picks == [{"category": "cs.CV", "n": 2, "pool": 30}] and len(fake.runs) == 1
    assert fake.runs[0][0] == 2 and fake.runs[0][1]["k"] == 4
    assert at.session_state["section"] == "Evaluation" and at.selectbox(key="eval_folder").value == NEW_FOLDER
    assert any("Evaluation finished: 2 of 2" in s.value for s in at.success)
    assert not button(at, "Run evaluation").disabled
    at.run()
    assert len(fake.runs) == 1                                                                # a rerun never starts it again


def test_an_empty_index_or_no_internet_stops_the_evaluation_with_a_reason(two_runs, monkeypatch):
    fake = FakeEvaluation(two_runs, monkeypatch, pick_error=ValueError("The index is empty. Ingest papers first."))
    at = go(open_app(), "Evaluation")
    button(at, "Run evaluation").click().run()
    assert fake.runs == [] and any("The index is empty" in e.value for e in at.error) and not button(at, "Run evaluation").disabled
    fake.pick_error = ConnectionError("Connection refused")
    button(at, "Run evaluation").click().run()
    assert fake.runs == [] and any("Ollama" in e.value or "Choose held-out papers" in e.value for e in at.error)


def test_no_suitable_paper_and_an_unavailable_model_are_explained(two_runs, monkeypatch):
    fake = FakeEvaluation(two_runs, monkeypatch, found=0)
    at = go(open_app(), "Evaluation")
    button(at, "Run evaluation").click().run()
    assert fake.runs == [] and any("No suitable recent paper" in w.value for w in at.warning)
    fake.found = 2
    import src.main as m
    monkeypatch.setattr(m, "llm_problem", lambda settings: "Ollama is not running.")
    button(at, "Run evaluation").click().run()
    assert fake.runs == [] and any("Ollama is not running" in e.value for e in at.error)


def test_a_failed_evaluation_says_what_is_saved_and_what_is_not(two_runs, monkeypatch):
    FakeEvaluation(two_runs, monkeypatch, run_error=RuntimeError("model exploded"))
    at = go(open_app(), "Evaluation")
    button(at, "Run evaluation").click().run()
    assert not at.exception and any("model exploded" in e.value and "no evaluation folder was written" in e.value for e in at.error)
    assert not (two_runs / "evaluation" / NEW_FOLDER).exists() and not button(at, "Run evaluation").disabled


def test_a_cut_off_evaluation_is_reported_honestly(two_runs):
    at = open_app()
    at.session_state["running_eval"] = "run"
    at.run()
    assert any("The evaluation did not finish" in w.value and "no evaluation folder" in w.value for w in at.warning)


def test_reset_learning_asks_first_keeps_a_copy_and_can_be_cancelled(two_runs):
    data = json.dumps({"arms": {"2": {"n": 1, "reward": 1.0}, "3": {"n": 0, "reward": 0.0}, "5": {"n": 0, "reward": 0.0}}})
    (two_runs / "bandit.json").write_text(data, encoding="utf-8")
    at = go(open_app(), "Learning (bandit)")
    button(at, key="bandit_reset_ask").click().run()
    assert (two_runs / "bandit.json").exists() and any("Reset what the bandit has learned?" in w.value for w in at.warning)
    button(at, key="bandit_reset_no").click().run()
    assert (two_runs / "bandit.json").read_text(encoding="utf-8") == data
    button(at, key="bandit_reset_ask").click().run()
    button(at, key="bandit_reset_yes").click().run()
    assert not (two_runs / "bandit.json").exists() and (two_runs / "bandit.json.bak").read_text(encoding="utf-8") == data
    assert any("The bandit was reset" in s.value for s in at.success) and any("has not learned anything yet" in i.value for i in at.info)
    assert button(at, key="bandit_reset_ask").disabled                                      # nothing left to reset
    at.run()
    assert not (two_runs / "bandit.json").exists()


def test_run_a_research_with_the_bandit_opens_the_start_page_with_the_option_ticked(two_runs):
    at = go(open_app(), "Learning (bandit)")
    button(at, key="go_adaptive").click().run()
    assert at.session_state["section"] == "Start New Research" and at.checkbox(key="opt_adaptive").value is True


# ------------------------------------------------------------------ the menu, the Runs page and the run dropdown
def test_the_menu_highlights_the_open_page_and_groups_the_items(two_runs):
    at = open_app()
    kinds = {b.label: b.proto.type for b in at.sidebar.button}
    assert kinds["Start New Research"] == "primary" and set(kinds.values()) == {"primary", "tertiary"}
    assert [c.value for c in at.sidebar.caption] == ["Research", "Selected run", "Project"]
    go(at, "Evaluation")
    kinds = {b.label: b.proto.type for b in at.sidebar.button}
    assert kinds["Evaluation"] == "primary" and kinds["Start New Research"] == "tertiary"
    assert all(not b.disabled for b in at.sidebar.button)


def test_open_on_the_runs_page_shows_that_run(two_runs, table):
    at = go(open_app(), "Runs")
    table.pick(1)
    at.run()
    button(at, key="runs_open").click().run()
    assert not at.exception and at.session_state["section"] == "Logic tree"
    assert at.selectbox(key="selected_run").value == "run_20260101_000001.json" and any("first topic" in s.value for s in at.subheader)


def test_the_run_dropdown_changes_the_run_and_the_choice_survives_other_pages(two_runs):
    at = go(open_app(), "Logic tree")
    assert at.selectbox(key="selected_run").value == "run_20260102_000002.json"            # the newest run first
    at.selectbox(key="selected_run").select("run_20260101_000001.json").run()
    assert any("first topic" in s.value for s in at.subheader)
    for page in ("Literature Citations", "Runs", "Evaluation", "Learning (bandit)", "Start New Research", "Draft"):
        go(at, page)
    assert at.selectbox(key="selected_run").value == "run_20260101_000001.json"            # still the run you chose
    assert any("first topic" in s.value for s in at.subheader) and any("run_20260101_000001.json" in c.value for c in at.caption)


def test_the_real_model_filter_lives_on_the_runs_page_and_is_remembered(two_runs):
    mock = full(topic="mock topic")
    mock["llm_mode"] = "TEST MODE (mock model)"
    (two_runs / "runs" / "run_20260103_000003.json").write_text(json.dumps(mock), encoding="utf-8")
    at = go(open_app(), "Runs")
    assert at.checkbox(key="only_real_cb").value is True and "run_20260103_000003.json" not in list(at.dataframe[0].value["file"])
    assert any("1 run(s) made with the mock model are hidden" in c.value for c in at.caption)
    at.checkbox(key="only_real_cb").uncheck().run()
    assert "run_20260103_000003.json" in list(at.dataframe[0].value["file"])
    go(at, "Logic tree")
    go(at, "Runs")
    assert at.checkbox(key="only_real_cb").value is False                                  # remembered after leaving the page


def test_the_fetch_tries_the_main_words_when_the_whole_topic_finds_nothing(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch, papers=5, empty_first=1)
    at = open_app()
    at.checkbox(key="opt_fetch").check()
    submit(at, "decision tree depth 1 versus depth 5 accuracy on the wine dataset")
    assert [c[0] for c in ingest.calls] == ["decision tree depth 1 versus depth 5 accuracy on the wine dataset", "decision tree depth wine"]
    lines = at.session_state["last_job"]["lines"]
    assert len(fake.calls) == 1 and 'using "decision tree depth wine"' in lines[0] and "5 paper(s)" in lines[0]


def test_your_own_keywords_are_the_only_search_text_and_are_never_replaced(two_runs, monkeypatch):
    fake, ingest = FakePipeline(two_runs, monkeypatch), FakeIngest(monkeypatch, papers=0)
    at = open_app()
    at.checkbox(key="opt_fetch").check()
    at.text_input(key="opt_keywords").set_value("decision tree depth")
    submit(at, "decision tree depth 1 versus depth 5 accuracy on the wine dataset")
    assert [c[0] for c in ingest.calls] == ["decision tree depth"] and len(fake.calls) == 1


# ------------------------------------------------------------------ cleaner pages and the blue theme
def test_the_marked_explanations_are_gone_and_the_important_ones_stay(two_runs):
    at = open_app()
    assert not any("Runs the same pipeline" in c.value for c in at.caption)                 # under the Start Research button
    go(at, "Evaluation")
    assert not any("The agent is run on recent papers" in c.value for c in at.caption)      # the Evaluation intro
    assert any("not a human rating" in c.value for c in at.caption) or not at.dataframe          # the rater note stays when there are ratings
    go(at, "Execution Performance & Costs")
    assert not any("Current setup in .env" in c.value or "Token counts are not recorded" in c.value for c in at.caption)
    money = next(m for m in at.metric if m.label == "Money")
    assert money.value == "None" and "Free local model" in money.help and "Token counts are not recorded" in money.help
    go(at, "Draft")
    assert any("without a language model" in m.value for m in at.markdown)                     # kept on purpose
    go(at, "Literature Citations")
    assert any("Similarity is the search score" in c.value for c in at.caption)                # kept on purpose


def test_a_test_mode_run_still_says_its_money_is_not_real(two_runs):
    from src.dashboard.data import cost_summary
    assert cost_summary(full(llm_mode="TEST MODE (mock model)"))["money"] == "none (test mode)"


def test_the_blue_theme_is_set_for_light_and_dark():
    import tomllib
    config = tomllib.loads((Path(APP).parents[2] / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["theme"]["light"]["primaryColor"] == "#2f6fed" and config["theme"]["dark"]["primaryColor"] == "#2f6fed"
    r, g, b = (int("2f6fed"[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    assert 1.05 / (luminance + 0.05) >= 4.5                                                    # white text on the blue buttons is readable
