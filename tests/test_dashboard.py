"""The dashboard's data layer and its Streamlit page. The records are TEST DATA written by hand in the real shape."""
import copy
import json
import sys
from pathlib import Path

import pytest

from src.dashboard.data import STEPS, citations, cost_summary, load_runs, overview_rows, trace_steps
from src.main import main
from tests.test_draft import RECORD

FULL = copy.deepcopy(RECORD)
FULL.update({
    "step_seconds": {"retrieve": 0.4, "review": 90.1, "hypothesis": 70.2, "experiment_design": 14.5, "execute": 21.9, "critic": 0.1},
    "review_grounding": {"cited_ids": ["arxiv:1111.11111"], "unknown_ids": [], "weakly_supported": ["A weak sentence."], "grounded": True},
    "hypothesis_grounding": {"cited_ids": ["arxiv:1111.11111"], "unknown_ids": ["arxiv:9999.99999"], "weakly_supported": [], "grounded": False},
})
FULL["execution"] = {"final_status": "SUCCESS", "message": "ok", "attempt_count": 1,
                     "attempts": [{"number": 1, "note": "", "result": {"status": "SUCCESS", "exit_code": 0, "duration_seconds": 21.9}}]}


def full(**changes):
    r = copy.deepcopy(FULL)
    r.update(changes)
    return r


def test_load_runs_is_newest_first_and_skips_broken_files(tmp_path):
    (tmp_path / "run_20260101_000001.json").write_text(json.dumps(full(topic="old")), encoding="utf-8")
    (tmp_path / "run_20260101_000002.json").write_text(json.dumps(full(topic="new")), encoding="utf-8")
    (tmp_path / "run_20260101_000003.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "run_20260101_000004.json").write_text(json.dumps({"no": "topic"}), encoding="utf-8")
    assert [e.record["topic"] for e in load_runs(tmp_path)] == ["new", "old"]
    assert load_runs(tmp_path / "missing") == []


def test_the_logic_tree_has_the_six_steps_in_order_with_honest_statuses():
    steps = trace_steps(full())
    assert [s.key for s in steps] == [k for k, _ in STEPS]
    by = {s.key: s for s in steps}
    assert by["retrieve"].status == "done" and by["retrieve"].seconds == 0.4
    assert by["review"].status == "warning" and any("A weak sentence." in line for line in by["review"].lines)
    assert by["hypothesis"].status == "warning" and any("arxiv:9999.99999" in line and "NOT retrieved" in line for line in by["hypothesis"].lines)
    assert by["execute"].status == "done" and by["critic"].status == "done"
    assert any("corrected t-test" in line for line in by["critic"].lines) or any("b minus a" in line for line in by["critic"].lines)


def test_a_run_without_evidence_stops_after_the_first_step():
    steps = trace_steps(full(evidence=[], status="NO_EVIDENCE"))
    assert steps[0].status == "failed" and [s.status for s in steps[1:]] == ["not run"] * 5


@pytest.mark.parametrize("final, expected", [("TIMEOUT", "failed"), ("FAILED", "failed"),
                                             ("SANDBOX_UNAVAILABLE", "not run"), ("NOT_RUN", "not run")])
def test_execution_status_maps_to_the_step_status(final, expected):
    r = full()
    r["execution"] = {"final_status": final, "attempt_count": 1, "attempts": []}
    assert {s.key: s for s in trace_steps(r)}["execute"].status == expected


def test_a_design_made_by_fallback_rules_is_flagged_and_a_no_result_critic_is_failed():
    r = full()
    r["experiment"]["source"] = "keyword_spec"
    r["critic"] = {"verdict": "NO_RESULT", "summary_lines": []}
    by = {s.key: s for s in trace_steps(r)}
    assert by["experiment_design"].status == "warning" and by["critic"].status == "failed"


def test_an_old_record_without_grounding_or_timing_still_works():
    old = copy.deepcopy(RECORD)
    steps = trace_steps(old)
    assert len(steps) == 6 and all(s.seconds is None for s in steps)
    assert cost_summary(old)["total_seconds"] == 0


def test_citations_show_where_each_paper_was_cited_and_list_unknown_ids():
    c = citations(full())
    first = c["rows"][0]
    assert first["id"] == "arxiv:1111.11111" and first["cited by review"] and first["cited by hypothesis"]
    assert not c["rows"][1]["cited by review"] and not c["rows"][1]["cited by hypothesis"]
    assert c["unknown"] == ["arxiv:9999.99999"]


def test_costs_are_time_and_attempts_not_money_or_tokens():
    cost = cost_summary(full())
    assert cost["total_seconds"] == 197.2 and cost["sandbox_seconds"] == 21.9 and cost["attempts"] == 1
    assert "free local model" in cost["money"] and "Token counts are not recorded" in cost["note"]
    assert "test mode" in cost_summary(full(llm_mode="TEST MODE (mock model)"))["money"]


def test_overview_rows_mark_test_mode_runs(tmp_path):
    (tmp_path / "run_20260101_000001.json").write_text(json.dumps(full(llm_mode="TEST MODE (mock model)")), encoding="utf-8")
    row = overview_rows(load_runs(tmp_path))[0]
    assert row["mode"] == "TEST MODE" and row["file"] == "run_20260101_000001.json" and row["verdict"] == "INCONCLUSIVE"


def test_dashboard_command_explains_a_missing_streamlit(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "streamlit", None)
    assert main(["dashboard"]) == 1 and "requirements-dashboard.txt" in capsys.readouterr().out


def test_dashboard_command_starts_streamlit_with_the_app_file(monkeypatch):
    pytest.importorskip("streamlit")
    seen = {}
    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: seen.update(cmd=cmd) or 0)
    assert main(["dashboard"]) == 0
    assert seen["cmd"][1:4] == ["-m", "streamlit", "run"] and seen["cmd"][-1].endswith("app.py")


# ------------------------------------------------------------------ the Streamlit page itself
APP = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    runs = tmp_path / "s" / "runs"
    runs.mkdir(parents=True)
    return runs


def test_page_shows_a_hint_when_there_are_no_runs(app_env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("No run records yet" in i.value for i in at.info)


def test_page_shows_the_steps_and_writes_a_draft(app_env):
    from streamlit.testing.v1 import AppTest
    (app_env / "run_20260101_000001.json").write_text(json.dumps(full()), encoding="utf-8")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert len(labels) == 6 and any(l.startswith("3. Hypothesis: WARNING") for l in labels)
    assert any("arxiv:9999.99999" in e.value for e in at.error)          # cited but never retrieved
    at.button[0].click().run()
    assert not at.exception and any("written" in s.value for s in at.success)


def test_page_labels_a_test_mode_run(app_env):
    from streamlit.testing.v1 import AppTest
    (app_env / "run_20260101_000001.json").write_text(json.dumps(full(llm_mode="TEST MODE (mock model)")), encoding="utf-8")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("TEST MODE" in w.value for w in at.warning)
