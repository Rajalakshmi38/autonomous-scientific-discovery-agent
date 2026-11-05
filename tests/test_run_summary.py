"""Tests for the run-record summary. The records below are HAND-MADE test data."""
import json

from src.main import main
from src.utils.run_summary import format_summary, load_runs, summarize


def record(llm="ollama:llama3.2:1b (local, free)", status="COMPLETED", exec_status="FAILED", attempts=2,
           final="FAILED", cited=None, verdict="NO_RESULT"):
    return {
        "llm_mode": llm, "status": status, "execution_status": exec_status,
        "hypothesis": {"cited_ids": cited or []},
        "execution": {"attempt_count": attempts, "final_status": final} if attempts else None,
        "critic": {"verdict": verdict} if verdict else None,
    }


def test_real_and_mock_runs_are_kept_apart():
    recs = [
        record(),                                                                   # real, failed
        record(exec_status="SUCCESS", final="SUCCESS", attempts=2, cited=["arxiv:1"], verdict="INCONCLUSIVE"),   # real, fixed by retry
        record(llm="TEST MODE (rule-based mock, no LLM reasoning)", exec_status="SUCCESS", final="SUCCESS",
               attempts=1, cited=["arxiv:1"], verdict="INCONCLUSIVE"),
        record(status="NO_EVIDENCE", exec_status="NOT_RUN", attempts=0, verdict=None),
    ]
    s = summarize(recs)
    real, mock = s["real model"], s["mock (test mode)"]
    assert (real["runs"], real["completed"], real["no_evidence"]) == (3, 2, 1)
    assert real["execution"] == {"FAILED": 1, "SUCCESS": 1}
    assert real["fixed_by_retry"] == 1 and real["hypothesis_cited_a_paper"] == 1 and real["average_attempts"] == 2.0
    assert mock["runs"] == 1 and mock["execution"] == {"SUCCESS": 1}


def test_old_records_without_new_fields_are_still_counted():
    old = {"llm_mode": "ollama:x", "status": "COMPLETED_WEEK1", "execution_status": "PENDING - sandbox not available"}
    s = summarize([old])["real model"]
    assert s["completed"] == 1 and s["critic"] == {"not run": 1} and s["average_attempts"] is None


def test_unreadable_files_are_skipped_and_reported(tmp_path):
    (tmp_path / "run_1.json").write_text(json.dumps(record()), encoding="utf-8")
    (tmp_path / "run_2.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "run_3.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")            # not a run record: ignored
    records, unreadable = load_runs(tmp_path)
    assert len(records) == 1 and unreadable == 2
    assert "2 record file(s) could not be read" in " ".join(format_summary(summarize(records), unreadable))


def test_missing_folder_gives_empty_result(tmp_path):
    assert load_runs(tmp_path / "nope") == ([], 0)


def test_summary_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    assert main(["summary"]) == 0
    assert "No run records yet" in capsys.readouterr().out
    runs = tmp_path / "s" / "runs"
    runs.mkdir(parents=True)
    (runs / "run_a.json").write_text(json.dumps(record()), encoding="utf-8")
    assert main(["summary"]) == 0
    out = capsys.readouterr().out
    assert "real model: 1 run(s)" in out and "FAILED" in out


def test_records_from_before_the_sandbox_are_shown_as_not_run():
    old = {"llm_mode": "ollama:x", "status": "COMPLETED", "execution_status": "PENDING - sandbox execution is not available until later"}
    assert summarize([old])["real model"]["execution"] == {"NOT_RUN": 1}
