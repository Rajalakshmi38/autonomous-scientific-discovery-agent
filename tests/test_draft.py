"""The draft generator. The records here are TEST DATA written by hand in the shape of real run records."""
import copy
import json

import pytest

from src.draft.generator import (build_draft, draft_kind, latest_record_path, load_record, tex_escape, to_latex,
                                 to_markdown, write_draft)
from src.main import main

LINE = "iris (accuracy): b minus a = -0.00667 (95% CI -0.0398 to 0.0265); corrected t-test p = 0.679; dz = -0.29"
RECORD = {
    "topic": "random forest number of trees", "created_at": "2026-01-01T10:00:00",
    "llm_mode": "ollama:llama3.2:3b (local, free)", "status": "COMPLETED", "execution_status": "SUCCESS",
    "evidence": [{"id": "arxiv:1111.11111", "title": "How many trees? 50% & more_x", "url": "https://arxiv.org/abs/1111.11111v1", "score": 0.7},
                 {"id": "arxiv:2222.22222", "title": "Other paper", "url": "https://arxiv.org/abs/2222.22222v1", "score": 0.6}],
    "sources_cited_by_hypothesis": [{"id": "arxiv:1111.11111", "title": "t", "url": "u"}],
    "review": {"summary": "Variance of order epsilon \u03b5^(-2). [arxiv:1111.11111]"},
    "hypothesis": {"statement": "A random forest with 50 trees scores higher than one with 5 trees on iris.",
                   "rationale": "More trees lower variance.", "test_plan": "Compare both on iris."},
    "experiment": {"source": "model_spec", "design_summary": "iris: random_forest(n_estimators=50) vs random_forest(n_estimators=5), accuracy, 20 resamples",
                   "spec": {"dataset": "iris", "metric": "accuracy", "n_resamples": 20, "test_size": 0.3, "seed": 0,
                            "baseline": {"model": "random_forest", "params": {"n_estimators": 5}, "scale": False},
                            "treatment": {"model": "random_forest", "params": {"n_estimators": 50}, "scale": False}},
                   "note": ""},
    "execution": {"final_status": "SUCCESS", "message": "The script finished normally.", "attempt_count": 1, "attempts": []},
    "critic": {"verdict": "INCONCLUSIVE", "alpha": 0.05, "data_origin": "benchmark",
               "comparisons": [{"name": "iris (accuracy)", "labels": ["a", "b"],
                                "result": {"mean_a": 0.9522, "mean_b": 0.9456, "n_a": 20, "n_b": 20}}],
               "findings": [{"severity": "warning", "code": "benchmark", "message": "The data are a small classic benchmark dataset."}],
               "summary_lines": [LINE]},
    "warnings": ["Experiment design: the keyword rules were used."],
}


def rec(**changes):
    r = copy.deepcopy(RECORD)
    r.update(changes)
    return r


def test_a_successful_run_gives_a_paper_with_the_critic_lines_copied_unchanged():
    d = build_draft(rec())
    md = to_markdown(d)
    assert d.kind == "paper" and md.startswith("# Draft report: random forest number of trees")
    assert LINE in md                                                    # copied word for word
    assert [h for h, _ in d.sections] == ["Introduction", "Method", "Results", "Limitations", "References"]
    assert "A random forest with 50 trees scores higher" in md and "random_forest(n_estimators=50)" in md
    assert "has not been peer reviewed" in md and "not a verified research finding" in md
    assert "(cited by the hypothesis)" in md and "https://arxiv.org/abs/1111.11111v1" in md
    assert "does not show that there is no effect" in md                  # INCONCLUSIVE wording
    assert "small classic benchmark dataset" in md                        # critic finding carried into Limitations


def test_a_significant_run_is_not_called_proof():
    r = rec()
    r["critic"]["verdict"] = "SIGNIFICANT"
    md = to_markdown(build_draft(r))
    assert "does not make the hypothesis true" in md


@pytest.mark.parametrize("changes, text", [
    ({"execution_status": "TIMEOUT"}, "SANDBOX_TIMEOUT_SECONDS"),
    ({"execution_status": "SANDBOX_UNAVAILABLE"}, "Docker"),
    ({"execution_status": "FAILED"}, "error"),
])
def test_a_run_without_a_verified_result_gives_a_failed_run_report_not_a_paper(changes, text):
    r = rec(**changes)
    r["critic"] = {"verdict": "NO_RESULT", "alpha": 0.05, "findings": [{"severity": "error", "code": "no_result", "message": "There are no results to check."}], "summary_lines": []}
    r["execution"] = {"final_status": changes["execution_status"], "message": "It did not work.", "attempt_count": 1,
                      "attempts": [{"number": 1, "note": "", "result": {"status": changes["execution_status"], "exit_code": 1, "duration_seconds": 3.0, "stderr": "Traceback ... boom"}}]}
    d = build_draft(r)
    md = to_markdown(d)
    assert d.kind == "failed_run" and md.startswith("# Failed run report")
    assert "No research paper was written" in md and text in md and "Traceback ... boom" in md
    assert "## Abstract" not in md and "## Results" not in md and "p = 0.679" not in md


def test_a_run_with_no_evidence_gives_a_failed_run_report():
    r = rec(status="NO_EVIDENCE", execution_status="NOT_RUN", evidence=[], hypothesis=None, experiment=None, critic=None, execution=None)
    d = build_draft(r)
    assert d.kind == "failed_run" and "Ingest papers" in to_markdown(d)


def test_a_test_mode_record_is_labelled_at_the_top():
    md = to_markdown(build_draft(rec(llm_mode="TEST MODE (mock model)")))
    assert md.index("TEST MODE") < md.index("## Abstract")


def test_design_fallback_and_run_warnings_appear_in_the_limitations():
    r = rec()
    r["experiment"]["source"] = "keyword_spec"
    md = to_markdown(build_draft(r))
    assert "fixed keyword rules chose the design" in md and "the keyword rules were used" in md


def test_draft_kind_needs_success_a_verdict_and_a_hypothesis():
    assert draft_kind(rec()) == "paper"
    assert draft_kind(rec(hypothesis=None)) == "failed_run"
    r = rec()
    r["critic"]["verdict"] = "NOT_EVALUABLE"
    assert draft_kind(r) == "failed_run"


def test_latex_escapes_special_characters_and_has_a_complete_skeleton():
    tex = to_latex(build_draft(rec()))
    assert tex_escape("50% & a_b #1 $5 {x}") == r"50\% \& a\_b \#1 \$5 \{x\}"
    assert r"\begin{abstract}" in tex and r"\end{document}" in tex
    assert tex.count(r"\begin{") == tex.count(r"\end{")
    assert "\u03b5" not in tex and "epsilon" in tex                       # non-ASCII is replaced, so pdflatex can compile it
    assert r"\url{https://arxiv.org/abs/1111.11111v1}" in tex
    assert "50\\% \\& more\\_x" in tex


def test_latex_of_a_failed_run_report_is_also_complete():
    r = rec(execution_status="TIMEOUT", critic=None, execution=None)
    tex = to_latex(build_draft(r))
    assert r"\section*{Summary}" in tex and tex.count(r"\begin{") == tex.count(r"\end{")


def test_latest_record_prefers_a_real_model_run(tmp_path):
    (tmp_path / "run_20260101_000001.json").write_text(json.dumps(rec()), encoding="utf-8")
    (tmp_path / "run_20260101_000002.json").write_text(json.dumps(rec(llm_mode="TEST MODE (mock model)")), encoding="utf-8")
    assert latest_record_path(tmp_path).name == "run_20260101_000001.json"
    assert latest_record_path(tmp_path / "missing") is None


def test_write_draft_writes_the_chosen_formats_and_rejects_bad_input(tmp_path):
    src = tmp_path / "run_20260101_000001.json"
    src.write_text(json.dumps(rec()), encoding="utf-8")
    kind, files = write_draft(src, tmp_path / "out", "md")
    assert kind == "paper" and [f.suffix for f in files] == [".md"] and files[0].read_text(encoding="utf-8").startswith("# Draft")
    assert [f.suffix for f in write_draft(src, tmp_path / "out", "both")[1]] == [".md", ".tex"]
    with pytest.raises(ValueError):
        write_draft(src, tmp_path / "out", "pdf")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_record(bad)


def test_cli_draft_writes_files_and_says_which_kind(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    assert main(["draft"]) == 1 and "No run record" in capsys.readouterr().out
    runs = tmp_path / "s" / "runs"
    runs.mkdir(parents=True)
    (runs / "run_20260101_000001.json").write_text(json.dumps(rec()), encoding="utf-8")
    assert main(["draft"]) == 0
    out = capsys.readouterr().out
    assert "unreviewed" in out and (tmp_path / "s" / "drafts" / "run_20260101_000001_draft.md").exists()
    (runs / "run_20260101_000002.json").write_text(json.dumps(rec(execution_status="TIMEOUT", critic=None)), encoding="utf-8")
    assert main(["draft", "--run", str(runs / "run_20260101_000002.json"), "--format", "md"]) == 0
    assert "FAILED RUN REPORT" in capsys.readouterr().out


def test_the_model_summary_keeps_its_paragraphs_and_bullets():
    r = rec()
    r["review"]["summary"] = "**Evidence Summary**\n\nThe papers discuss trees.\n\n* [arxiv:1111.11111] first point\n* [arxiv:2222.22222] second point"
    md = to_markdown(build_draft(r))
    assert "- [arxiv:1111.11111] first point" in md and "- [arxiv:2222.22222] second point" in md
    assert "**" not in md and "Evidence Summary" in md
    assert r"\item [arxiv:1111.11111] first point" in to_latex(build_draft(r))


def test_limitations_do_not_repeat_the_same_warning():
    r = rec()
    r["warnings"] = ["Critic: The data are a small classic benchmark dataset.", "Experiment design: something new happened."]
    md = to_markdown(build_draft(r))
    assert md.count("small classic benchmark dataset") == 1               # also a critic finding, so not repeated
    assert "Run warning: Experiment design: something new happened." in md  # a new warning is kept


def _logistic_design(r):
    r["experiment"]["spec"]["treatment"] = {"model": "logistic_regression", "params": {}, "scale": False}
    r["experiment"]["spec"]["baseline"] = {"model": "random_forest", "params": {}, "scale": False}
    return r


def test_an_old_record_gets_the_mismatch_warning_when_the_draft_is_made():
    md = to_markdown(build_draft(_logistic_design(rec())))
    assert "Possible mismatch" in md and "uses logistic_regression" in md


def test_the_failed_run_report_also_shows_the_mismatch():
    r = _logistic_design(rec(execution_status="TIMEOUT", critic=None, execution=None))
    md = to_markdown(build_draft(r))
    assert md.startswith("# Failed run report") and "Possible mismatch" in md


def test_a_matching_design_adds_no_mismatch_line_and_a_stored_one_is_not_repeated():
    assert "Possible mismatch" not in to_markdown(build_draft(rec()))
    r = _logistic_design(rec())
    r["experiment"]["note"] = "Possible mismatch with the hypothesis: the experiment uses logistic_regression, which the hypothesis does not mention."
    assert to_markdown(build_draft(r)).count("Possible mismatch") == 1
