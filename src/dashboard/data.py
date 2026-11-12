"""Everything the dashboard shows, computed from the saved run records. No UI code and no language model here.

Costs: the project uses free local models, so there is no money cost. What a record holds is wall-clock time per
pipeline step, the sandbox time of each attempt and the number of attempts. Token counts are NOT recorded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STEPS = [("retrieve", "Retrieve papers"), ("review", "Literature review"), ("hypothesis", "Hypothesis"),
         ("experiment_design", "Experiment design"), ("execute", "Sandbox execution"), ("critic", "Statistical critic")]


@dataclass
class RunEntry:
    path: Path
    record: Dict[str, Any]

    @property
    def is_test_mode(self) -> bool:
        return "TEST MODE" in str(self.record.get("llm_mode", ""))

    @property
    def label(self) -> str:
        r = self.record
        verdict = (r.get("critic") or {}).get("verdict") or "no verdict"
        mode = "TEST MODE" if self.is_test_mode else "real model"
        return f"{str(r.get('created_at', self.path.stem)).replace('T', ' ')} | {r.get('topic')} | {verdict} | {mode}"


@dataclass
class Step:
    key: str
    title: str
    status: str                                   # done | warning | failed | not run
    seconds: Optional[float]
    lines: List[str] = field(default_factory=list)


def load_runs(runs_dir: Path) -> List[RunEntry]:
    """All readable run records, newest first. Broken files are skipped, not fatal."""
    out: List[RunEntry] = []
    root = Path(runs_dir)
    for path in sorted(root.glob("run_*.json"), reverse=True) if root.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "topic" in data:
            out.append(RunEntry(path, data))
    return out


def _grounding(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    return record.get(key) or {}


def _short(text: Any, limit: int = 300) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def trace_steps(record: Dict[str, Any]) -> List[Step]:
    """The agent's logic tree: the six pipeline steps in order, each with what it did and how it ended."""
    secs = record.get("step_seconds") or {}
    evidence = record.get("evidence") or []
    review_g, hyp_g = _grounding(record, "review_grounding"), _grounding(record, "hypothesis_grounding")
    hyp, exp = record.get("hypothesis") or {}, record.get("experiment") or {}
    execu, critic = record.get("execution") or {}, record.get("critic") or {}
    steps: List[Step] = []

    steps.append(Step("retrieve", "Retrieve papers", "done" if evidence else "failed", secs.get("retrieve"),
                      [f"{len(evidence)} paper(s) retrieved."] +
                      [f"{e.get('id')}: {_short(e.get('title'), 90)} (similarity {e.get('score')})" for e in evidence]
                      if evidence else ["No relevant papers were found, so the run stopped. Ingest papers first."]))
    if not evidence:
        return steps + [Step(k, t, "not run", None) for k, t in STEPS[1:]]

    r_lines = [f"Papers cited: {', '.join(review_g.get('cited_ids') or []) or 'none'}"]
    r_status = "done"
    if review_g.get("unknown_ids"):
        r_lines.append("Cited but NOT retrieved: " + ", ".join(review_g["unknown_ids"]))
        r_status = "warning"
    for s in review_g.get("weakly_supported") or []:
        r_lines.append("Weak lexical support: " + _short(s))
        r_status = "warning"
    steps.append(Step("review", "Literature review", r_status, secs.get("review"), r_lines))

    if not hyp:
        steps.append(Step("hypothesis", "Hypothesis", "failed", secs.get("hypothesis"), ["No hypothesis was produced."]))
    else:
        h_lines = [str(hyp.get("statement")), f"Papers cited: {', '.join(hyp_g.get('cited_ids') or []) or 'none'}"]
        h_status = "done" if hyp_g.get("grounded", True) else "warning"
        if hyp_g.get("unknown_ids"):
            h_lines.append("Cited but NOT retrieved: " + ", ".join(hyp_g["unknown_ids"]))
            h_status = "warning"
        for s in hyp_g.get("weakly_supported") or []:
            h_lines.append("Weak lexical support: " + _short(s))
            h_status = "warning"
        if not hyp_g.get("cited_ids"):
            h_lines.append("The hypothesis cites no retrieved paper.")
            h_status = "warning"
        steps.append(Step("hypothesis", "Hypothesis", h_status, secs.get("hypothesis"), h_lines))

    if not exp:
        steps.append(Step("experiment_design", "Experiment design", "not run", secs.get("experiment_design")))
    else:
        e_lines = [str(exp.get("design_summary")), f"How the design was made: {exp.get('source')}"]
        e_status = "warning" if exp.get("source") in ("keyword_spec", "default_spec") or exp.get("note") else "done"
        if exp.get("note"):
            e_lines.append(_short(exp["note"], 500))
        steps.append(Step("experiment_design", "Experiment design", e_status, secs.get("experiment_design"), e_lines))

    if not execu:
        steps.append(Step("execute", "Sandbox execution", "not run", secs.get("execute")))
    else:
        final = execu.get("final_status")
        x_lines = [f"Final status: {final}; attempts: {execu.get('attempt_count')}."]
        for a in execu.get("attempts") or []:
            res = a.get("result") or {}
            x_lines.append(f"Attempt {a.get('number')}: {res.get('status')} in {res.get('duration_seconds')} s"
                           + (f" ({a.get('note')})" if a.get("note") else ""))
        steps.append(Step("execute", "Sandbox execution", "done" if final == "SUCCESS" else
                          ("not run" if final in ("NOT_RUN", "SANDBOX_UNAVAILABLE") else "failed"),
                          secs.get("execute"), x_lines))

    if not critic:
        steps.append(Step("critic", "Statistical critic", "not run", secs.get("critic")))
    else:
        verdict = critic.get("verdict")
        steps.append(Step("critic", "Statistical critic", "failed" if verdict == "NO_RESULT" else "done",
                          secs.get("critic"), [f"Verdict: {verdict} (data origin: {critic.get('data_origin')})"]
                          + list(critic.get("summary_lines") or [])))
    return steps


def citations(record: Dict[str, Any]) -> Dict[str, Any]:
    """Every retrieved paper with where it was cited, plus ids that were cited but never retrieved."""
    by_review = set(_grounding(record, "review_grounding").get("cited_ids") or [])
    by_hyp = set(_grounding(record, "hypothesis_grounding").get("cited_ids") or [])
    rows = [{"id": e.get("id"), "title": e.get("title"), "similarity": e.get("score"),
             "status": e.get("status", "not recorded"),
             "cited by review": e.get("id") in by_review, "cited by hypothesis": e.get("id") in by_hyp,
             "link": e.get("url")} for e in record.get("evidence") or []]
    unknown = sorted(set(_grounding(record, "review_grounding").get("unknown_ids") or [])
                     | set(_grounding(record, "hypothesis_grounding").get("unknown_ids") or []))
    return {"rows": rows, "unknown": unknown}


def cost_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    secs = {k: float(v) for k, v in (record.get("step_seconds") or {}).items() if isinstance(v, (int, float))}
    attempts = (record.get("execution") or {}).get("attempts") or []
    sandbox = round(sum(float((a.get("result") or {}).get("duration_seconds") or 0) for a in attempts), 2)
    test_mode = "TEST MODE" in str(record.get("llm_mode", ""))
    return {"step_seconds": secs, "total_seconds": round(sum(secs.values()), 1), "sandbox_seconds": sandbox,
            "attempts": len(attempts), "model": record.get("llm_mode"),
            "money": "none (test mode)" if test_mode else "none (free local model)",
            "note": "Token counts are not recorded; only time and attempts."}


def overview_rows(entries: List[RunEntry]) -> List[Dict[str, Any]]:
    rows = []
    for e in entries:
        r = e.record
        rows.append({"time": str(r.get("created_at", "")).replace("T", " "), "topic": r.get("topic"),
                     "mode": "TEST MODE" if e.is_test_mode else "real model", "execution": r.get("execution_status"),
                     "verdict": (r.get("critic") or {}).get("verdict") or "-",
                     "seconds": cost_summary(r)["total_seconds"], "file": e.path.name})
    return rows
