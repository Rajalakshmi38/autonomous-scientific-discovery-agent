"""Summarise the saved run records (storage/runs/run_*.json) into plain counts.

Nothing here is estimated: every number is a count of records on this machine. Runs with the mock model
(TEST MODE) are kept apart from runs with a real model, so a good-looking mock result can never be
mistaken for evidence about the real model. Older records that lack newer fields are still counted.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

GROUPS = ("real model", "mock (test mode)")


def load_runs(runs_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Return (records, number of unreadable files)."""
    records: List[Dict[str, Any]] = []
    unreadable = 0
    if not Path(runs_dir).is_dir():
        return records, unreadable
    for path in sorted(Path(runs_dir).glob("run_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        if isinstance(data, dict):
            records.append(data)
        else:
            unreadable += 1
    return records, unreadable


def _group_of(record: Dict[str, Any]) -> str:
    return GROUPS[1] if "TEST MODE" in str(record.get("llm_mode", "")) else GROUPS[0]


def _execution_label(record: Dict[str, Any]) -> str:
    """Records from before the sandbox existed carry a long 'pending' sentence: show them as NOT_RUN."""
    status = str(record.get("execution_status", "NOT_RUN"))
    return "NOT_RUN" if status.startswith("PENDING") else status


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for group in GROUPS:
        runs = [r for r in records if _group_of(r) == group]
        completed = [r for r in runs if str(r.get("status", "")).startswith("COMPLETED")]
        cited = [r for r in completed if (r.get("hypothesis") or {}).get("cited_ids")]
        executed = [r for r in completed if (r.get("execution") or {}).get("attempt_count")]
        attempts = [r["execution"]["attempt_count"] for r in executed]
        repaired = [r for r in executed if r["execution"]["attempt_count"] > 1 and r["execution"].get("final_status") == "SUCCESS"]
        out[group] = {
            "runs": len(runs),
            "completed": len(completed),
            "no_evidence": sum(1 for r in runs if r.get("status") == "NO_EVIDENCE"),
            "hypothesis_cited_a_paper": len(cited),
            "execution": dict(Counter(_execution_label(r) for r in completed)),
            "critic": dict(Counter(str((r.get("critic") or {}).get("verdict", "not run")) for r in completed)),
            "fixed_by_retry": len(repaired),
            "average_attempts": round(sum(attempts) / len(attempts), 2) if attempts else None,
        }
    return out


def format_summary(summary: Dict[str, Dict[str, Any]], unreadable: int = 0) -> List[str]:
    lines: List[str] = []
    for group, s in summary.items():
        lines.append(f"{group}: {s['runs']} run(s)")
        if not s["runs"]:
            continue
        lines.append(f"  completed the pipeline : {s['completed']}  (no evidence found: {s['no_evidence']})")
        lines.append(f"  hypothesis cited a retrieved paper : {s['hypothesis_cited_a_paper']} of {s['completed']}")
        lines.append(f"  experiment results     : {s['execution'] or 'none'}")
        lines.append(f"  critic verdicts        : {s['critic'] or 'none'}")
        lines.append(f"  fixed by the retry loop: {s['fixed_by_retry']}   average attempts: {s['average_attempts']}")
    if unreadable:
        lines.append(f"({unreadable} record file(s) could not be read and were skipped.)")
    return lines
