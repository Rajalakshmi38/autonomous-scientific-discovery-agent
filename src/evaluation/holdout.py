"""Holdout evaluation.

The idea: take recent papers that are NOT in the index and are newer than everything in it, run the normal pipeline
on each paper's title, and compare what the agent produced with the paper it never saw.

What is measured automatically (all of it is a rough proxy, not a quality score):
  * did the run finish, did the hypothesis cite a retrieved paper, were any cited ids never retrieved,
  * did the sandbox succeed and what did the critic say,
  * embedding similarity between the generated hypothesis and the held-out abstract, next to its similarity to the
    OTHER held-out abstracts (a hypothesis that points at its own paper should score higher there than elsewhere).
What cannot be measured automatically (relevance, testability, groundedness) goes to a CSV sheet for a human to score.
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.models import Paper
from src.retrieval.embeddings import EmbeddingError

logger = logging.getLogger("evaluation")

RATING_COLUMNS = ["relevance_1to5", "testability_1to5", "groundedness_1to5"]
SHEET_COLUMNS = ["held_out_id", "held_out_title", "held_out_url", "run_file", "topic", "hypothesis",
                 "hypothesis_cites_retrieved_paper", "invalid_citations", "execution", "critic_verdict", "design",
                 "design_source", "design_mismatch", "similarity_to_own_abstract",
                 "mean_similarity_to_other_abstracts", "own_abstract_rank", "error"] + RATING_COLUMNS + ["rater", "comments"]
LIMITS = [
    "The sample is tiny (a handful of papers), so none of these numbers can show that the agent is good or bad in general.",
    "Embedding similarity is only a rough proxy. A high value does not mean a good hypothesis, and a low value does not "
    "mean a bad one.",
    "Execution and critic results come from small experiments on toy datasets, not from the held-out papers' own methods.",
    "Relevance, testability and groundedness need a human. The ratings are subjective, and one rater is a small panel.",
    "The topic given to the agent is the held-out paper's title, which is a simple choice and may favour some papers.",
]


# ------------------------------------------------------------------ choosing the held-out papers
def pick_heldout(fetched: List[Paper], indexed: List[Paper], n: int) -> Tuple[List[Paper], str]:
    """Papers that are not in the index AND were published after the newest indexed paper. Returns (papers, cutoff)."""
    known = {p.uid for p in indexed}
    cutoff = max((p.published for p in indexed if p.published), default="")
    seen: set = set()
    chosen: List[Paper] = []
    for p in fetched:
        if p.uid in known or p.uid in seen or not p.abstract.strip() or not p.published or p.published <= cutoff:
            continue
        seen.add(p.uid)
        chosen.append(p)
        if len(chosen) == n:
            break
    return chosen, cutoff


# ------------------------------------------------------------------ automatic checks on one saved run
def auto_checks(record: Dict[str, Any]) -> Dict[str, Any]:
    hg, rg = record.get("hypothesis_grounding") or {}, record.get("review_grounding") or {}
    exp, critic = record.get("experiment") or {}, record.get("critic") or {}
    unknown = set(hg.get("unknown_ids") or []) | set(rg.get("unknown_ids") or [])
    note = " ".join([str(exp.get("note", ""))] + [str(w) for w in record.get("warnings") or []])
    return {
        "hypothesis": (record.get("hypothesis") or {}).get("statement", ""),
        "hypothesis_cites_retrieved_paper": bool(hg.get("cited_ids")),
        "invalid_citations": len(unknown),
        "execution": record.get("execution_status", "NOT_RUN"),
        "critic_verdict": critic.get("verdict") or "-",
        "design": exp.get("design_summary", ""),
        "design_source": exp.get("source", ""),
        "design_mismatch": "Possible mismatch" in note,
    }


def similarity_matrix(hypotheses: List[str], abstracts: List[str], embed: Callable[[List[str]], Any]) -> List[List[float]]:
    """Cosine similarity of every hypothesis (rows) with every abstract (columns)."""
    def unit(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)
    return (unit(embed(hypotheses)) @ unit(embed(abstracts)).T).tolist()


# ------------------------------------------------------------------ the evaluation itself
def evaluate_heldout(papers: List[Paper], run_topic: Callable[[str], Path], embed: Callable[[List[str]], Any],
                     out_dir: Path, cutoff: str = "") -> Dict[str, Any]:
    """Run the pipeline on each held-out paper's title and write results.json, rating_sheet.csv and summary.md.
    ``run_topic`` runs the pipeline and returns the path of the saved run record. The results file is rewritten
    after every run, so an interrupted evaluation keeps what it already finished."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    held = [{"id": p.uid, "title": p.title, "published": p.published, "url": p.url, "abstract": p.abstract} for p in papers]
    rows: List[Dict[str, Any]] = []
    for p in papers:
        row: Dict[str, Any] = {"held_out_id": p.uid, "held_out_title": p.title, "held_out_url": p.url, "topic": p.title}
        try:
            path = Path(run_topic(p.title))
            record = json.loads(path.read_text(encoding="utf-8"))
            row.update(run_file=path.name, **auto_checks(record))
        except Exception as exc:     # one failing run (model, network, disk) must not stop the whole evaluation
            logger.exception("Evaluation run failed for %s", p.uid)
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        _write_json(out_dir, held, rows, cutoff, {})
    _add_similarity(rows, papers, embed)
    summary = summarize_results(rows)
    _write_json(out_dir, held, rows, cutoff, summary)
    _write_sheet(out_dir / "rating_sheet.csv", rows)
    (out_dir / "summary.md").write_text(summary_markdown(summary, len(papers), cutoff), encoding="utf-8")
    return {"rows": rows, "summary": summary, "out_dir": out_dir}


def _add_similarity(rows: List[Dict[str, Any]], papers: List[Paper], embed: Callable[[List[str]], Any]) -> None:
    idx = [i for i, r in enumerate(rows) if r.get("hypothesis")]
    if not idx or not papers:
        return
    try:
        sims = similarity_matrix([rows[i]["hypothesis"] for i in idx], [p.abstract for p in papers], embed)
    except (EmbeddingError, OSError, ValueError) as exc:
        logger.warning("Similarity could not be computed: %s", exc)
        for i in idx:
            rows[i]["error"] = (rows[i].get("error", "") + f" similarity not computed: {exc}").strip()
        return
    for k, i in enumerate(idx):
        own = sims[k][i]
        others = [s for j, s in enumerate(sims[k]) if j != i]
        rows[i]["similarity_to_own_abstract"] = round(own, 4)
        rows[i]["mean_similarity_to_other_abstracts"] = round(float(np.mean(others)), 4) if others else None
        rows[i]["own_abstract_rank"] = 1 + sum(1 for s in sims[k] if s > own)


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ran = [r for r in rows if "run_file" in r]
    own = [r["similarity_to_own_abstract"] for r in rows if r.get("similarity_to_own_abstract") is not None]
    other = [r["mean_similarity_to_other_abstracts"] for r in rows if r.get("mean_similarity_to_other_abstracts") is not None]
    return {
        "papers": len(rows), "runs_finished": len(ran), "runs_failed": len(rows) - len(ran),
        "with_hypothesis": sum(1 for r in ran if r.get("hypothesis")),
        "hypothesis_cites_retrieved_paper": sum(1 for r in ran if r.get("hypothesis_cites_retrieved_paper")),
        "invalid_citations_total": sum(int(r.get("invalid_citations") or 0) for r in ran),
        "execution": dict(Counter(r.get("execution") for r in ran)),
        "critic_verdicts": dict(Counter(r.get("critic_verdict") for r in ran)),
        "design_from_fallback_rules": sum(1 for r in ran if r.get("design_source") in ("keyword_spec", "default_spec")),
        "design_mismatch_warnings": sum(1 for r in ran if r.get("design_mismatch")),
        "mean_similarity_own": round(statistics.mean(own), 4) if own else None,
        "mean_similarity_others": round(statistics.mean(other), 4) if other else None,
        "own_abstract_ranked_first": sum(1 for r in rows if r.get("own_abstract_rank") == 1),
        "similarity_count": len(own),
    }


def summary_markdown(s: Dict[str, Any], n_papers: int, cutoff: str) -> str:
    lines = ["# Holdout evaluation", "",
             f"Held-out papers: {n_papers} (published after {cutoff or 'the newest indexed paper'} and not in the index).", "",
             "## Automatic checks", "",
             f"- Runs finished: {s['runs_finished']} of {s['papers']} (failed: {s['runs_failed']})",
             f"- Runs with a hypothesis: {s['with_hypothesis']}",
             f"- Hypotheses that cite a retrieved paper: {s['hypothesis_cites_retrieved_paper']} of {s['runs_finished']}",
             f"- Cited ids that were never retrieved (total): {s['invalid_citations_total']}",
             f"- Sandbox results: {s['execution'] or 'none'}",
             f"- Critic verdicts: {s['critic_verdicts'] or 'none'}",
             f"- Designs chosen by fixed fallback rules: {s['design_from_fallback_rules']}",
             f"- Design/hypothesis mismatch warnings: {s['design_mismatch_warnings']}"]
    if s["mean_similarity_own"] is not None:
        lines += [f"- Mean similarity of a hypothesis to its own held-out abstract: {s['mean_similarity_own']}",
                  f"- Mean similarity to the other held-out abstracts: {s['mean_similarity_others']}",
                  f"- Own abstract ranked first: {s['own_abstract_ranked_first']} of {s['similarity_count']}"]
    lines += ["", "## Human ratings", "",
              "Open rating_sheet.csv, read each hypothesis next to the held-out paper, and fill in relevance, testability "
              "and groundedness (1 to 5), then run: python -m src.main evaluate --ratings FILE", "",
              "## Limits of this evaluation", ""] + [f"- {x}" for x in LIMITS]
    return "\n".join(lines) + "\n"


def _write_json(out_dir: Path, held: List[Dict[str, Any]], rows: List[Dict[str, Any]], cutoff: str, summary: Dict[str, Any]) -> None:
    payload = {"cutoff_published_after": cutoff, "held_out": held, "rows": rows, "summary": summary, "limits": LIMITS}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_sheet(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:     # utf-8-sig so Excel opens it correctly
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r.get(c) is None else r.get(c, "")) for c in SHEET_COLUMNS})


# ------------------------------------------------------------------ reading the human ratings back
def summarize_ratings(csv_path: Path) -> str:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "The rating sheet has no rows."
    out = [f"Rating sheet: {len(rows)} row(s)."]
    for col in RATING_COLUMNS:
        values, bad = [], []
        for r in rows:
            raw = (r.get(col) or "").strip()
            if not raw:
                continue
            if raw.isdigit() and 1 <= int(raw) <= 5:
                values.append(int(raw))
            else:
                bad.append(f"{r.get('held_out_id')}: {raw!r}")
        mean = f"mean {statistics.mean(values):.2f}" if values else "no ratings yet"
        out.append(f"- {col}: {len(values)} rated, {mean}" + (f"; ignored (not a whole number 1 to 5): {', '.join(bad)}" if bad else ""))
    out.append("Small sample, one rater: treat these averages as a rough indication only.")
    return "\n".join(out)
