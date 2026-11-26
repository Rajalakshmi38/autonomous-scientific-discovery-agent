"""Read-only helpers for the Evaluation and Learning (bandit) sections of the dashboard.

They only READ files that other commands wrote (`evaluate` and `run --adaptive`). Nothing here writes, deletes or runs a model.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

_FOLDER = re.compile(r"^holdout_\d{8}_\d{6}$")
RATING_FIELDS = ("relevance_1to5", "testability_1to5", "groundedness_1to5")


def evaluation_folders(storage_dir: Path) -> List[Path]:
    """Evaluation result folders (storage/evaluation/holdout_*), newest first."""
    root = Path(storage_dir) / "evaluation"
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and _FOLDER.match(p.name)), reverse=True)


def _number(text: Any, as_int: bool = False) -> Optional[float]:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return int(value) if as_int else value


def load_evaluation(folder: Path) -> Dict[str, Any]:
    """The saved summary text and one row per held-out paper. Missing files give empty results, not errors."""
    folder = Path(folder)
    summary = ""
    summary_file = folder / "summary.md"
    if summary_file.is_file():
        lines = summary_file.read_text(encoding="utf-8-sig").splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]                                 # the folder name is shown instead of the big title
        summary = "\n".join(re.sub(r"^## ", "#### ", line) for line in lines).strip()
    rows: List[Dict[str, Any]] = []
    rated = 0
    sheet = folder / "rating_sheet.csv"
    if sheet.is_file():
        with sheet.open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                scores = [str(raw.get(f) or "").strip() for f in RATING_FIELDS]
                filled = all(scores)
                rated += 1 if filled else 0
                rows.append({
                    "paper": raw.get("held_out_title") or "",
                    "cites": "yes" if str(raw.get("hypothesis_cites_retrieved_paper")).strip().upper() == "TRUE" else "no",
                    "execution": raw.get("execution") or "",
                    "verdict": raw.get("critic_verdict") or "-",
                    "similarity": _number(raw.get("similarity_to_own_abstract")),
                    "rank": _number(raw.get("own_abstract_rank"), as_int=True),
                    "ratings": ("relevance %s, testability %s, groundedness %s" % tuple(scores)) if filled else "not rated yet",
                    "rated_by": (raw.get("rater") or "").strip() or ("not recorded" if filled else "-"),
                    "relevance": _number(scores[0], as_int=True), "testability": _number(scores[1], as_int=True),
                    "groundedness": _number(scores[2], as_int=True), "comments": (raw.get("comments") or "").strip(),
                })
    return {"summary": summary, "rows": rows, "rated": rated}


def bandit_rows(path: Path, epsilon: float = 0.2) -> Dict[str, Any]:
    """What the strategy bandit has learned so far (the same numbers as `python -m src.main bandit`)."""
    path = Path(path)
    if not path.is_file():
        return {"total": 0, "rows": [], "best": None}
    from src.rl.bandit import StrategyBandit
    bandit = StrategyBandit(path, epsilon=epsilon)            # reading only: nothing is saved until an update
    rows = [{"papers": arm, "runs": int(bandit.stats[str(arm)]["n"]),
             "average": f"{bandit.mean(arm):.2f}" if bandit.stats[str(arm)]["n"] else "not tried yet"} for arm in bandit.arms]
    return {"total": bandit.total_runs(), "rows": rows, "best": bandit.best_arm()}


class RatingError(ValueError):
    """The ratings cannot be saved (a message for the person)."""


def save_ratings(sheet: Path, scores: List[Dict[str, Any]], rater: str) -> int:
    """Write 1 to 5 ratings into an evaluation's rating_sheet.csv. Returns how many papers were changed.

    `scores` has one dict per row, in the sheet's order, with keys relevance, testability, groundedness (1 to 5 or None) and
    comments. Only rows whose ratings or comment changed get the new rater name, so earlier ratings keep their own rater.
    The previous sheet is kept as rating_sheet.previous.csv, and the new one is written whole or not at all.
    """
    sheet = Path(sheet)
    if not sheet.is_file():
        raise RatingError("This evaluation has no rating sheet.")
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    needed = list(RATING_FIELDS)
    if any(f not in fields for f in needed) or "rater" not in fields:
        raise RatingError("The rating sheet does not have the expected columns.")
    if len(scores) != len(rows):
        raise RatingError("The sheet changed while you were rating. Reload the page and try again.")
    rater = " ".join(str(rater or "").split())[:80]
    changed = 0
    updated = []
    for row, new in zip(rows, scores):
        row = dict(row)
        values = []
        for key, field in zip(("relevance", "testability", "groundedness"), RATING_FIELDS):
            value = new.get(key)
            if value in (None, ""):
                values.append("")
                continue
            try:
                number = int(float(value))
            except (TypeError, ValueError):
                raise RatingError(f"Ratings must be whole numbers from 1 to 5 (got {value!r}).")
            if not 1 <= number <= 5:
                raise RatingError(f"Ratings must be from 1 to 5 (got {number}).")
            values.append(str(number))
        comment = " ".join(str(new.get("comments") or "").split())[:500]
        old = [str(row.get(f) or "").strip() for f in RATING_FIELDS]
        if values != old or ("comments" in row and comment != str(row.get("comments") or "").strip()):
            changed += 1
            for field, value in zip(RATING_FIELDS, values):
                row[field] = value
            if "comments" in row:
                row["comments"] = comment
            row["rater"] = rater
        updated.append(row)
    if changed and not rater:
        raise RatingError("Type your name as the rater, so it is clear who made these ratings.")
    if not changed:
        return 0
    tmp = sheet.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated)
    shutil.copyfile(sheet, sheet.with_name("rating_sheet.previous.csv"))
    os.replace(tmp, sheet)
    return changed


def reset_bandit(path: Path) -> bool:
    """The same as `python -m src.main bandit --reset`, but keeps a copy (bandit.json.bak) so a mistake can be undone.
    Returns True when there was something to reset."""
    path = Path(path)
    if not path.is_file():
        return False
    shutil.copyfile(path, path.with_name(path.name + ".bak"))
    path.unlink()
    return True
