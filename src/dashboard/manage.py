"""Delete saved research runs safely. No UI code and no language model here.

What belongs to ONE run (the existing storage layout, nothing new):
    storage/runs/run_<timestamp>.json                         the run record (it also holds the experiment code,
                                                              the execution attempts and the citations)
    storage/drafts/run_<timestamp>_draft.md / .tex            drafts written from that record
    storage/drafts/run_<timestamp>_failed_run_report.md / .tex

What is SHARED and is never touched here: storage/papers.db, storage/vectors/, storage/memory.jsonl (its entries carry
no run id, so they cannot be attributed to one run), storage/bandit.json, storage/watchlist.json, storage/logs/app.log
(one rotating log for every run), storage/evaluation/, data/ and config/.

Safety rules: only exact, validated file names are ever deleted (no wildcards on user input, no paths), nothing outside
the runs and drafts folders is reachable, a missing file is not an error, and the run record is kept when one of its drafts
could not be deleted so the person can try again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

RUN_NAME = re.compile(r"^run_\d{8}_\d{6}(?:_\d+)?\.json$")
DRAFT_NAME = re.compile(r"^run_\d{8}_\d{6}(?:_\d+)?_(?:draft|failed_run_report)\.(?:md|tex)$")
_DRAFT_SUFFIXES = ("_draft.md", "_draft.tex", "_failed_run_report.md", "_failed_run_report.tex")


@dataclass
class DeleteResult:
    deleted: List[str] = field(default_factory=list)     # names of files that were removed
    missing: List[str] = field(default_factory=list)     # run records that were already gone
    errors: List[str] = field(default_factory=list)      # human-readable problems

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def runs_deleted(self) -> int:
        return sum(1 for n in self.deleted if RUN_NAME.fullmatch(n))


def is_run_file_name(name: str) -> bool:
    return bool(RUN_NAME.fullmatch(str(name)))


def draft_files_for(drafts_dir: Path, run_name: str) -> List[Path]:
    """The drafts that exist for one run. Exact names only, so run_X never matches run_X_2."""
    if not is_run_file_name(run_name):
        return []
    stem = run_name[: -len(".json")]
    found = [Path(drafts_dir) / f"{stem}{suffix}" for suffix in _DRAFT_SUFFIXES]
    return [p for p in found if p.is_file() or p.is_symlink()]


def _unlink(path: Path, result: DeleteResult) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return True                                   # already gone: the goal is reached
    except OSError as exc:
        result.errors.append(f"Could not delete {path.name}: {exc.strerror or exc}")
        return False
    result.deleted.append(path.name)
    return True


def delete_run(runs_dir: Path, drafts_dir: Path, run_name: str) -> DeleteResult:
    """Delete one run record and its own drafts. Deleting a run that is already gone is reported, not raised."""
    result = DeleteResult()
    if not is_run_file_name(run_name):
        result.errors.append(f"{run_name!r} is not a run file name, so nothing was deleted.")
        return result
    record = Path(runs_dir) / run_name
    for draft in draft_files_for(drafts_dir, run_name):
        _unlink(draft, result)
    if result.errors:                                 # keep the record so the person can retry
        result.errors.append("The run record was kept because a draft could not be deleted.")
        return result
    if not (record.is_file() or record.is_symlink()):
        result.missing.append(run_name)
        return result
    _unlink(record, result)
    return result


def delete_all_runs(runs_dir: Path, drafts_dir: Path) -> DeleteResult:
    """Delete every run record (readable or not) and every run draft. Nothing else in either folder is touched."""
    result = DeleteResult()
    runs_dir, drafts_dir = Path(runs_dir), Path(drafts_dir)
    try:
        records = sorted(p.name for p in runs_dir.iterdir() if RUN_NAME.fullmatch(p.name)) if runs_dir.is_dir() else []
    except OSError as exc:
        result.errors.append(f"Could not read the runs folder: {exc.strerror or exc}")
        return result
    for name in records:
        part = delete_run(runs_dir, drafts_dir, name)
        result.deleted += part.deleted
        result.missing += part.missing
        result.errors += part.errors
    try:                                              # drafts whose run record is already gone
        leftovers = sorted(p for p in drafts_dir.iterdir() if DRAFT_NAME.fullmatch(p.name)) if drafts_dir.is_dir() else []
    except OSError as exc:
        result.errors.append(f"Could not read the drafts folder: {exc.strerror or exc}")
        return result
    for path in leftovers:
        _unlink(path, result)
    return result
