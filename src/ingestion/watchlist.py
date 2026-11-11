"""A saved list of searches that `update` re-runs, so the index can be refreshed regularly.

Nothing runs in the background. To refresh on a schedule, schedule the command `python -m src.main update`
(for example with Windows Task Scheduler). Each update only stores and indexes papers that are not yet in the index.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

SOURCES = ("arxiv", "pubmed")


@dataclass
class WatchEntry:
    source: str
    query: str
    max_results: int = 20
    category: Optional[str] = None

    def describe(self) -> str:
        extra = f", category {self.category}" if self.category else ""
        return f"{self.source}: \"{self.query}\" (up to {self.max_results}{extra})"


def validate(entry: WatchEntry) -> None:
    if entry.source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if not entry.query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= entry.max_results <= 100:
        raise ValueError("max-results must be between 1 and 100")
    if entry.category and entry.source != "arxiv":
        raise ValueError("a category can only be used with the arxiv source")


def load(path: Path) -> List[WatchEntry]:
    if not Path(path).exists():
        return []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return [WatchEntry(**item) for item in raw]
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"The watch list {path} is damaged ({exc}). Fix or delete the file.") from exc


def save(path: Path, entries: List[WatchEntry]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")


def add(path: Path, entry: WatchEntry) -> bool:
    """Add an entry. Returns False when the same search is already on the list."""
    validate(entry)
    entries = load(path)
    if entry in entries:
        return False
    save(path, entries + [entry])
    return True


def remove(path: Path, number: int) -> WatchEntry:
    entries = load(path)
    if not 1 <= number <= len(entries):
        raise ValueError(f"There is no entry {number}. The list has {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}.")
    gone = entries.pop(number - 1)
    save(path, entries)
    return gone
