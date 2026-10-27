"""Memory of earlier runs, so a longer research task does not repeat itself.

Each finished run adds one line to ``storage/memory.jsonl``. Before the hypothesis and experiment are
written, the most relevant earlier attempts on a similar topic are rendered as a short, size-limited note
that goes into the prompt. This is context management, not learning: the model is only SHOWN what happened.

Mock-model (TEST MODE) runs are kept out of the memory of real runs, so placeholder text never steers a
real model.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = frozenset("the and for with from that this into using use based between over under about study".split())
HEADER = ("Previous attempts on similar topics (newest first). Do not repeat a design that already failed or was "
          "inconclusive unless you change something important.")


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


class MemoryStore:
    def __init__(self, path: Path, max_entries: int = 500) -> None:
        self.path = Path(path)
        self.max_entries = max_entries

    def add(self, **fields: Any) -> None:
        entry = {"when": time.strftime("%Y-%m-%d %H:%M"), **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        entries = self.entries()
        if len(entries) > self.max_entries:                      # keep the file small
            self.path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries[-self.max_entries:]),
                                 encoding="utf-8")

    def entries(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue                                          # skip damaged lines
            if isinstance(value, dict):
                out.append(value)
        return out

    def similar(self, topic: str, limit: int = 5, include_mock: bool = False) -> List[Dict[str, Any]]:
        wanted = _tokens(topic)
        scored = []
        for entry in self.entries():
            if not include_mock and "TEST MODE" in str(entry.get("llm_mode", "")):
                continue
            overlap = len(wanted & _tokens(str(entry.get("topic", ""))))
            if overlap >= 2 or (wanted and overlap / len(wanted) >= 0.5):
                scored.append(entry)
        return list(reversed(scored))[:limit]                     # newest first

    def render(self, topic: str, budget_chars: int = 1200, include_mock: bool = False) -> str:
        lines: List[str] = []
        used = len(HEADER)
        for e in self.similar(topic, include_mock=include_mock):
            line = (f"- {e.get('when', '?')}: hypothesis: {str(e.get('hypothesis', ''))[:200]} | "
                    f"experiment: {str(e.get('design', 'unknown'))[:120]} | "
                    f"result: {e.get('execution_status', '?')}, critic: {e.get('critic_verdict', 'none')}")
            if used + len(line) > budget_chars:
                break
            lines.append(line)
            used += len(line)
        return (HEADER + "\n" + "\n".join(lines)) if lines else ""
