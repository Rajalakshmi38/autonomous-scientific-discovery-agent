"""SQLite-backed storage for paper metadata (standard library only)."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from src.models import Paper

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    uid TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class PaperStore:
    def __init__(self, db_path: Path | str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def upsert_many(self, papers: Iterable[Paper]) -> int:
        rows = [(p.uid, json.dumps(p.to_dict(), ensure_ascii=False)) for p in papers]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO papers (uid, data) VALUES (?, ?) "
                "ON CONFLICT(uid) DO UPDATE SET data = excluded.data",
                rows,
            )
        return len(rows)

    def get(self, uid: str) -> Optional[Paper]:
        row = self._conn.execute("SELECT data FROM papers WHERE uid = ?", (uid,)).fetchone()
        return Paper.from_dict(json.loads(row[0])) if row else None

    def all(self) -> List[Paper]:
        rows = self._conn.execute("SELECT data FROM papers ORDER BY ingested_at, uid").fetchall()
        return [Paper.from_dict(json.loads(r[0])) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
