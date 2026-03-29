from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .config import APP_DATA_DIR, DB_PATH


def _utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SearchDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS roots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    root_id INTEGER NOT NULL,
                    rel_path TEXT NOT NULL,
                    abs_path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    subtitle_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    last_indexed_at TEXT NOT NULL,
                    UNIQUE(root_id, rel_path),
                    FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    root_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE,
                    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, model_name),
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_files_root_id ON files(root_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_root_id ON chunks(root_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings(model_name);
                """
            )

    def ensure_root(self, conn: sqlite3.Connection, root_path: str) -> int:
        row = conn.execute("SELECT id FROM roots WHERE path = ?", (root_path,)).fetchone()
        now = _utc_now()
        if row:
            conn.execute("UPDATE roots SET updated_at = ? WHERE id = ?", (now, row["id"]))
            return int(row["id"])

        cursor = conn.execute(
            "INSERT INTO roots(path, created_at, updated_at) VALUES(?, ?, ?)",
            (root_path, now, now),
        )
        return int(cursor.lastrowid)

    def get_root_id(self, conn: sqlite3.Connection, root_path: str) -> int | None:
        row = conn.execute("SELECT id FROM roots WHERE path = ?", (root_path,)).fetchone()
        return int(row["id"]) if row else None

    def get_file_rows(self, conn: sqlite3.Connection, root_id: int) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM files WHERE root_id = ? ORDER BY rel_path",
            (root_id,),
        ).fetchall()

    def upsert_file(
        self,
        conn: sqlite3.Connection,
        root_id: int,
        rel_path: str,
        abs_path: str,
        size: int,
        mtime: float,
        subtitle_count: int,
        chunk_count: int,
    ) -> int:
        now = _utc_now()
        row = conn.execute(
            "SELECT id FROM files WHERE root_id = ? AND rel_path = ?",
            (root_id, rel_path),
        ).fetchone()
        if row:
            file_id = int(row["id"])
            conn.execute(
                """
                UPDATE files
                SET abs_path = ?, size = ?, mtime = ?, subtitle_count = ?, chunk_count = ?, last_indexed_at = ?
                WHERE id = ?
                """,
                (abs_path, size, mtime, subtitle_count, chunk_count, now, file_id),
            )
            return file_id

        cursor = conn.execute(
            """
            INSERT INTO files(root_id, rel_path, abs_path, size, mtime, subtitle_count, chunk_count, last_indexed_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (root_id, rel_path, abs_path, size, mtime, subtitle_count, chunk_count, now),
        )
        return int(cursor.lastrowid)

    def delete_missing_files(
        self,
        conn: sqlite3.Connection,
        root_id: int,
        keep_rel_paths: Sequence[str],
    ) -> int:
        existing = {
            row["rel_path"]: row["id"]
            for row in conn.execute("SELECT id, rel_path FROM files WHERE root_id = ?", (root_id,))
        }
        stale_ids = [file_id for rel_path, file_id in existing.items() if rel_path not in keep_rel_paths]
        for file_id in stale_ids:
            self.delete_file_chunks(conn, file_id)
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        return len(stale_ids)

    def delete_file_chunks(self, conn: sqlite3.Connection, file_id: int) -> None:
        conn.execute(
            "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
            (file_id,),
        )
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    def insert_chunks(
        self,
        conn: sqlite3.Connection,
        root_id: int,
        file_id: int,
        chunk_rows: Iterable[tuple[str, int, int, int, float, float, str]],
    ) -> None:
        conn.executemany(
            """
            INSERT INTO chunks(id, root_id, file_id, chunk_index, line_start, line_end, start_sec, end_sec, text)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ((chunk_id, root_id, file_id, chunk_index, line_start, line_end, start_sec, end_sec, text)
             for chunk_id, chunk_index, line_start, line_end, start_sec, end_sec, text in chunk_rows),
        )

    def insert_embeddings(
        self,
        conn: sqlite3.Connection,
        rows: Iterable[tuple[str, str, int, bytes]],
    ) -> None:
        now = _utc_now()
        conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings(chunk_id, model_name, dim, vector, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            ((chunk_id, model_name, dim, vector, now) for chunk_id, model_name, dim, vector in rows),
        )

    def fetch_summary(self, conn: sqlite3.Connection, root_path: str, model_name: str) -> dict | None:
        root_id = self.get_root_id(conn, root_path)
        if root_id is None:
            return None

        file_counts = conn.execute(
            """
            SELECT COUNT(*) AS total_files,
                   COALESCE(SUM(chunk_count), 0) AS total_chunks,
                   MAX(last_indexed_at) AS last_indexed_at
            FROM files
            WHERE root_id = ?
            """,
            (root_id,),
        ).fetchone()
        indexed_files = conn.execute(
            """
            SELECT COUNT(DISTINCT chunks.file_id) AS indexed_files
            FROM chunks
            JOIN embeddings ON embeddings.chunk_id = chunks.id
            WHERE chunks.root_id = ? AND embeddings.model_name = ?
            """,
            (root_id, model_name),
        ).fetchone()

        return {
            "root_path": root_path,
            "root_id": root_id,
            "total_files": int(file_counts["total_files"] or 0),
            "total_chunks": int(file_counts["total_chunks"] or 0),
            "indexed_files": int(indexed_files["indexed_files"] or 0),
            "last_indexed_at": file_counts["last_indexed_at"],
            "model_name": model_name,
        }
