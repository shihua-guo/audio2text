from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .database import SearchDatabase
from .embeddings import QwenEmbeddingBackend
from .models import IndexJobStatus, SearchResult
from .srt_parser import chunk_subtitles, parse_srt_file


ProgressCallback = Callable[[IndexJobStatus], None]


class SemanticSearchService:
    def __init__(self, db: SearchDatabase, embedding_backend: QwenEmbeddingBackend) -> None:
        self.db = db
        self.embedding_backend = embedding_backend
        self._cache: dict[tuple[str, str], tuple[np.ndarray, list[dict]]] = {}

    def _invalidate_cache(self, root_path: str, model_name: str) -> None:
        self._cache.pop((root_path, model_name), None)

    def summarize_root(self, root_path: str, model_name: str) -> dict | None:
        with self.db.connect() as conn:
            return self.db.fetch_summary(conn, str(Path(root_path).resolve()), model_name)

    def index_root(
        self,
        root_path: str,
        model_name: str,
        force_rebuild: bool,
        callback: ProgressCallback | None = None,
        job: IndexJobStatus | None = None,
    ) -> dict:
        root = Path(root_path).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"目录不存在: {root}")

        srt_files = sorted(root.rglob("*.srt"))
        deleted_count = 0

        if job:
            job.total_files = len(srt_files)
            job.state = "running"
            job.message = "正在扫描字幕文件"
            if callback:
                callback(job)

        with self.db.connect() as conn:
            root_id = self.db.ensure_root(conn, str(root))
            known_files = {
                row["rel_path"]: row
                for row in self.db.get_file_rows(conn, root_id)
            }
            rel_paths = [str(file_path.relative_to(root)) for file_path in srt_files]
            deleted_count = self.db.delete_missing_files(conn, root_id, rel_paths)

            if job:
                job.deleted_files = deleted_count
                if callback:
                    callback(job)

            for index, file_path in enumerate(srt_files, start=1):
                rel_path = str(file_path.relative_to(root))
                stat = file_path.stat()
                existing = known_files.get(rel_path)
                unchanged = (
                    not force_rebuild
                    and existing is not None
                    and int(existing["size"]) == int(stat.st_size)
                    and float(existing["mtime"]) == float(stat.st_mtime)
                )

                if job:
                    job.current_file = rel_path
                    job.message = f"正在处理 {rel_path}"
                    if callback:
                        callback(job)

                if unchanged:
                    if job:
                        job.skipped_files += 1
                        job.processed_files = index
                        if callback:
                            callback(job)
                    continue

                lines = parse_srt_file(file_path)
                chunks = chunk_subtitles(lines, rel_path=rel_path)

                file_id = self.db.upsert_file(
                    conn,
                    root_id=root_id,
                    rel_path=rel_path,
                    abs_path=str(file_path),
                    size=int(stat.st_size),
                    mtime=float(stat.st_mtime),
                    subtitle_count=len(lines),
                    chunk_count=len(chunks),
                )
                self.db.delete_file_chunks(conn, file_id)
                self.db.insert_chunks(
                    conn,
                    root_id=root_id,
                    file_id=file_id,
                    chunk_rows=[
                        (
                            chunk.chunk_id,
                            chunk.chunk_index,
                            chunk.line_start,
                            chunk.line_end,
                            chunk.start_sec,
                            chunk.end_sec,
                            chunk.text,
                        )
                        for chunk in chunks
                    ],
                )

                embeddings = (
                    self.embedding_backend.encode_documents(
                        [chunk.text for chunk in chunks],
                        model_name=model_name,
                    )
                    if chunks
                    else np.zeros((0, 0), dtype=np.float32)
                )

                self.db.insert_embeddings(
                    conn,
                    rows=[
                        (
                            chunk.chunk_id,
                            model_name,
                            int(vector.shape[0]),
                            vector.astype(np.float32).tobytes(),
                        )
                        for chunk, vector in zip(chunks, embeddings)
                    ],
                )

                if job:
                    job.indexed_files += 1
                    job.processed_files = index
                    if callback:
                        callback(job)

        self._invalidate_cache(str(root), model_name)
        summary = self.summarize_root(str(root), model_name)
        return {
            "root_path": str(root),
            "model_name": model_name,
            "summary": summary,
            "deleted_files": deleted_count,
            "total_files": len(srt_files),
        }

    def _load_search_matrix(self, root_path: str, model_name: str) -> tuple[np.ndarray, list[dict]]:
        cache_key = (root_path, model_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        with self.db.connect() as conn:
            root_id = self.db.get_root_id(conn, root_path)
            if root_id is None:
                return np.zeros((0, 0), dtype=np.float32), []

            rows = conn.execute(
                """
                SELECT chunks.id AS chunk_id,
                       chunks.chunk_index,
                       chunks.line_start,
                       chunks.line_end,
                       chunks.start_sec,
                       chunks.end_sec,
                       chunks.text,
                       files.rel_path,
                       files.abs_path,
                       chunks.file_id,
                       embeddings.vector,
                       embeddings.dim
                FROM chunks
                JOIN files ON files.id = chunks.file_id
                JOIN embeddings ON embeddings.chunk_id = chunks.id
                WHERE chunks.root_id = ? AND embeddings.model_name = ?
                ORDER BY files.rel_path, chunks.chunk_index
                """,
                (root_id, model_name),
            ).fetchall()

            if not rows:
                return np.zeros((0, 0), dtype=np.float32), []

            items = []
            vectors = []
            for row in rows:
                vectors.append(np.frombuffer(row["vector"], dtype=np.float32, count=int(row["dim"])))
                items.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "chunk_index": int(row["chunk_index"]),
                        "line_start": int(row["line_start"]),
                        "line_end": int(row["line_end"]),
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "text": row["text"],
                        "rel_path": row["rel_path"],
                        "abs_path": row["abs_path"],
                        "file_id": int(row["file_id"]),
                    }
                )

        matrix = np.vstack(vectors).astype(np.float32)
        self._cache[cache_key] = (matrix, items)
        return matrix, items

    def search(self, root_path: str, model_name: str, query: str, limit: int = 10) -> list[SearchResult]:
        root = str(Path(root_path).resolve())
        matrix, items = self._load_search_matrix(root, model_name)
        if matrix.size == 0 or not items:
            return []

        query_vector = self.embedding_backend.encode_query(query, model_name=model_name)
        scores = matrix @ query_vector
        top_indices = np.argsort(-scores)[:limit]

        results: list[SearchResult] = []
        with self.db.connect() as conn:
            for idx in top_indices:
                item = items[int(idx)]
                context_rows = conn.execute(
                    """
                    SELECT chunk_index, text
                    FROM chunks
                    WHERE file_id = ? AND chunk_index BETWEEN ? AND ?
                    ORDER BY chunk_index
                    """,
                    (item["file_id"], max(item["chunk_index"] - 1, 0), item["chunk_index"] + 1),
                ).fetchall()

                prev_text = ""
                next_text = ""
                for row in context_rows:
                    if int(row["chunk_index"]) == item["chunk_index"] - 1:
                        prev_text = row["text"]
                    elif int(row["chunk_index"]) == item["chunk_index"] + 1:
                        next_text = row["text"]

                results.append(
                    SearchResult(
                        chunk_id=item["chunk_id"],
                        score=float(scores[int(idx)]),
                        rel_path=item["rel_path"],
                        abs_path=item["abs_path"],
                        start_sec=item["start_sec"],
                        end_sec=item["end_sec"],
                        text=item["text"],
                        chunk_index=item["chunk_index"],
                        line_start=item["line_start"],
                        line_end=item["line_end"],
                        prev_text=prev_text,
                        next_text=next_text,
                    )
                )

        return results
