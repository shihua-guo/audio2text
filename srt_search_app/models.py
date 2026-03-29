from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SubtitleChunk:
    chunk_id: str
    chunk_index: int
    line_start: int
    line_end: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    rel_path: str
    abs_path: str
    start_sec: float
    end_sec: float
    text: str
    chunk_index: int
    line_start: int
    line_end: int
    prev_text: str = ""
    next_text: str = ""


@dataclass
class IndexJobStatus:
    job_id: str
    root_path: str
    model_name: str
    force_rebuild: bool
    state: str = "queued"
    message: str = ""
    total_files: int = 0
    processed_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    current_file: str = ""
    error: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: Optional[str] = None
