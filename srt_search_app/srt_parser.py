from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pysrt

from .config import CHUNKING
from .models import SubtitleChunk


@dataclass
class SubtitleLine:
    index: int
    start_sec: float
    end_sec: float
    text: str


def _normalize_subtitle_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def parse_srt_file(file_path: Path) -> list[SubtitleLine]:
    subtitles = pysrt.open(str(file_path), encoding="utf-8")
    lines: list[SubtitleLine] = []
    for subtitle in subtitles:
        text = _normalize_subtitle_text(subtitle.text)
        if not text:
            continue
        lines.append(
            SubtitleLine(
                index=int(subtitle.index),
                start_sec=subtitle.start.ordinal / 1000.0,
                end_sec=subtitle.end.ordinal / 1000.0,
                text=text,
            )
        )
    return lines


def chunk_subtitles(lines: list[SubtitleLine], rel_path: str) -> list[SubtitleChunk]:
    if not lines:
        return []

    chunks: list[SubtitleChunk] = []
    start = 0
    chunk_index = 0

    while start < len(lines):
        end = min(start + CHUNKING.max_lines, len(lines))
        current = lines[start:end]

        while len(current) > 1:
            text = " ".join(line.text for line in current)
            duration = current[-1].end_sec - current[0].start_sec
            if len(text) <= CHUNKING.max_chars and duration <= CHUNKING.max_duration_seconds:
                break
            current = current[:-1]
            end -= 1

        text = " ".join(line.text for line in current).strip()
        if text:
            source = f"{rel_path}|{chunk_index}|{current[0].index}|{current[-1].index}|{text}"
            chunk_id = hashlib.sha1(source.encode("utf-8")).hexdigest()
            chunks.append(
                SubtitleChunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    line_start=current[0].index,
                    line_end=current[-1].index,
                    start_sec=current[0].start_sec,
                    end_sec=current[-1].end_sec,
                    text=text,
                )
            )
            chunk_index += 1

        if end >= len(lines):
            break
        start = max(end - CHUNKING.overlap_lines, start + 1)

    return chunks
