from __future__ import annotations

from dataclasses import dataclass

from portable_runtime import DATA_DIR, RESOURCE_ROOT, load_runtime_config


RUNTIME = load_runtime_config()
PROJECT_ROOT = RESOURCE_ROOT
APP_DATA_DIR = DATA_DIR / "search_app"
DB_PATH = APP_DATA_DIR / "semantic_search.db"


@dataclass(frozen=True)
class ChunkingConfig:
    max_lines: int = 5
    overlap_lines: int = 2
    max_chars: int = 220
    max_duration_seconds: float = 30.0


@dataclass(frozen=True)
class EmbeddingConfig:
    default_model_name: str = RUNTIME.embedding_model_dir or "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 8
    max_length: int = 2048
    query_instruction: str = (
        "Given a subtitle search query, retrieve relevant subtitle passages that answer the query."
    )


CHUNKING = ChunkingConfig()
EMBEDDING = EmbeddingConfig()
