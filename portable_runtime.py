from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def get_bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent


APP_ROOT = get_bundle_root()
RESOURCE_ROOT = get_resource_root()
CONFIG_DIR = APP_ROOT / "config"
DATA_DIR = APP_ROOT / "data"
MODELS_DIR = APP_ROOT / "models"
RUNTIME_CONFIG_PATH = CONFIG_DIR / "runtime_config.json"


def _resolve_path(value: str, base: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


@dataclass(frozen=True)
class RuntimeConfig:
    capswriter_dir: str
    asr_model_dir: str
    punc_model_dir: str
    embedding_model_dir: str
    ffmpeg_path: str
    search_port: int


def _default_config_payload() -> dict:
    return {
        "capswriter_dir": "",
        "asr_model_dir": "models/Qwen3-ASR/Qwen3-ASR-1.7B",
        "punc_model_dir": "models/Punct-CT-Transformer/punc_ct-transformer_cn-en",
        "embedding_model_dir": "models/embeddings/Qwen3-Embedding-0.6B",
        "ffmpeg_path": "",
        "search_port": 8000,
    }


def ensure_runtime_layout() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNTIME_CONFIG_PATH.exists():
        RUNTIME_CONFIG_PATH.write_text(
            json.dumps(_default_config_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


@lru_cache(maxsize=1)
def load_runtime_config() -> RuntimeConfig:
    ensure_runtime_layout()
    payload = _default_config_payload()
    try:
        loaded = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload.update(loaded)
    except Exception:
        pass

    return RuntimeConfig(
        capswriter_dir=_resolve_path(str(payload.get("capswriter_dir", "")).strip(), APP_ROOT),
        asr_model_dir=_resolve_path(str(payload.get("asr_model_dir", "")).strip(), APP_ROOT),
        punc_model_dir=_resolve_path(str(payload.get("punc_model_dir", "")).strip(), APP_ROOT),
        embedding_model_dir=_resolve_path(str(payload.get("embedding_model_dir", "")).strip(), APP_ROOT),
        ffmpeg_path=_resolve_path(str(payload.get("ffmpeg_path", "")).strip(), APP_ROOT),
        search_port=int(payload.get("search_port", 8000) or 8000),
    )


def reload_runtime_config() -> RuntimeConfig:
    load_runtime_config.cache_clear()
    return load_runtime_config()
