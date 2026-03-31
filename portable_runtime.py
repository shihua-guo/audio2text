from __future__ import annotations

import json
import os
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
    embedding_api_base: str
    embedding_api_key: str
    ffmpeg_path: str
    search_port: int


def _default_config_payload() -> dict:
    return {
        "capswriter_dir": "",
        "asr_model_dir": "models/Qwen3-ASR/Qwen3-ASR-1.7B",
        "punc_model_dir": "models/Punct-CT-Transformer/punc_ct-transformer_cn-en",
        "embedding_model_dir": "models/embeddings/Qwen3-Embedding-0.6B",
        "embedding_api_base": "",
        "embedding_api_key": "",
        "ffmpeg_path": "",
        "search_port": 8000,
    }


def _text_value(payload: dict, key: str, *env_names: str) -> str:
    value = str(payload.get(key, "")).strip()
    if value:
        return value
    for env_name in env_names:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
    return ""


def _resolve_embedding_model(value: str, base: Path, api_base: str) -> str:
    if not value:
        return ""
    if api_base:
        return value
    return _resolve_path(value, base)


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

    embedding_api_base = _text_value(
        payload,
        "embedding_api_base",
        "AUDIO2TEXT_EMBEDDING_API_BASE",
        "OPENAI_BASE_URL",
    )
    embedding_api_key = _text_value(
        payload,
        "embedding_api_key",
        "AUDIO2TEXT_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    )
    embedding_model_dir = _text_value(payload, "embedding_model_dir")

    return RuntimeConfig(
        capswriter_dir=_resolve_path(_text_value(payload, "capswriter_dir"), APP_ROOT),
        asr_model_dir=_resolve_path(_text_value(payload, "asr_model_dir"), APP_ROOT),
        punc_model_dir=_resolve_path(_text_value(payload, "punc_model_dir"), APP_ROOT),
        embedding_model_dir=_resolve_embedding_model(embedding_model_dir, APP_ROOT, embedding_api_base),
        embedding_api_base=embedding_api_base,
        embedding_api_key=embedding_api_key,
        ffmpeg_path=_resolve_path(_text_value(payload, "ffmpeg_path"), APP_ROOT),
        search_port=int(payload.get("search_port", 8000) or 8000),
    )


def reload_runtime_config() -> RuntimeConfig:
    load_runtime_config.cache_clear()
    return load_runtime_config()
