from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable
from urllib import error, request

import numpy as np

from .config import EMBEDDING

if TYPE_CHECKING:
    import torch


def _last_token_pool(
    last_hidden_states: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.float32, copy=False)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        if norm > 0:
            array = array / norm
        return array.astype(np.float32, copy=False)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32, copy=False)


@dataclass
class EmbeddingModelInfo:
    model_name: str
    device: str
    max_length: int


class LocalEmbeddingBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._loaded_model_name = ""
        self._device: str | None = None

    def _get_device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self, model_name: str) -> EmbeddingModelInfo:
        with self._lock:
            device = self._get_device()
            if self._model is not None and self._loaded_model_name == model_name:
                return EmbeddingModelInfo(model_name=model_name, device=device, max_length=EMBEDDING.max_length)

            from transformers import AutoModel, AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            except Exception as exc:
                raise RuntimeError(
                    f"无法加载 embedding 模型: {model_name}。"
                    "请确认本地模型路径可用，或在 runtime_config.json 配置 embedding_api_base 后改为 OpenAI 兼容接口。"
                ) from exc
            model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._loaded_model_name = model_name
            return EmbeddingModelInfo(model_name=model_name, device=device, max_length=EMBEDDING.max_length)

    def encode_documents(
        self,
        texts: Iterable[str],
        model_name: str,
        batch_size: int = EMBEDDING.batch_size,
    ) -> np.ndarray:
        return self._encode(list(texts), model_name=model_name, batch_size=batch_size)

    def encode_query(
        self,
        query: str,
        model_name: str,
    ) -> np.ndarray:
        query_text = f"Instruct: {EMBEDDING.query_instruction}\nQuery:{query}"
        vectors = self._encode([query_text], model_name=model_name, batch_size=1)
        return vectors[0]

    def _encode(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        import torch
        import torch.nn.functional as F

        info = self._load(model_name)
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None
        assert model is not None

        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                batch_dict = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=info.max_length,
                    return_tensors="pt",
                )
                batch_dict = {key: value.to(info.device) for key, value in batch_dict.items()}
                model_output = model(**batch_dict)
                embeddings = _last_token_pool(model_output.last_hidden_state, batch_dict["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)


class OpenAICompatibleEmbeddingBackend:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds

    def encode_documents(
        self,
        texts: Iterable[str],
        model_name: str,
        batch_size: int = EMBEDDING.batch_size,
    ) -> np.ndarray:
        return self._encode(list(texts), model_name=model_name, batch_size=batch_size)

    def encode_query(
        self,
        query: str,
        model_name: str,
    ) -> np.ndarray:
        query_text = f"Instruct: {EMBEDDING.query_instruction}\nQuery:{query}"
        vectors = self._encode([query_text], model_name=model_name, batch_size=1)
        return vectors[0]

    def _endpoint_url(self) -> str:
        if self._base_url.endswith("/embeddings"):
            return self._base_url
        return f"{self._base_url}/embeddings"

    def _encode(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            outputs.append(self._request_embeddings(batch, model_name=model_name))

        return np.concatenate(outputs, axis=0)

    def _request_embeddings(self, texts: list[str], model_name: str) -> np.ndarray:
        payload = json.dumps(
            {
                "model": model_name,
                "input": texts,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = request.Request(
            self._endpoint_url(),
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"embedding API 调用失败: HTTP {exc.code}。"
                f"响应内容: {body[:300] or '(空响应)'}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"embedding API 连接失败: {exc.reason}") from exc

        try:
            payload_json = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("embedding API 返回了无法解析的 JSON 响应") from exc

        items = payload_json.get("data")
        if not isinstance(items, list):
            raise RuntimeError("embedding API 响应缺少 data 字段")
        if len(items) != len(texts):
            raise RuntimeError(
                f"embedding API 返回数量不匹配: 期望 {len(texts)} 条，实际 {len(items)} 条"
            )

        vectors: list[np.ndarray] = []
        for item in sorted(items, key=lambda value: int(value.get("index", 0))):
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("embedding API 响应格式不正确，缺少 embedding 数组")
            vectors.append(np.asarray(embedding, dtype=np.float32))

        return _normalize_rows(np.vstack(vectors))


class EmbeddingBackend:
    def __init__(self) -> None:
        if EMBEDDING.use_api:
            self._backend = OpenAICompatibleEmbeddingBackend(
                base_url=EMBEDDING.api_base_url,
                api_key=EMBEDDING.api_key,
            )
        else:
            self._backend = LocalEmbeddingBackend()

    def encode_documents(
        self,
        texts: Iterable[str],
        model_name: str,
        batch_size: int = EMBEDDING.batch_size,
    ) -> np.ndarray:
        return self._backend.encode_documents(texts, model_name=model_name, batch_size=batch_size)

    def encode_query(self, query: str, model_name: str) -> np.ndarray:
        return self._backend.encode_query(query, model_name=model_name)


QwenEmbeddingBackend = EmbeddingBackend
