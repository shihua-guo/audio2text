from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .config import EMBEDDING


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@dataclass
class EmbeddingModelInfo:
    model_name: str
    device: str
    max_length: int


class QwenEmbeddingBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._loaded_model_name = ""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self, model_name: str) -> EmbeddingModelInfo:
        with self._lock:
            if self._model is not None and self._loaded_model_name == model_name:
                return EmbeddingModelInfo(model_name=model_name, device=self._device, max_length=EMBEDDING.max_length)

            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.to(self._device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._loaded_model_name = model_name
            return EmbeddingModelInfo(model_name=model_name, device=self._device, max_length=EMBEDDING.max_length)

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
