import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import portable_runtime

try:
    import numpy as np
    from srt_search_app.embeddings import OpenAICompatibleEmbeddingBackend
except ModuleNotFoundError:
    np = None
    OpenAICompatibleEmbeddingBackend = None


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class RuntimeConfigEmbeddingApiTests(unittest.TestCase):
    def test_api_mode_keeps_embedding_model_as_raw_name(self):
        with TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir)
            config_dir = app_root / "config"
            data_dir = app_root / "data"
            models_dir = app_root / "models"
            config_dir.mkdir(parents=True)
            payload = {
                "embedding_model_dir": "text-embedding-3-small",
                "embedding_api_base": "https://api.example.com/v1",
                "embedding_api_key": "",
            }
            config_path = config_dir / "runtime_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with (
                patch.object(portable_runtime, "APP_ROOT", app_root),
                patch.object(portable_runtime, "CONFIG_DIR", config_dir),
                patch.object(portable_runtime, "DATA_DIR", data_dir),
                patch.object(portable_runtime, "MODELS_DIR", models_dir),
                patch.object(portable_runtime, "RUNTIME_CONFIG_PATH", config_path),
                patch.dict("os.environ", {"AUDIO2TEXT_EMBEDDING_API_KEY": "env-key"}, clear=False),
            ):
                portable_runtime.load_runtime_config.cache_clear()
                runtime = portable_runtime.load_runtime_config()

            self.assertEqual(runtime.embedding_model_dir, "text-embedding-3-small")
            self.assertEqual(runtime.embedding_api_base, "https://api.example.com/v1")
            self.assertEqual(runtime.embedding_api_key, "env-key")
            portable_runtime.load_runtime_config.cache_clear()


@unittest.skipIf(np is None or OpenAICompatibleEmbeddingBackend is None, "numpy 未安装，跳过 embedding backend 用例")
class OpenAICompatibleEmbeddingBackendTests(unittest.TestCase):
    def test_encode_documents_batches_requests_and_normalizes_vectors(self):
        calls = []

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            calls.append(
                {
                    "url": req.full_url,
                    "headers": {key.lower(): value for key, value in req.header_items()},
                    "body": body,
                    "timeout": timeout,
                }
            )
            vectors = [
                {"index": index, "embedding": [float(index + 3), 4.0]}
                for index, _ in enumerate(body["input"])
            ]
            return FakeResponse({"data": vectors})

        backend = OpenAICompatibleEmbeddingBackend(
            base_url="https://api.example.com/v1",
            api_key="secret-token",
        )

        with patch("srt_search_app.embeddings.request.urlopen", side_effect=fake_urlopen):
            matrix = backend.encode_documents(
                ["alpha", "beta", "gamma"],
                model_name="text-embedding-3-small",
                batch_size=2,
            )

        self.assertEqual(matrix.shape, (3, 2))
        self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), np.ones(3)))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["url"], "https://api.example.com/v1/embeddings")
        self.assertEqual(calls[0]["body"]["model"], "text-embedding-3-small")
        self.assertEqual(calls[0]["body"]["input"], ["alpha", "beta"])
        self.assertEqual(calls[1]["body"]["input"], ["gamma"])
        self.assertEqual(calls[0]["headers"]["authorization"], "Bearer secret-token")

    def test_encode_query_wraps_instruction_prefix(self):
        captured = {}

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            captured["body"] = body
            return FakeResponse({"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

        backend = OpenAICompatibleEmbeddingBackend(
            base_url="https://api.example.com/v1",
            api_key="",
        )

        with patch("srt_search_app.embeddings.request.urlopen", side_effect=fake_urlopen):
            vector = backend.encode_query("老师开始讲考试规则", model_name="text-embedding-3-small")

        self.assertEqual(vector.shape, (2,))
        self.assertTrue(np.allclose(np.linalg.norm(vector), 1.0))
        self.assertEqual(captured["body"]["model"], "text-embedding-3-small")
        self.assertTrue(captured["body"]["input"][0].startswith("Instruct: "))
        self.assertIn("Query:老师开始讲考试规则", captured["body"]["input"][0])


if __name__ == "__main__":
    unittest.main()
