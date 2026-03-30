import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "mp3totext.py"


def load_mp3totext(fake_sherpa):
    module_name = f"mp3totext_under_test_{uuid.uuid4().hex}"

    fake_numpy = types.SimpleNamespace(
        ndarray=object,
        float32="float32",
        int16="int16",
        frombuffer=lambda *args, **kwargs: None,
    )
    fake_funasr = types.SimpleNamespace(CT_Transformer=lambda *args, **kwargs: None)

    sys.modules["numpy"] = fake_numpy
    sys.modules["sherpa_onnx"] = fake_sherpa
    sys.modules["funasr_onnx"] = fake_funasr

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Mp3ToTextRuntimeTests(unittest.TestCase):
    def test_discovers_capswriter_dir_from_model_dir(self):
        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=object))

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "CapsWriter-Offline"
            model_dir = root / "models" / "Qwen3-ASR" / "Qwen3-ASR-1.7B"
            (root / "util").mkdir(parents=True)
            model_dir.mkdir(parents=True)
            (root / "util" / "qwen_asr_gguf.py").write_text("# stub\n", encoding="utf-8")

            self.assertEqual(module.discover_capswriter_dir_from_model_dir(model_dir), root)

    def test_uses_from_qwen3_asr_when_native_api_is_available(self):
        calls = []

        class FakeRecognizer:
            @classmethod
            def from_qwen3_asr(cls, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace()

        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=FakeRecognizer))

        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            for filename in (
                *module.QWEN3_ASR_FILENAMES,
                *module.QWEN3_TOKENIZER_FILENAMES,
            ):
                (model_dir / filename).write_text("stub\n", encoding="utf-8")

            module.AudioTranscriber(
                num_threads=3,
                model_paths=module.ModelPaths(model_dir=model_dir),
                use_aligner=False,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["conv_frontend"], str(model_dir / module.QWEN3_ASR_FILENAMES[0]))
        self.assertEqual(calls[0]["encoder"], str(model_dir / module.QWEN3_ASR_FILENAMES[1]))
        self.assertEqual(calls[0]["decoder"], str(model_dir / module.QWEN3_ASR_FILENAMES[2]))
        self.assertEqual(calls[0]["tokenizer"], str(model_dir))
        self.assertEqual(calls[0]["num_threads"], 3)

    def test_missing_tokenizer_files_raises_clear_error(self):
        class FakeRecognizer:
            @classmethod
            def from_qwen3_asr(cls, **kwargs):
                return types.SimpleNamespace()

        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=FakeRecognizer))

        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            for filename in module.QWEN3_ASR_FILENAMES:
                (model_dir / filename).write_text("stub\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "tokenizer"):
                module.AudioTranscriber(
                    model_paths=module.ModelPaths(model_dir=model_dir),
                    use_aligner=False,
                )


if __name__ == "__main__":
    unittest.main()
