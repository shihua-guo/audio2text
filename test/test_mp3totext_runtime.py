import importlib.util
import sys
import types
import unittest
import uuid
import io
import contextlib
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "mp3totext.py"


def load_mp3totext(fake_sherpa):
    module_name = f"mp3totext_under_test_{uuid.uuid4().hex}"

    if not isinstance(fake_sherpa, types.ModuleType):
        sherpa_module = types.ModuleType("sherpa_onnx")
        for key, value in vars(fake_sherpa).items():
            setattr(sherpa_module, key, value)
        sherpa_module.__path__ = []
        fake_sherpa = sherpa_module

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

    def test_discovers_capswriter_dir_from_model_dir_with_package_layout(self):
        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=object))

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "CapsWriter-Offline"
            model_dir = root / "models" / "Qwen3-ASR" / "Qwen3-ASR-1.7B"
            adapter_dir = root / "util" / "qwen_asr_gguf"
            adapter_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            (adapter_dir / "__init__.py").write_text("# stub\n", encoding="utf-8")

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

    def test_falls_back_to_submodule_offline_recognizer(self):
        calls = []

        class ExportedRecognizer:
            pass

        class SubmoduleRecognizer:
            @classmethod
            def from_qwen3_asr(cls, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace()

        fake_sherpa = types.SimpleNamespace(
            OfflineRecognizer=ExportedRecognizer,
            __version__="1.12.34",
            __file__="C:/fake/site-packages/sherpa_onnx/__init__.py",
        )

        sys.modules["sherpa_onnx.offline_recognizer"] = types.SimpleNamespace(
            OfflineRecognizer=SubmoduleRecognizer
        )
        module = load_mp3totext(fake_sherpa)

        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            for filename in (
                *module.QWEN3_ASR_FILENAMES,
                *module.QWEN3_TOKENIZER_FILENAMES,
            ):
                (model_dir / filename).write_text("stub\n", encoding="utf-8")

            module.AudioTranscriber(
                model_paths=module.ModelPaths(model_dir=model_dir),
                use_aligner=False,
            )

        self.assertEqual(len(calls), 1)

    def test_passes_gpu_flags_to_capswriter_adapter(self):
        calls = []

        def fake_create_asr_engine(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(engine=types.SimpleNamespace(aligner=None))

        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=object))

        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            for filename in module.QWEN3_ASR_FILENAMES:
                (model_dir / filename).write_text("stub\n", encoding="utf-8")

            module.AudioTranscriber(
                model_paths=module.ModelPaths(model_dir=model_dir),
                create_asr_engine=fake_create_asr_engine,
                use_aligner=False,
                use_dml=True,
                use_vulkan=False,
            )

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["use_dml"])
        self.assertFalse(calls[0]["vulkan_enable"])
        self.assertEqual(calls[0]["chunk_size"], module.QWEN_CHUNK_SIZE)
        self.assertEqual(calls[0]["pad_to"], int(module.QWEN_CHUNK_SIZE))

    def test_passes_chunk_settings_to_capswriter_runtime(self):
        engine_calls = []

        class FakeEngine:
            def asr(self, **kwargs):
                engine_calls.append(kwargs)
                return types.SimpleNamespace(text="stub", alignment=None)

        def fake_create_asr_engine(**kwargs):
            return types.SimpleNamespace(engine=FakeEngine())

        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=object))

        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            for filename in module.QWEN3_ASR_FILENAMES:
                (model_dir / filename).write_text("stub\n", encoding="utf-8")

            transcriber = module.AudioTranscriber(
                model_paths=module.ModelPaths(model_dir=model_dir),
                create_asr_engine=fake_create_asr_engine,
                use_aligner=False,
                qwen_chunk_size=6.0,
                qwen_memory_chunks=0,
            )

            audio_file = Path(temp_dir) / "input.wav"
            audio_file.write_bytes(b"stub")
            transcriber.read_audio = lambda _: [0.0] * (module.SAMPLE_RATE * 2)
            transcriber.transcribe_audio(str(audio_file))

        self.assertEqual(len(engine_calls), 1)
        self.assertEqual(engine_calls[0]["chunk_size_sec"], 6.0)
        self.assertEqual(engine_calls[0]["memory_chunks"], 0)

    def test_reports_directml_activation_status(self):
        def fake_create_asr_engine(**kwargs):
            return types.SimpleNamespace(
                engine=types.SimpleNamespace(
                    aligner=None,
                    encoder=types.SimpleNamespace(active_dml=True),
                )
            )

        module = load_mp3totext(types.SimpleNamespace(OfflineRecognizer=object))
        module.get_onnxruntime_providers = lambda: ["DmlExecutionProvider", "CPUExecutionProvider"]

        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            for filename in module.QWEN3_ASR_FILENAMES:
                (model_dir / filename).write_text("stub\n", encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                module.AudioTranscriber(
                    model_paths=module.ModelPaths(model_dir=model_dir),
                    create_asr_engine=fake_create_asr_engine,
                    use_aligner=False,
                    use_dml=True,
                )

        text = output.getvalue()
        self.assertIn("onnxruntime providers", text)
        self.assertIn("CapsWriter Encoder 实际启用了 DirectML", text)

    def test_gpu_flags_require_capswriter_adapter(self):
        class FakeRecognizer:
            @classmethod
            def from_qwen3_asr(cls, **kwargs):
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

            with self.assertRaisesRegex(RuntimeError, "CapsWriter 适配器"):
                module.AudioTranscriber(
                    model_paths=module.ModelPaths(model_dir=model_dir),
                    use_aligner=False,
                    use_dml=True,
                )

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
