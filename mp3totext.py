#!/usr/bin/env python3
"""
MP3文件转文本工具 V3
基于 sherpa-onnx 和 Qwen3 ASR 模型

特点：
1. 使用 Qwen3 ASR 模型进行离线语音识别
2. 可选使用 Qwen3 Aligner 生成更精确时间戳
3. 支持进度保存和断点续传
4. 支持批量处理音频文件
5. 纯离线运行，无需网络连接
"""

import os
import sys

# 在加载任何 GPU 相关库之前设置环境变量
os.environ["VK_ICD_FILENAMES"] = "none"
os.environ["GGML_VK_VISIBLE_DEVICES"] = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个 OpenMP 运行时共存

import time
import hashlib
import argparse
import subprocess
import shutil
import importlib
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

try:
    from portable_runtime import DATA_DIR, RUNTIME_CONFIG_PATH, load_runtime_config
except ImportError:
    DATA_DIR = SCRIPT_DIR / "data"
    RUNTIME_CONFIG_PATH = SCRIPT_DIR / "config" / "runtime_config.json"

    def load_runtime_config():
        return None

try:
    from funasr_onnx import CT_Transformer
except ImportError:
    CT_Transformer = None

# 全局配置
SAMPLE_RATE = 16000
NUM_THREADS = 4

MODEL_DIR_ENV = "AUDIO2TEXT_MODEL_DIR"
CAPS_WRITER_DIR_ENV = "AUDIO2TEXT_CAPSWRITER_DIR"
PUNC_MODEL_DIR_ENV = "AUDIO2TEXT_PUNC_MODEL_DIR"
LEGACY_CAPS_WRITER_DIR_ENV = "CAPS_WRITER_DIR"
DEFAULT_QWEN3_MODEL_SUBDIR = Path("models") / "Qwen3-ASR" / "Qwen3-ASR-1.7B"
DEFAULT_PUNC_MODEL_SUBDIR = (
    Path("models") / "Punct-CT-Transformer" / "punc_ct-transformer_cn-en"
)
QWEN3_ASR_FILENAMES = (
    "qwen3_asr_encoder_frontend.int4.onnx",
    "qwen3_asr_encoder_backend.int4.onnx",
    "qwen3_asr_llm.q4_k.gguf",
)
QWEN3_ALIGNER_FILENAMES = (
    "qwen3_aligner_encoder_frontend.int4.onnx",
    "qwen3_aligner_encoder_backend.int4.onnx",
    "qwen3_aligner_llm.q4_k.gguf",
)
QWEN3_TOKENIZER_FILENAMES = ("vocab.json", "merges.txt", "tokenizer_config.json")

# 是否启用 aligner
USE_ALIGNER = True
DEFAULT_LANGUAGE = "Chinese"
QWEN_CHUNK_SIZE = 40.0
QWEN_MEMORY_CHUNKS = 1
MAX_SUBTITLE_CHARS = 35
MAX_SUBTITLE_DURATION = 6.0
MAX_SUBTITLE_GAP = 1.0


def runtime_config_value(field_name: str) -> Optional[str]:
    runtime = load_runtime_config()
    if runtime is None:
        return None

    value = getattr(runtime, field_name, "")
    return value or None


def runtime_config_model_hint() -> str:
    if load_runtime_config() is None:
        return ""
    return f" 也可以在配置文件 {RUNTIME_CONFIG_PATH} 中设置相关模型目录。"


def get_ffmpeg_executable() -> str:
    configured = runtime_config_value("ffmpeg_path")
    if configured and Path(configured).exists():
        return configured
    return "ffmpeg"


def get_default_input_output_dirs() -> Tuple[str, str]:
    if load_runtime_config() is None:
        default_dir = r"D:\video\mp3"
        return default_dir, default_dir

    input_dir = DATA_DIR / "audio_input"
    output_dir = DATA_DIR / "audio_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(input_dir), str(output_dir)


def get_capswriter_runtime_dir() -> Path:
    runtime_dir = DATA_DIR / "capswriter_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "logs").mkdir(parents=True, exist_ok=True)
    return runtime_dir


@dataclass
class ModelPaths:
    """运行时模型路径"""

    model_dir: Optional[Path]
    punc_model_dir: Optional[Path] = None
    capswriter_dir: Optional[Path] = None

    def required_asr_files(self) -> List[Path]:
        if self.model_dir is None:
            return []
        return [self.model_dir / name for name in QWEN3_ASR_FILENAMES]

    def optional_aligner_files(self) -> List[Path]:
        if self.model_dir is None:
            return []
        return [self.model_dir / name for name in QWEN3_ALIGNER_FILENAMES]

    def tokenizer_files(self) -> List[Path]:
        if self.model_dir is None:
            return []
        return [self.model_dir / name for name in QWEN3_TOKENIZER_FILENAMES]

    @property
    def qwen3_asr_frontend(self) -> Path:
        return self.model_dir / QWEN3_ASR_FILENAMES[0]

    @property
    def qwen3_asr_backend(self) -> Path:
        return self.model_dir / QWEN3_ASR_FILENAMES[1]

    @property
    def qwen3_asr_llm(self) -> Path:
        return self.model_dir / QWEN3_ASR_FILENAMES[2]

    @property
    def qwen3_aligner_frontend(self) -> Path:
        return self.model_dir / QWEN3_ALIGNER_FILENAMES[0]

    @property
    def qwen3_aligner_backend(self) -> Path:
        return self.model_dir / QWEN3_ALIGNER_FILENAMES[1]

    @property
    def qwen3_aligner_llm(self) -> Path:
        return self.model_dir / QWEN3_ALIGNER_FILENAMES[2]


def normalize_path(path_value: Optional[str]) -> Optional[Path]:
    """展开环境变量和用户目录"""
    if not path_value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    return Path(expanded)


def configured_path(*values: Optional[str]) -> Optional[Path]:
    """返回首个已配置路径，不要求路径已存在"""
    for value in values:
        path = normalize_path(value)
        if path is not None:
            return path
    return None


def has_capswriter_adapter(capswriter_dir: Path) -> bool:
    """兼容 CapsWriter 将 qwen_asr_gguf 实现为单文件或包目录两种结构"""
    adapter_file = capswriter_dir / "util" / "qwen_asr_gguf.py"
    adapter_package = capswriter_dir / "util" / "qwen_asr_gguf" / "__init__.py"
    return adapter_file.exists() or adapter_package.exists()


def discover_capswriter_dir_from_model_dir(model_dir: Optional[Path]) -> Optional[Path]:
    """从模型目录向上查找 CapsWriter 根目录"""
    if model_dir is None:
        return None

    for candidate in model_dir.parents:
        if has_capswriter_adapter(candidate):
            return candidate

    return None


def resolve_model_dir(explicit_model_dir: Optional[str], capswriter_dir: Optional[Path]) -> Optional[Path]:
    """解析 ASR 模型目录"""
    configured = configured_path(
        explicit_model_dir,
        os.getenv(MODEL_DIR_ENV),
        runtime_config_value("asr_model_dir"),
    )
    if configured is not None:
        return configured

    candidates: List[Path] = []
    if capswriter_dir is not None:
        candidates.append(capswriter_dir / DEFAULT_QWEN3_MODEL_SUBDIR)

    candidates.extend(
        [
            Path.cwd() / DEFAULT_QWEN3_MODEL_SUBDIR,
            SCRIPT_DIR / DEFAULT_QWEN3_MODEL_SUBDIR,
            Path.cwd() / "Qwen3-ASR-1.7B",
            SCRIPT_DIR / "Qwen3-ASR-1.7B",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else None


def resolve_punc_model_dir(explicit_punc_model_dir: Optional[str], capswriter_dir: Optional[Path]) -> Optional[Path]:
    """解析标点模型目录"""
    configured = configured_path(
        explicit_punc_model_dir,
        os.getenv(PUNC_MODEL_DIR_ENV),
        runtime_config_value("punc_model_dir"),
    )
    if configured is not None:
        return configured

    candidates: List[Path] = []
    if capswriter_dir is not None:
        candidates.append(capswriter_dir / DEFAULT_PUNC_MODEL_SUBDIR)

    candidates.extend(
        [
            Path.cwd() / DEFAULT_PUNC_MODEL_SUBDIR,
            SCRIPT_DIR / DEFAULT_PUNC_MODEL_SUBDIR,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def build_model_paths(
    explicit_model_dir: Optional[str] = None,
    explicit_punc_model_dir: Optional[str] = None,
    explicit_capswriter_dir: Optional[str] = None,
) -> ModelPaths:
    """根据命令行参数和环境变量解析模型路径"""
    capswriter_dir = configured_path(
        explicit_capswriter_dir,
        os.getenv(CAPS_WRITER_DIR_ENV),
        os.getenv(LEGACY_CAPS_WRITER_DIR_ENV),
        runtime_config_value("capswriter_dir"),
    )
    model_dir = resolve_model_dir(explicit_model_dir, capswriter_dir)

    if capswriter_dir is None:
        capswriter_dir = discover_capswriter_dir_from_model_dir(model_dir)
        if model_dir is None and capswriter_dir is not None:
            model_dir = resolve_model_dir(explicit_model_dir, capswriter_dir)

    punc_model_dir = resolve_punc_model_dir(explicit_punc_model_dir, capswriter_dir)
    return ModelPaths(model_dir=model_dir, punc_model_dir=punc_model_dir, capswriter_dir=capswriter_dir)


def load_capswriter_adapter(
    capswriter_dir: Optional[Path],
) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """按需加载 CapsWriter 的 Qwen3 适配器"""
    if capswriter_dir is None:
        return None, None

    if not has_capswriter_adapter(capswriter_dir):
        return None, f"未在 {capswriter_dir} 找到 util/qwen_asr_gguf.py 或 util/qwen_asr_gguf/__init__.py"

    for path in [capswriter_dir, capswriter_dir / "internal"]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        capswriter_config = importlib.import_module("config_client")
        if hasattr(capswriter_config, "BASE_DIR"):
            capswriter_config.BASE_DIR = str(get_capswriter_runtime_dir())
    except Exception:
        pass

    try:
        from util.qwen_asr_gguf import create_asr_engine as capswriter_create_asr_engine
    except Exception as exc:
        return None, f"从 {capswriter_dir} 导入 util.qwen_asr_gguf.create_asr_engine 失败: {exc}"

    return capswriter_create_asr_engine, None


def get_sherpa_runtime_info() -> str:
    """返回当前 sherpa-onnx 运行时信息"""
    if sherpa_onnx is None:
        return "sherpa-onnx: 未安装"

    version = getattr(sherpa_onnx, "__version__", "unknown")
    module_file = getattr(sherpa_onnx, "__file__", "unknown")
    return f"sherpa-onnx version={version}, module={module_file}"


def resolve_offline_recognizer_cls():
    """兼容不同导出方式，获取 sherpa-onnx 的 OfflineRecognizer"""
    candidates = []

    top_level_cls = getattr(sherpa_onnx, "OfflineRecognizer", None) if sherpa_onnx else None
    if top_level_cls is not None:
        candidates.append(top_level_cls)

    if sherpa_onnx is not None:
        try:
            from sherpa_onnx.offline_recognizer import OfflineRecognizer as module_offline_recognizer
        except Exception:
            module_offline_recognizer = None

        if module_offline_recognizer is not None and module_offline_recognizer not in candidates:
            candidates.append(module_offline_recognizer)

    for candidate in candidates:
        if hasattr(candidate, "from_qwen3_asr") or hasattr(candidate, "from_qwen3"):
            return candidate

    return candidates[0] if candidates else None


def get_onnxruntime_providers() -> Optional[List[str]]:
    """返回 onnxruntime 当前可用 provider 列表"""
    with contextlib.suppress(Exception):
        import onnxruntime

        return list(onnxruntime.get_available_providers())

    return None

def format_srt_timestamp(seconds: float) -> str:
    """格式化为标准 SRT 时间戳"""
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def check_ffmpeg():
    """检查ffmpeg是否可用"""
    ffmpeg_executable = get_ffmpeg_executable()
    if shutil.which(ffmpeg_executable) is None and not Path(ffmpeg_executable).exists():
        try:
            import librosa
            print("警告: 未找到ffmpeg，将尝试使用librosa作为后备音频解码器")
        except ImportError:
            print("错误: 未找到ffmpeg，且未安装librosa，无法进行音频解码")
            print("Windows下载地址: https://www.gyan.dev/ffmpeg/builds/")
            sys.exit(1)


@dataclass
class Segment:
    """时间段类"""
    start: float
    duration: float = 0.0
    text: str = ""

    @property
    def end(self):
        return self.start + self.duration

    def __str__(self):
        return f"{format_srt_timestamp(self.start)} --> {format_srt_timestamp(self.end)}\n{self.text}"


class ProgressManager:
    """进度管理器"""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.processed_files: Dict[str, str] = {}

    def load_progress(self):
        self.processed_files = {}
        if not os.path.exists(self.progress_file):
            return

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        file_hash, output_path = line.split('|', 1)
                        self.processed_files[file_hash] = output_path
            print(f"已加载 {len(self.processed_files)} 个已处理文件的进度")
        except Exception as e:
            print(f"加载进度文件失败: {e}")
            self.processed_files = {}

    def save_progress(self, file_hash: str, output_path: str):
        self.processed_files[file_hash] = output_path
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                for hash_val, path_val in self.processed_files.items():
                    f.write(f"{hash_val}|{path_val}\n")
        except Exception as e:
            print(f"保存进度失败: {e}")

    def is_processed(self, file_hash: str):
        return file_hash in self.processed_files

    def get_processed_count(self):
        return len(self.processed_files)


class AudioTranscriber:
    """音频转文本器（Qwen3 ASR）"""

    def __init__(
        self,
        num_threads: int = NUM_THREADS,
        model_paths: Optional[ModelPaths] = None,
        create_asr_engine: Optional[Callable[..., Any]] = None,
        capswriter_status: Optional[str] = None,
        use_aligner: bool = USE_ALIGNER,
        use_dml: bool = False,
        use_vulkan: bool = False,
    ):
        self.num_threads = num_threads
        self.model_paths = model_paths or build_model_paths()
        self.create_asr_engine = create_asr_engine
        self.capswriter_status = capswriter_status
        self.use_aligner = use_aligner
        self.use_dml = use_dml
        self.use_vulkan = use_vulkan
        self.asr_model = None
        self.aligner = None
        self.punc_model = None
        self.load_models()

    def _check_required_model_files(self):
        if self.model_paths.model_dir is None:
            raise FileNotFoundError(
                f"未找到 Qwen3 ASR 模型目录。请使用 --model-dir 或环境变量 {MODEL_DIR_ENV} 指向包含 "
                f"{', '.join(QWEN3_ASR_FILENAMES)} 的目录。{runtime_config_model_hint()}"
            )

        missing_files = [path for path in self.model_paths.required_asr_files() if not path.exists()]
        if missing_files:
            missing_text = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                "未找到完整的 Qwen3 ASR 模型文件，请确认 --model-dir 指向正确目录。缺失文件:\n"
                f"{missing_text}{runtime_config_model_hint()}"
            )

    def _load_native_qwen3_asr(self):
        if sherpa_onnx is None:
            raise RuntimeError(
                "未安装 sherpa-onnx，且 CapsWriter 适配器不可用。请安装依赖，或者通过 "
                f"--capswriter-dir / 环境变量 {CAPS_WRITER_DIR_ENV} 指向 CapsWriter-Offline。"
            )

        recognizer_cls = resolve_offline_recognizer_cls()
        if recognizer_cls is None:
            raise RuntimeError(
                "当前 sherpa-onnx 安装不包含 OfflineRecognizer。"
                f"\n{get_sherpa_runtime_info()}"
            )

        if hasattr(recognizer_cls, "from_qwen3_asr"):
            missing_tokenizers = [path for path in self.model_paths.tokenizer_files() if not path.exists()]
            if missing_tokenizers:
                missing_text = "\n".join(str(path) for path in missing_tokenizers)
                raise RuntimeError(
                    "当前未能加载 CapsWriter 适配器，而 sherpa-onnx 原生 Qwen3-ASR API 还需要 tokenizer 文件。"
                    f"\n缺失文件:\n{missing_text}\n"
                    f"请安装 CapsWriter-Offline 并设置 --capswriter-dir，或补齐上述 tokenizer 文件。"
                )

            self.asr_model = recognizer_cls.from_qwen3_asr(
                conv_frontend=str(self.model_paths.qwen3_asr_frontend),
                encoder=str(self.model_paths.qwen3_asr_backend),
                decoder=str(self.model_paths.qwen3_asr_llm),
                tokenizer=str(self.model_paths.model_dir),
                num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE,
                debug=False,
            )
            print("Qwen3 ASR 模型加载成功 (使用 sherpa-onnx 原生 API)")
            return

        if hasattr(recognizer_cls, "from_qwen3"):
            self.asr_model = recognizer_cls.from_qwen3(
                encoder_frontend=str(self.model_paths.qwen3_asr_frontend),
                encoder_backend=str(self.model_paths.qwen3_asr_backend),
                decoder=str(self.model_paths.qwen3_asr_llm),
                num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE,
                debug=False,
            )
            print("Qwen3 ASR 模型加载成功 (使用 sherpa-onnx 旧版原生 API)")
            return

        raise RuntimeError(
            "当前安装的 sherpa-onnx 不包含 Qwen3-ASR Python API。"
            " PyPI 版 1.12.34 提供的是 OfflineRecognizer.from_qwen3_asr，不是 from_qwen3。"
            f"\n{get_sherpa_runtime_info()}"
        )

    def _report_capswriter_acceleration_status(self):
        engine = getattr(self.asr_model, "engine", None)
        encoder = getattr(engine, "encoder", None)
        active_dml = getattr(encoder, "active_dml", None)

        if self.use_dml:
            providers = get_onnxruntime_providers()
            if providers:
                print(f"onnxruntime providers: {providers}")

            if active_dml is True:
                print("CapsWriter Encoder 实际启用了 DirectML；GGUF Decoder 仍可能主要占用 CPU。")
            elif active_dml is False:
                print("警告: 已请求 DirectML，但 CapsWriter Encoder 未启用 DirectML。请确认已安装 onnxruntime-directml。")
            else:
                print("警告: 已请求 DirectML，但无法从 CapsWriter 适配器确认其是否实际生效。")

        if self.use_vulkan:
            print("提示: Vulkan 目前只是请求 CapsWriter 的 GGUF/Vulkan 路径，实际转录过程中 CPU 仍可能是主要瓶颈。")

    def load_models(self):
        """加载模型"""
        print("正在加载模型...")
        start_time = time.time()

        self._check_required_model_files()

        print("加载 Qwen3 ASR 模型...")
        if self.model_paths.capswriter_dir is not None:
            print(f"CapsWriter 目录: {self.model_paths.capswriter_dir}")
        if self.capswriter_status:
            print(f"CapsWriter 适配器不可用: {self.capswriter_status}")
        if self.use_dml:
            print("Qwen3 ASR 加速后端: DirectML")
        elif self.use_vulkan:
            print("Qwen3 ASR 加速后端: Vulkan")
        else:
            print("Qwen3 ASR 加速后端: CPU")

        try:
            if self.create_asr_engine:
                self.asr_model = self.create_asr_engine(
                    model_dir=str(self.model_paths.model_dir),
                    encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
                    encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
                    llm_fn="qwen3_asr_llm.q4_k.gguf",
                    n_ctx=4096,
                    chunk_size=QWEN_CHUNK_SIZE,
                    pad_to=int(QWEN_CHUNK_SIZE),
                    use_dml=self.use_dml,
                    vulkan_enable=self.use_vulkan,
                    verbose=False,
                    enable_aligner=self.use_aligner,
                )
                print("Qwen3 ASR 模型加载成功 (使用 CapsWriter 适配器，支持长音频分块)")
                self._report_capswriter_acceleration_status()
            else:
                if self.use_dml or self.use_vulkan:
                    raise RuntimeError(
                        "当前 GPU 开关仅对 CapsWriter 适配器路径生效，但脚本没有加载到 CapsWriter 适配器。"
                        "\n请确认 --capswriter-dir 指向完整的 CapsWriter-Offline 根目录。"
                    )
                self._load_native_qwen3_asr()
        except Exception as e:
            print(f"加载 Qwen3 ASR 模型失败: {e}")
            raise

        # 加载 Qwen3 aligner（可选）
        if self.create_asr_engine:
            if self.use_aligner and getattr(getattr(self.asr_model, "engine", None), "aligner", None):
                print("Qwen3 Aligner 已通过 CapsWriter 适配器启用")
            elif self.use_aligner:
                print("警告: CapsWriter 适配器未启用对齐器，SRT 将使用粗略时间戳")
        elif self.use_aligner:
            aligner_cls = getattr(sherpa_onnx, "Aligner", None) if sherpa_onnx else None
            if aligner_cls is None or not hasattr(aligner_cls, "from_qwen3"):
                print("警告: 当前 sherpa-onnx Python 包未提供 Qwen3 Aligner 接口，SRT 将使用粗略时间戳")
            elif all(path.exists() for path in self.model_paths.optional_aligner_files()):
                try:
                    print("加载 Qwen3 Aligner 模型...")
                    self.aligner = aligner_cls.from_qwen3(
                        encoder_frontend=str(self.model_paths.qwen3_aligner_frontend),
                        encoder_backend=str(self.model_paths.qwen3_aligner_backend),
                        decoder=str(self.model_paths.qwen3_aligner_llm),
                        num_threads=self.num_threads,
                        sample_rate=SAMPLE_RATE,
                        debug=False,
                    )
                    print("Qwen3 Aligner 模型加载成功")
                except Exception as e:
                    print(f"Qwen3 Aligner 加载失败，将退化为无精确时间戳模式: {e}")
                    self.aligner = None
            else:
                print("警告: 未找到完整的 Qwen3 Aligner 模型，SRT 将使用粗略时间戳")

        # 加载标点模型
        punc_model_path = (
            self.model_paths.punc_model_dir / "model_quant.onnx"
            if self.model_paths.punc_model_dir is not None
            else None
        )
        if punc_model_path is not None and punc_model_path.exists():
            try:
                print(f"加载标点模型: {punc_model_path}")
                import jieba
                import logging
                jieba.setLogLevel(logging.INFO)

                if CT_Transformer is None:
                    raise RuntimeError("未安装 funasr-onnx")

                self.punc_model = CT_Transformer(str(self.model_paths.punc_model_dir), quantize=True)
                print("标点模型加载成功")
            except Exception as e:
                print(f"标点模型加载失败: {e}")
                print("警告: 标点模型不可用，将跳过标点符号添加")
                self.punc_model = None
        else:
            print("警告: 未找到标点模型，不添加标点符号")

        load_time = time.time() - start_time
        print(f"模型加载完成 (耗时: {load_time:.2f}秒)")

    def read_audio(self, audio_path: str) -> np.ndarray:
        """读取音频并转为16k单声道float32"""
        try:
            import librosa
            samples, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return samples.astype(np.float32)
        except ImportError:
            # Fallback to ffmpeg if librosa is not available
            ffmpeg_cmd = [
                get_ffmpeg_executable(),
                "-i", audio_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-"
            ]

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            data = process.stdout.read()
            process.wait()

            if not data:
                raise RuntimeError(f"ffmpeg 解码失败: {audio_path}")

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        except Exception as e:
            raise RuntimeError(f"音频读取失败 ({audio_path}): {e}")

    def split_text_to_segments(self, text: str, total_duration: float) -> List[Segment]:
        """
        如果没有 aligner，则按标点/长度粗略切句，并均分时间
        """
        if not text.strip():
            return []

        separators = "。！？；.!?;\n"
        sentences = []
        current = ""

        for ch in text:
            current += ch
            if ch in separators:
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        if not sentences:
            sentences = [text.strip()]

        n = len(sentences)
        seg_duration = total_duration / max(n, 1)

        segments = []
        for i, sentence in enumerate(sentences):
            start = i * seg_duration
            duration = seg_duration if i < n - 1 else max(total_duration - start, 0.1)
            segments.append(Segment(start=start, duration=duration, text=sentence))

        return segments

    def build_segments_from_alignment(self, items: List[object]) -> List[Segment]:
        """把逐词/逐句对齐结果合并为更适合字幕显示的片段"""
        if not items:
            return []

        segments = []
        current_text = ""
        current_start = None
        current_end = None
        strong_breaks = "。！？!?；;"

        def flush_segment():
            nonlocal current_text, current_start, current_end
            if current_start is None or not current_text.strip():
                current_text = ""
                current_start = None
                current_end = None
                return

            end_time = max(current_end if current_end is not None else current_start, current_start + 0.1)
            segment_text = self.add_punctuation(current_text.strip())
            segments.append(
                Segment(
                    start=float(current_start),
                    duration=float(end_time - current_start),
                    text=segment_text.strip(),
                )
            )
            current_text = ""
            current_start = None
            current_end = None

        for item in items:
            item_text = getattr(item, "text", "")
            if item_text is None:
                continue

            start = float(getattr(item, "start_time", 0.0))
            end = float(getattr(item, "end_time", start))
            if end < start:
                end = start

            if current_start is None:
                current_start = start
                current_end = end

            gap = start - (current_end if current_end is not None else start)
            projected_text = f"{current_text}{item_text}"
            should_flush = (
                bool(current_text.strip())
                and (
                    gap > MAX_SUBTITLE_GAP
                    or len(projected_text.strip()) > MAX_SUBTITLE_CHARS
                    or (end - current_start) > MAX_SUBTITLE_DURATION
                )
            )

            if should_flush:
                flush_segment()
                current_start = start
                current_end = end
                current_text = item_text
            else:
                current_text = projected_text
                current_end = max(current_end if current_end is not None else end, end)

            if any(ch in item_text for ch in strong_breaks):
                flush_segment()

        flush_segment()
        return segments

    def transcribe_audio(self, audio_path: str) -> Tuple[List[Segment], float]:
        """
        转录音频文件
        返回:
            List[Segment]: 时间段列表
            float: 处理耗时(秒)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        print(f"开始转录音频: {os.path.basename(audio_path)}")
        start_time = time.time()

        samples = self.read_audio(audio_path)
        total_duration = len(samples) / SAMPLE_RATE

        text = ""
        segments = []

        if self.create_asr_engine and hasattr(self.asr_model, "engine"):
            print("正在使用 Qwen 分块识别长音频...")
            result = self.asr_model.engine.asr(
                audio=samples,
                context="",
                language=DEFAULT_LANGUAGE,
                chunk_size_sec=QWEN_CHUNK_SIZE,
                memory_chunks=QWEN_MEMORY_CHUNKS,
            )
            text = result.text

            alignment = getattr(result, "alignment", None)
            if alignment and getattr(alignment, "items", None):
                segments = self.build_segments_from_alignment(alignment.items)
        else:
            # 原生 sherpa-onnx 路径
            stream = self.asr_model.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)

            print("正在解码识别结果...")
            self.asr_model.decode_stream(stream)

            # 兼容不同版本返回字段
            result = stream.result

            if hasattr(result, "text"):
                text = result.text
            elif hasattr(result, "tokens"):
                text = "".join(result.tokens)
            else:
                text = str(result)

        text = text.strip()
        text_with_punc = self.add_punctuation(text)

        # 如果 aligner 可用，尝试生成更准确时间戳
        if segments:
            segments = [seg for seg in segments if seg.text]
        elif self.aligner and text_with_punc:
            try:
                # 注意：不同版本的 aligner API 可能不同
                # 这里只写通用思路，如果你本地版本方法名不同，需要据此调整
                align_result = self.aligner.align(
                    samples=samples,
                    sample_rate=SAMPLE_RATE,
                    text=text_with_punc
                )

                # 假设 align_result.segments 存在
                if hasattr(align_result, "segments"):
                    for seg in align_result.segments:
                        segments.append(
                            Segment(
                                start=float(seg.start),
                                duration=float(seg.end - seg.start),
                                text=seg.text
                            )
                        )
                else:
                    segments = self.split_text_to_segments(text_with_punc, total_duration)

            except Exception as e:
                print(f"对齐失败，改用粗略时间戳: {e}")
                segments = self.split_text_to_segments(text_with_punc, total_duration)
        else:
            segments = self.split_text_to_segments(text_with_punc, total_duration)

        processing_time = time.time() - start_time
        return segments, processing_time

    def add_punctuation(self, text: str) -> str:
        """添加标点符号"""
        if not text:
            return text

        if not self.punc_model:
            return text

        try:
            result = self.punc_model(text)
            return result[0] if result else text
        except Exception as e:
            print(f"添加标点失败: {e}")
            return text


class MP3ToTextConverter:
    """音频转文本转换器"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        num_threads: int = NUM_THREADS,
        model_paths: Optional[ModelPaths] = None,
        create_asr_engine: Optional[Callable[..., Any]] = None,
        capswriter_status: Optional[str] = None,
        use_aligner: bool = USE_ALIGNER,
        use_dml: bool = False,
        use_vulkan: bool = False,
    ):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.transcriber = AudioTranscriber(
            num_threads=num_threads,
            model_paths=model_paths,
            create_asr_engine=create_asr_engine,
            capswriter_status=capswriter_status,
            use_aligner=use_aligner,
            use_dml=use_dml,
            use_vulkan=use_vulkan,
        )

        progress_file = os.path.join(output_dir, "processed_qwen3_files.txt")
        self.progress_manager = ProgressManager(progress_file)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}

    def get_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_audio_files(self) -> List[Path]:
        audio_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_extensions:
                    audio_files.append(file_path)

        return sorted(audio_files)

    def save_results(self, segments: List[Segment], audio_path: Path, file_hash: str):
        relative_path = audio_path.relative_to(self.input_dir)
        output_file = (self.output_dir / relative_path).with_suffix(".txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(seg.text)
                if not seg.text.endswith("\n"):
                    f.write("\n")

        srt_file = output_file.with_suffix(".srt")
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                print(i, file=f)
                print(seg, file=f)
                print("", file=f)

        self.progress_manager.save_progress(file_hash, str(output_file))
        return str(output_file), str(srt_file)

    def convert(self, resume: bool = True):
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print("-" * 50)

        if resume:
            self.progress_manager.load_progress()
            processed_count = self.progress_manager.get_processed_count()
            print(f"找到 {processed_count} 个已处理的文件")
        else:
            processed_count = 0

        audio_files = self.get_audio_files()
        if not audio_files:
            print(f"在目录 {self.input_dir} 中未找到支持的音频文件")
            return

        total_files = len(audio_files)
        print(f"找到 {total_files} 个音频文件")

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for i, audio_file in enumerate(audio_files, 1):
            file_name = audio_file.name
            file_hash = self.get_file_hash(str(audio_file))

            if self.progress_manager.is_processed(file_hash):
                print(f"[{i}/{total_files}] 跳过已处理的文件: {file_name}")
                skipped_count += 1
                continue

            print(f"[{i}/{total_files}] 正在处理: {file_name}")

            try:
                segments, processing_time = self.transcriber.transcribe_audio(str(audio_file))
                txt_file, srt_file = self.save_results(segments, audio_file, file_hash)

                success_count += 1
                print(f"转换成功: {file_name} (耗时: {processing_time:.2f}秒)")
                print(f"文本文件: {txt_file}")
                print(f"SRT字幕: {srt_file}")

            except Exception as e:
                failed_count += 1
                print(f"转换失败: {file_name}")
                print(f"错误信息: {e}")

            print("-" * 50)

        print("\n转换完成!")
        print(f"总文件数: {total_files}")
        print(f"跳过文件: {skipped_count}")
        print(f"成功转换: {success_count}")
        print(f"转换失败: {failed_count}")

        if total_files > 0:
            base = total_files - skipped_count
            success_rate = (success_count / base) * 100 if base > 0 else 0
            print(f"成功率: {success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="MP3转文本工具 V3 - Qwen3 ASR")
    default_input_dir, default_output_dir = get_default_input_output_dirs()
    parser.add_argument("--input", default=default_input_dir, help="输入音频文件目录")
    parser.add_argument("--output", default=default_output_dir, help="输出目录")
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help="线程数")
    parser.add_argument("--new", action="store_true", help="重新处理所有文件，不使用断点续传")
    parser.add_argument("--model-dir", help="Qwen3 ASR 模型目录")
    parser.add_argument("--capswriter-dir", help="CapsWriter-Offline 根目录")
    parser.add_argument("--punc-model-dir", help="标点模型目录")
    parser.add_argument("--no-aligner", action="store_true", help="禁用精确时间戳对齐")
    parser.add_argument("--dml", action="store_true", help="使用 DirectML 加速 Qwen3 ASR（需 CapsWriter 适配器）")
    parser.add_argument("--vulkan", action="store_true", help="使用 Vulkan 加速 Qwen3 ASR（需 CapsWriter 适配器）")

    args = parser.parse_args()

    if args.dml and args.vulkan:
        parser.error("--dml 和 --vulkan 不能同时使用")

    check_ffmpeg()

    model_paths = build_model_paths(
        explicit_model_dir=args.model_dir,
        explicit_punc_model_dir=args.punc_model_dir,
        explicit_capswriter_dir=args.capswriter_dir,
    )
    create_asr_engine, capswriter_status = load_capswriter_adapter(model_paths.capswriter_dir)

    try:
        converter = MP3ToTextConverter(
            args.input,
            args.output,
            args.threads,
            model_paths=model_paths,
            create_asr_engine=create_asr_engine,
            capswriter_status=capswriter_status,
            use_aligner=not args.no_aligner,
            use_dml=args.dml,
            use_vulkan=args.vulkan,
        )
    except Exception as e:
        print(f"初始化转换器失败: {e}")
        return

    try:
        converter.convert(resume=not args.new)
    except KeyboardInterrupt:
        print("\n用户中断，程序退出。下次运行将自动从断点处继续。")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
