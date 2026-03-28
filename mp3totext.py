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
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import sherpa_onnx

# --- Add CapsWriter-Offline to path ---
CAPS_WRITER_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline"
INTERNAL_DIR = os.path.join(CAPS_WRITER_DIR, "internal")
for path in [CAPS_WRITER_DIR, INTERNAL_DIR]:
    if path not in sys.path:
        sys.path.append(path)

try:
    from util.qwen_asr_gguf import create_asr_engine
except ImportError:
    print("警告: 无法从 CapsWriter-Offline 加载 create_asr_engine")
    create_asr_engine = None

from funasr_onnx import CT_Transformer

# 全局配置
SAMPLE_RATE = 16000
NUM_THREADS = 4

# 模型路径配置
MODEL_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline\models\Qwen3-ASR\Qwen3-ASR-1.7B"

QWEN3_ASR_FRONTEND = os.path.join(MODEL_DIR, "qwen3_asr_encoder_frontend.int4.onnx")
QWEN3_ASR_BACKEND = os.path.join(MODEL_DIR, "qwen3_asr_encoder_backend.int4.onnx")
QWEN3_ASR_LLM = os.path.join(MODEL_DIR, "qwen3_asr_llm.q4_k.gguf")

QWEN3_ALIGNER_FRONTEND = os.path.join(MODEL_DIR, "qwen3_aligner_encoder_frontend.int4.onnx")
QWEN3_ALIGNER_BACKEND = os.path.join(MODEL_DIR, "qwen3_aligner_encoder_backend.int4.onnx")
QWEN3_ALIGNER_LLM = os.path.join(MODEL_DIR, "qwen3_aligner_llm.q4_k.gguf")

PUNC_MODEL_DIR = r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline\models\Punct-CT-Transformer\punc_ct-transformer_cn-en"

# 是否启用 aligner
USE_ALIGNER = True
DEFAULT_LANGUAGE = "Chinese"
QWEN_CHUNK_SIZE = 40.0
QWEN_MEMORY_CHUNKS = 1
MAX_SUBTITLE_CHARS = 35
MAX_SUBTITLE_DURATION = 6.0
MAX_SUBTITLE_GAP = 1.0


def format_srt_timestamp(seconds: float) -> str:
    """格式化为标准 SRT 时间戳"""
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def check_ffmpeg():
    """检查ffmpeg是否可用"""
    if shutil.which("ffmpeg") is None:
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

    def __init__(self, num_threads: int = NUM_THREADS):
        self.num_threads = num_threads
        self.asr_model = None
        self.aligner = None
        self.punc_model = None
        self.load_models()

    def load_models(self):
        """加载模型"""
        print("正在加载模型...")
        start_time = time.time()

        # 检查Qwen3 ASR模型
        for path in [QWEN3_ASR_FRONTEND, QWEN3_ASR_BACKEND, QWEN3_ASR_LLM]:
            if not os.path.exists(path):
                print(f"错误: 未找到Qwen3 ASR模型文件: {path}")
                sys.exit(1)

        print("加载 Qwen3 ASR 模型...")
        try:
            if create_asr_engine:
                self.asr_model = create_asr_engine(
                    model_dir=MODEL_DIR,
                    encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
                    encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
                    llm_fn="qwen3_asr_llm.q4_k.gguf",
                    n_ctx=4096,
                    chunk_size=QWEN_CHUNK_SIZE,
                    pad_to=int(QWEN_CHUNK_SIZE),
                    use_dml=False,
                    vulkan_enable=False,
                    verbose=False,
                    enable_aligner=USE_ALIGNER,
                )
                print("Qwen3 ASR 模型加载成功 (使用 CapsWriter 适配器，支持长音频分块)")
            else:
                # 尝试原生 sherpa-onnx (如果版本支持)
                self.asr_model = sherpa_onnx.OfflineRecognizer.from_qwen3(
                    encoder_frontend=QWEN3_ASR_FRONTEND,
                    encoder_backend=QWEN3_ASR_BACKEND,
                    decoder=QWEN3_ASR_LLM,
                    num_threads=self.num_threads,
                    sample_rate=SAMPLE_RATE,
                    debug=False,
                )
                print("Qwen3 ASR 模型加载成功 (原生)")
        except Exception as e:
            print(f"加载 Qwen3 ASR 模型失败: {e}")
            raise

        # 加载 Qwen3 aligner（可选）
        if create_asr_engine:
            if USE_ALIGNER and getattr(getattr(self.asr_model, "engine", None), "aligner", None):
                print("Qwen3 Aligner 已通过 CapsWriter 适配器启用")
            elif USE_ALIGNER:
                print("警告: CapsWriter 适配器未启用对齐器，SRT 将使用粗略时间戳")
        elif USE_ALIGNER:
            if all(os.path.exists(p) for p in [QWEN3_ALIGNER_FRONTEND, QWEN3_ALIGNER_BACKEND, QWEN3_ALIGNER_LLM]):
                try:
                    print("加载 Qwen3 Aligner 模型...")
                    self.aligner = sherpa_onnx.Aligner.from_qwen3(
                        encoder_frontend=QWEN3_ALIGNER_FRONTEND,
                        encoder_backend=QWEN3_ALIGNER_BACKEND,
                        decoder=QWEN3_ALIGNER_LLM,
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
        punc_model_path = os.path.join(PUNC_MODEL_DIR, "model_quant.onnx")
        if os.path.exists(punc_model_path):
            try:
                print(f"加载标点模型: {punc_model_path}")
                import jieba
                import logging
                jieba.setLogLevel(logging.INFO)

                self.punc_model = CT_Transformer(PUNC_MODEL_DIR, quantize=True)
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
                "ffmpeg",
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

        if create_asr_engine and hasattr(self.asr_model, "engine"):
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

    def __init__(self, input_dir: str, output_dir: str, num_threads: int = NUM_THREADS):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.transcriber = AudioTranscriber(num_threads)

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
    parser.add_argument("--input", default=r"D:\video\mp3", help="输入音频文件目录")
    parser.add_argument("--output", default=r"D:\video\mp3", help="输出目录")
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help="线程数")
    parser.add_argument("--new", action="store_true", help="重新处理所有文件，不使用断点续传")

    args = parser.parse_args()

    check_ffmpeg()

    try:
        converter = MP3ToTextConverter(args.input, args.output, args.threads)
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
