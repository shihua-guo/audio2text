# audio2text

这是一个基于 **Qwen3-ASR** 和 **CapsWriter-Offline** 的离线音频转文本工具。它能够将音频文件（如 MP3, WAV, M4A 等）识别为纯文本（.txt）和带时间戳的 SRT 字幕文件（.srt）。

## 主要功能

*   **离线识别**: 使用 Qwen3-ASR 1.7B 模型进行离线语音转录，保护数据隐私。
*   **精确对齐**: 集成 Qwen3 Aligner 模型，生成更精准的字幕时间戳。
*   **标点恢复**: 内置标点符号恢复模型（CT-Transformer），提高识别结果的可读性。
*   **批量处理**: 支持对指定目录下的多个音频文件进行批量转录。
*   **断点续传**: 自动保存处理进度，支持中断后继续处理未完成的文件。
*   **长音频支持**: 通过 CapsWriter-Offline 适配器，支持对长音频进行分块处理。

## 环境要求

*   Python 3.8+
*   ffmpeg (用于音频解码，推荐)
*   [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) (推荐，用于 GGUF 长音频分块识别)

## 快速开始

1.  确保已安装依赖：
    ```bash
    pip install numpy librosa funasr-onnx jieba sherpa-onnx
    ```

2.  准备模型目录。`--model-dir` 需要指向包含以下文件的目录：
    ```text
    qwen3_asr_encoder_frontend.int4.onnx
    qwen3_asr_encoder_backend.int4.onnx
    qwen3_asr_llm.q4_k.gguf
    ```

3.  如果你使用 CapsWriter-Offline 的 Qwen3 GGUF 模型，额外传入 `--capswriter-dir` 指向 CapsWriter 根目录。

4.  运行转录工具：
    ```bash
    python mp3totext.py --input "D:\your\audio\dir" --output "D:\output\dir" --model-dir "D:\path\to\Qwen3-ASR-1.7B" --capswriter-dir "D:\path\to\CapsWriter-Offline"
    ```

4.1 如果你想直接尝试 GPU 开关：
    ```bash
    python mp3totext.py --input "D:\your\audio\dir" --output "D:\output\dir" --model-dir "D:\path\to\Qwen3-ASR-1.7B" --capswriter-dir "D:\path\to\CapsWriter-Offline" --dml
    ```
    或：
    ```bash
    python mp3totext.py --input "D:\your\audio\dir" --output "D:\output\dir" --model-dir "D:\path\to\Qwen3-ASR-1.7B" --capswriter-dir "D:\path\to\CapsWriter-Offline" --vulkan
    ```

5.  也可以使用环境变量，避免每次都传参：
    ```powershell
    $env:AUDIO2TEXT_MODEL_DIR="D:\path\to\Qwen3-ASR-1.7B"
    $env:AUDIO2TEXT_CAPSWRITER_DIR="D:\path\to\CapsWriter-Offline"
    python mp3totext.py --input "D:\your\audio\dir" --output "D:\output\dir"
    ```

## 兼容性说明

*   本项目不再要求手改 `mp3totext.py` 里的本地绝对路径。
*   如果 `CapsWriter-Offline` 可用，脚本会优先使用其 `create_asr_engine`，这也是当前 `.gguf` 模型的推荐运行方式。
*   `--dml` 和 `--vulkan` 只对 CapsWriter 适配器路径生效，不能同时使用。
*   即使启用了 `--dml` 或 `--vulkan`，长音频转录时 CPU 仍可能是主要瓶颈；脚本会额外输出可用 provider 和 DirectML 的实际启用状态。
*   如果走原生 `sherpa-onnx` 回退路径，当前 PyPI `sherpa-onnx==1.12.34` 的 Python API 是 `OfflineRecognizer.from_qwen3_asr(...)`，不是 `from_qwen3(...)`。
*   原生 `sherpa-onnx` 回退路径除了模型文件外，还需要 `vocab.json`、`merges.txt`、`tokenizer_config.json`。

## 命令行参数

*   `--input`: 输入音频文件目录，默认为 `D:\video\mp3`。
*   `--output`: 输出结果目录，默认为 `D:\video\mp3`。
*   `--threads`: 识别使用的线程数。
*   `--new`: 强制重新处理所有文件，忽略已有的进度记录。
*   `--model-dir`: Qwen3 ASR 模型目录。
*   `--capswriter-dir`: CapsWriter-Offline 根目录。
*   `--punc-model-dir`: 标点模型目录。
*   `--no-aligner`: 禁用精确时间戳对齐。
*   `--dml`: 使用 DirectML 加速 Qwen3 ASR。
*   `--vulkan`: 使用 Vulkan 加速 Qwen3 ASR。

## SRT 语义检索界面

项目现在也包含一个本地 `SRT` 语义检索 Web 应用，适合在建立字幕文件后做按含义搜索。

### 额外依赖

```bash
pip install -r requirements.txt
```

### 启动方式

```bash
python run_search_app.py
```

启动后打开：

```text
http://127.0.0.1:8000
```

### 使用流程

1. 点击“选择文件夹”或手动填写包含 `.srt` 文件的根目录。
2. 选择 embedding 模型。这里既可以填本地模型目录，也可以填 HuggingFace 模型名，或者在配置了 OpenAI 兼容 API 后填写远程模型名。
3. 点击“增量更新索引”，等待后台完成本地切块和向量化。
4. 在搜索框输入自然语言问题，查看命中的字幕片段、时间范围和上下文。

### Embedding API 配置

如果你希望 `embedding_model_dir` 通过 OpenAI 兼容接口调用，可以在 `config/runtime_config.json` 中这样配置：

```json
{
  "embedding_model_dir": "text-embedding-3-small",
  "embedding_api_base": "https://api.openai.com/v1",
  "embedding_api_key": "sk-xxx"
}
```

说明：

*   配置了 `embedding_api_base` 之后，`embedding_model_dir` 会被当作远程模型名，不再按本地路径解析。
*   `embedding_api_key` 为空时不会发送 `Authorization` 头，适合无鉴权的本地兼容服务。
*   也支持通过环境变量覆盖：`AUDIO2TEXT_EMBEDDING_API_BASE`、`AUDIO2TEXT_EMBEDDING_API_KEY`，其中 API Key 也兼容 `OPENAI_API_KEY`。

### 模型建议

*   `Qwen/Qwen3-Embedding-0.6B`: 更适合 CPU 或一般机器。
*   `Qwen/Qwen3-Embedding-4B`: 语义能力更强，但更吃内存，首次加载更慢。

## 致谢

本项目依赖并使用了以下优秀工具和项目：
*   [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
*   [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
*   [FunASR](https://github.com/alibaba-damo-academy/FunASR)
