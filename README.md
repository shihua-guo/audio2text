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
*   [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) (需安装并配置模型路径)

## 快速开始

1.  确保已安装依赖：
    ```bash
    pip install numpy librosa funasr-onnx jieba sherpa-onnx
    ```

2.  修改 `mp3totext.py` 中的模型路径配置（如果需要）：
    ```python
    MODEL_DIR = r"C:\path\to\your\CapsWriter-Offline\models\Qwen3-ASR\Qwen3-ASR-1.7B"
    ```

3.  运行转录工具：
    ```bash
    python mp3totext.py --input "D:\your\audio\dir" --output "D:\output\dir"
    ```

## 命令行参数

*   `--input`: 输入音频文件目录，默认为 `D:\video\mp3`。
*   `--output`: 输出结果目录，默认为 `D:\video\mp3`。
*   `--threads`: 识别使用的线程数。
*   `--new`: 强制重新处理所有文件，忽略已有的进度记录。

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
2. 选择 embedding 模型，默认是 `Qwen/Qwen3-Embedding-0.6B`。
3. 点击“增量更新索引”，等待后台完成本地切块和向量化。
4. 在搜索框输入自然语言问题，查看命中的字幕片段、时间范围和上下文。

### 模型建议

*   `Qwen/Qwen3-Embedding-0.6B`: 更适合 CPU 或一般机器。
*   `Qwen/Qwen3-Embedding-4B`: 语义能力更强，但更吃内存，首次加载更慢。

## 致谢

本项目依赖并使用了以下优秀工具和项目：
*   [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
*   [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
*   [FunASR](https://github.com/alibaba-damo-academy/FunASR)
