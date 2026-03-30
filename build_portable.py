from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
PORTABLE_DIR = DIST_DIR / "audio2text_portable"
CONFIG_SRC = PROJECT_ROOT / "config" / "runtime_config.json"


SEARCH_BAT = r"""@echo off
setlocal
start "" "%~dp0audio2text_portable.exe" search
timeout /t 2 >nul
start "" http://127.0.0.1:8000
endlocal
"""


TRANSCRIBE_BAT = r"""@echo off
setlocal
"%~dp0audio2text_portable.exe" transcribe %*
endlocal
"""


SHOW_PATHS_BAT = r"""@echo off
setlocal
"%~dp0audio2text_portable.exe" show-paths
pause
endlocal
"""


OFFLINE_README = """Audio2Text 便携离线包
====================

1. 模型不会随包附带。
2. 请把模型复制到以下目录，或修改 config\\runtime_config.json:
   - ASR 模型: models\\Qwen3-ASR\\Qwen3-ASR-1.7B
   - 标点模型: models\\Punct-CT-Transformer\\punc_ct-transformer_cn-en
   - Embedding 模型: models\\embeddings\\Qwen3-Embedding-0.6B
3. 启动方式:
   - 启动字幕检索.bat
   - 启动音频转写.bat
4. 默认输入输出目录:
   - data\\audio_input
   - data\\audio_output
5. 如需检查当前实际读取的配置路径，请运行:
   - 显示当前配置路径.bat
"""


def build() -> None:
    subprocess.run(
        [
            "python",
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "portable_bundle.spec",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    PORTABLE_DIR.mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "config").mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "data" / "audio_input").mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "data" / "audio_output").mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "data" / "search_app").mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "models" / "Qwen3-ASR" / "Qwen3-ASR-1.7B").mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "models" / "Punct-CT-Transformer" / "punc_ct-transformer_cn-en").mkdir(parents=True, exist_ok=True)
    (PORTABLE_DIR / "models" / "embeddings" / "Qwen3-Embedding-0.6B").mkdir(parents=True, exist_ok=True)

    shutil.copy2(CONFIG_SRC, PORTABLE_DIR / "config" / "runtime_config.json")
    (PORTABLE_DIR / "启动字幕检索.bat").write_text(SEARCH_BAT, encoding="utf-8")
    (PORTABLE_DIR / "启动音频转写.bat").write_text(TRANSCRIBE_BAT, encoding="utf-8")
    (PORTABLE_DIR / "显示当前配置路径.bat").write_text(SHOW_PATHS_BAT, encoding="utf-8")
    (PORTABLE_DIR / "README_离线使用说明.txt").write_text(OFFLINE_README, encoding="utf-8")


if __name__ == "__main__":
    build()
