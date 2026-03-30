# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path.cwd().resolve()
sys.path.insert(0, str(project_root))

from portable_runtime import load_runtime_config


def detect_capswriter_dir() -> Path:
    runtime = load_runtime_config()
    candidates = [
        os.environ.get("AUDIO2TEXT_CAPSWRITER_DIR", "").strip(),
        runtime.capswriter_dir,
        r"C:\Users\shihu\Documents\software\CapsWriter-Offline-20260304\CapsWriter-Offline",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise SystemExit(
        "未找到 CapsWriter-Offline 代码目录。"
        "请在 config/runtime_config.json 设置 capswriter_dir，"
        "或设置环境变量 AUDIO2TEXT_CAPSWRITER_DIR 后再打包。"
    )


capswriter_dir = detect_capswriter_dir()
internal_dir = capswriter_dir / "internal"
pathex = [str(project_root), str(capswriter_dir), str(internal_dir)]
for path in pathex:
    if path not in sys.path:
        sys.path.insert(0, path)

frontend_datas = [
    (str(project_root / "srt_search_app" / "frontend"), "srt_search_app/frontend"),
]

datas = frontend_datas
datas += collect_data_files("uvicorn")
datas += collect_data_files("fastapi")
datas += collect_data_files("starlette")
datas += collect_data_files("transformers")
datas += collect_data_files("tokenizers")
datas += collect_data_files("funasr_onnx")
datas += collect_data_files("sherpa_onnx")
datas += collect_data_files("jieba")

hiddenimports = [
    "funasr_onnx",
    "jieba",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "sherpa_onnx",
]
hiddenimports += collect_submodules("funasr_onnx")
hiddenimports += collect_submodules("util.qwen_asr_gguf")


a = Analysis(
    ["portable_entry.py"],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="audio2text_portable",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="audio2text_portable",
)
