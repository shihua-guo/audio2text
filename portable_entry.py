from __future__ import annotations

import argparse
import sys

from portable_runtime import APP_ROOT, RUNTIME_CONFIG_PATH, load_runtime_config


def run_search_server() -> int:
    import uvicorn

    from srt_search_app.main import app

    runtime = load_runtime_config()
    uvicorn.run(app, host="127.0.0.1", port=runtime.search_port)
    return 0


def run_transcriber(extra_args: list[str]) -> int:
    import mp3totext

    sys.argv = ["mp3totext"] + extra_args
    mp3totext.main()
    return 0


def show_paths() -> int:
    runtime = load_runtime_config()
    print(f"应用目录: {APP_ROOT}")
    print(f"配置文件: {RUNTIME_CONFIG_PATH}")
    print(f"CapsWriter目录: {runtime.capswriter_dir or '(未配置)'}")
    print(f"ASR模型目录: {runtime.asr_model_dir or '(未配置)'}")
    print(f"标点模型目录: {runtime.punc_model_dir or '(未配置)'}")
    print(f"Embedding模型目录: {runtime.embedding_model_dir or '(未配置)'}")
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "transcribe":
        return run_transcriber(sys.argv[2:])

    parser = argparse.ArgumentParser(description="Audio2Text 便携版入口")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("search", help="启动 SRT 语义检索 Web 服务")
    subparsers.add_parser("show-paths", help="显示当前便携包的配置路径")

    args = parser.parse_args()

    if args.command == "search":
        return run_search_server()
    if args.command == "show-paths":
        return show_paths()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
