from srt_search_app.main import app
from portable_runtime import load_runtime_config


if __name__ == "__main__":
    import uvicorn

    runtime = load_runtime_config()
    uvicorn.run(app, host="127.0.0.1", port=runtime.search_port)
