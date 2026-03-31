from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import EMBEDDING, PROJECT_ROOT
from .database import SearchDatabase
from .embeddings import EmbeddingBackend
from .indexing import SemanticSearchService
from .job_manager import JobManager
from portable_runtime import RUNTIME_CONFIG_PATH, load_runtime_config


FRONTEND_DIR = PROJECT_ROOT / "srt_search_app" / "frontend"
RUNTIME = load_runtime_config()


class StartIndexRequest(BaseModel):
    root_path: str
    model_name: str = Field(default=EMBEDDING.default_model_name)
    force_rebuild: bool = False


class SearchRequest(BaseModel):
    root_path: str
    query: str
    model_name: str = Field(default=EMBEDDING.default_model_name)
    limit: int = Field(default=10, ge=1, le=50)


db = SearchDatabase()
embedding_backend = EmbeddingBackend()
service = SemanticSearchService(db=db, embedding_backend=embedding_backend)
jobs = JobManager()

app = FastAPI(title="SRT Semantic Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def format_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def choose_directory() -> str:
    powershell_command = """
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择包含 SRT 文件的目录'
$dialog.ShowNewFolderButton = $true
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
"""

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-STA",
                "-Command",
                powershell_command,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        selected_path = result.stdout.strip()
        if selected_path:
            return selected_path
    except Exception:
        pass

    try:
        import tkinter as tk
        from tkinter import filedialog

        selected_path = ""
        done = threading.Event()

        def open_dialog() -> None:
            nonlocal selected_path
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected_path = filedialog.askdirectory()
            root.destroy()
            done.set()

        thread = threading.Thread(target=open_dialog)
        thread.start()
        done.wait()
        return selected_path
    except Exception:
        return ""


@app.get("/")
async def index_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/models")
async def get_models() -> dict:
    return {
        "default_model_name": EMBEDDING.default_model_name,
        "provider": "openai-compatible-api" if EMBEDDING.use_api else "local",
        "api_base_url": EMBEDDING.api_base_url,
        "runtime_config_path": str(RUNTIME_CONFIG_PATH),
        "recommended_models": [
            EMBEDDING.default_model_name,
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
        ],
    }


@app.post("/api/folders/choose")
async def api_choose_folder() -> dict:
    path = choose_directory()
    return {"path": path}


@app.get("/api/summary")
async def api_summary(root_path: str, model_name: str = EMBEDDING.default_model_name) -> dict:
    summary = service.summarize_root(root_path, model_name)
    return {"summary": summary}


@app.post("/api/index/start")
async def api_start_index(request: StartIndexRequest) -> dict:
    root_path = str(Path(request.root_path).resolve())
    if not Path(root_path).exists():
        raise HTTPException(status_code=404, detail="目录不存在")

    job = jobs.create_job(
        root_path=root_path,
        model_name=request.model_name,
        force_rebuild=request.force_rebuild,
    )

    def run_job(job_status):
        service.index_root(
            root_path=job_status.root_path,
            model_name=job_status.model_name,
            force_rebuild=job_status.force_rebuild,
            callback=jobs.update_job,
            job=job_status,
        )

    jobs.run_in_background(job, run_job)
    return {"job_id": job.job_id}


@app.get("/api/index/jobs/{job_id}")
async def api_get_job(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"job": job.__dict__}


@app.post("/api/search")
async def api_search(request: SearchRequest) -> dict:
    root_path = str(Path(request.root_path).resolve())
    if not Path(root_path).exists():
        raise HTTPException(status_code=404, detail="目录不存在")

    try:
        results = service.search(
            root_path=root_path,
            model_name=request.model_name,
            query=request.query,
            limit=request.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "results": [
            {
                "chunk_id": result.chunk_id,
                "score": round(result.score * 100, 2),
                "rel_path": result.rel_path,
                "abs_path": result.abs_path,
                "start_sec": result.start_sec,
                "end_sec": result.end_sec,
                "start_time": format_timestamp(result.start_sec),
                "end_time": format_timestamp(result.end_sec),
                "text": result.text,
                "prev_text": result.prev_text,
                "next_text": result.next_text,
                "line_start": result.line_start,
                "line_end": result.line_end,
            }
            for result in results
        ]
    }
