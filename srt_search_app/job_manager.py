from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Callable

from .models import IndexJobStatus


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, IndexJobStatus] = {}
        self._lock = threading.Lock()

    def create_job(self, root_path: str, model_name: str, force_rebuild: bool) -> IndexJobStatus:
        job = IndexJobStatus(
            job_id=uuid.uuid4().hex,
            root_path=root_path,
            model_name=model_name,
            force_rebuild=force_rebuild,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> IndexJobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job: IndexJobStatus) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def run_in_background(
        self,
        job: IndexJobStatus,
        target: Callable[[IndexJobStatus], None],
    ) -> None:
        def runner() -> None:
            try:
                job.state = "running"
                self.update_job(job)
                target(job)
                if job.state != "failed":
                    job.state = "completed"
                    job.message = "索引已完成"
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)
                job.message = "索引失败"
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                self.update_job(job)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
