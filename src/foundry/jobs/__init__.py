from ..models import Job, JobStatus
from ..database import get_session, JobRecord, JobStatus as DBJobStatus
from typing import Optional, Callable, Awaitable
import uuid
import asyncio
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class JobRunner:
    def __init__(self, max_concurrent: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._running: dict[str, asyncio.Task] = {}

    def create_job(self, job_type: str, params: dict | None = None) -> Job:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = Job(job_id=job_id, job_type=job_type, params=params or {})
        session = get_session()
        record = JobRecord(
            job_id=job.job_id,
            job_type=job.job_type,
            status=DBJobStatus.pending,
            params=job.params,
            created_at=job.created_at,
        )
        session.add(record)
        session.commit()
        session.close()
        return job

    def update_status(self, job_id: str, status: JobStatus, error: Optional[str] = None, result_ref: Optional[str] = None) -> None:
        session = get_session()
        record = session.query(JobRecord).filter_by(job_id=job_id).first()
        if record:
            record.status = DBJobStatus(status.value)
            if error:
                record.error_message = error
            if result_ref:
                record.result_ref = result_ref
            if status in (JobStatus.running, JobStatus.completed, JobStatus.failed):
                setattr(record, f"{status.value}_at", datetime.utcnow())
            session.commit()
        session.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        session = get_session()
        record = session.query(JobRecord).filter_by(job_id=job_id).first()
        session.close()
        if not record:
            return None
        return Job(
            job_id=record.job_id,
            job_type=record.job_type,
            status=JobStatus(record.status.value),
            params=record.params or {},
            result_ref=record.result_ref,
            error_message=record.error_message,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
        )

    def list_jobs(self, limit: int = 50) -> list[Job]:
        session = get_session()
        records = session.query(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit).all()
        session.close()
        return [
            Job(
                job_id=r.job_id,
                job_type=r.job_type,
                status=JobStatus(r.status.value),
                params=r.params or {},
                result_ref=r.result_ref,
                error_message=r.error_message,
                created_at=r.created_at,
                started_at=r.started_at,
                completed_at=r.completed_at,
            ) for r in records
        ]
