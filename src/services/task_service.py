import uuid
import asyncio
import time
import json
import structlog
from typing import Optional

from ..config import settings
from ..models.enums import TaskStatus
from ..core.redis_client import get_redis
from ..utils.idempotency import release as _release_idem
from .ingestion_service import ingest_report

log = structlog.get_logger()

# Redis key: task:{task_id}  TTL: 7 days
_TASK_TTL = 7 * 86400
_ingest_sem = asyncio.Semaphore(settings.INGEST_WORKERS)


class TaskInfo:
    __slots__ = ("task_id", "report_id", "status", "detail", "created_at", "updated_at")

    def __init__(self, task_id: str, report_id: str,
                 status: TaskStatus = TaskStatus.PENDING,
                 detail: str = "",
                 created_at: float = 0.0,
                 updated_at: float = 0.0):
        self.task_id = task_id
        self.report_id = report_id
        self.status = status
        self.detail = detail
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()


def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


def _to_dict(info: TaskInfo) -> dict:
    return {
        "task_id": info.task_id,
        "report_id": info.report_id,
        "status": info.status.value,
        "detail": info.detail,
        "created_at": info.created_at,
        "updated_at": info.updated_at,
    }


def _from_dict(d: dict) -> TaskInfo:
    return TaskInfo(
        task_id=d["task_id"],
        report_id=d["report_id"],
        status=TaskStatus(d["status"]),
        detail=d.get("detail", ""),
        created_at=float(d.get("created_at", 0)),
        updated_at=float(d.get("updated_at", 0)),
    )


async def _save_task(info: TaskInfo) -> None:
    r = await get_redis()
    await r.set(_task_key(info.task_id), json.dumps(_to_dict(info)), ex=_TASK_TTL)


async def get_task(task_id: str) -> Optional[TaskInfo]:
    r = await get_redis()
    raw = await r.get(_task_key(task_id))
    if raw is None:
        return None
    return _from_dict(json.loads(raw))


async def submit_ingest_task(report_id: str, path: str) -> str:
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    info = TaskInfo(task_id=task_id, report_id=report_id)
    await _save_task(info)
    asyncio.create_task(_run_ingest(task_id, report_id, path))
    log.info("task.submit", task_id=task_id, report_id=report_id)
    return task_id


async def _run_ingest(task_id: str, report_id: str, path: str):
    info = await get_task(task_id)
    if info is None:
        return
    info.status = TaskStatus.RUNNING
    info.updated_at = time.time()
    await _save_task(info)
    log.info("task.running", task_id=task_id, report_id=report_id)

    async with _ingest_sem:
        try:
            await asyncio.wait_for(
                ingest_report(report_id, [path]),
                timeout=settings.INGEST_TIMEOUT,
            )
            info.status = TaskStatus.SUCCESS
            info.detail = "Ingestion completed"
            log.info("task.success", task_id=task_id, report_id=report_id)
        except asyncio.TimeoutError:
            info.status = TaskStatus.FAILED
            info.detail = f"Ingestion timed out after {settings.INGEST_TIMEOUT}s"
            log.error("task.timeout", task_id=task_id, report_id=report_id, timeout=settings.INGEST_TIMEOUT)
        except Exception as exc:
            info.status = TaskStatus.FAILED
            info.detail = str(exc)
            log.error("task.failed", task_id=task_id, report_id=report_id, error=str(exc))
        finally:
            info.updated_at = time.time()
            await _save_task(info)
            # Release idempotency token on failure so the same report_id can be retried.
            # On success the token is intentionally kept (TTL=IDEMPOTENCY_TTL) to block
            # duplicate submissions of an already-succeeded ingest within that window.
            if info.status == TaskStatus.FAILED:
                await _release_idem(f"upload:{report_id}")
            try:
                import os
                os.unlink(path)
            except Exception:
                pass
