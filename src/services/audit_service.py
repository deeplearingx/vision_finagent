import structlog
from ..models.schema import AuditLog

log = structlog.get_logger()


async def record(entry: AuditLog) -> None:
    log.info("audit", **entry.model_dump())
