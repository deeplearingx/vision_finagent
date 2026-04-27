import uuid
from ..core.redis_client import get_redis
from ..config import settings


def generate_token(prefix: str = "idem") -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


async def check_and_set(token: str) -> bool:
    """Returns True if token is new (first time seen), False if duplicate."""
    r = await get_redis()
    result = await r.set(token, "1", nx=True, ex=settings.IDEMPOTENCY_TTL)
    return bool(result)


async def release(token: str) -> None:
    """Release an idempotency token so the same key can be retried."""
    r = await get_redis()
    await r.delete(token)
