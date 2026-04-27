import asyncio
import uuid
from ..core.redis_client import get_redis


class DistributedLock:
    def __init__(self, key: str, ttl: int = 30):
        self.key = f"lock:{key}"
        self.ttl = ttl
        self._token = str(uuid.uuid4())
        self._watchdog: asyncio.Task | None = None

    async def acquire(self) -> bool:
        r = await get_redis()
        acquired = await r.set(self.key, self._token, nx=True, ex=self.ttl)
        if acquired:
            self._watchdog = asyncio.create_task(self._renew())
        return bool(acquired)

    async def release(self):
        if self._watchdog:
            self._watchdog.cancel()
        r = await get_redis()
        val = await r.get(self.key)
        if val == self._token:
            await r.delete(self.key)

    async def _renew(self):
        while True:
            await asyncio.sleep(self.ttl * 0.6)
            r = await get_redis()
            val = await r.get(self.key)
            if val == self._token:
                await r.expire(self.key, self.ttl)

    async def __aenter__(self):
        if not await self.acquire():
            raise RuntimeError(f"Could not acquire lock: {self.key}")
        return self

    async def __aexit__(self, *_):
        await self.release()
