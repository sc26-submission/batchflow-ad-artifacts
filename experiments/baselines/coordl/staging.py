from __future__ import annotations

import time
from typing import Any

import redis

from experiments.config.types import CoorDLRedisConfig


_PUBLISH_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
redis.call('SET', KEYS[2], 0)
redis.call('SET', KEYS[1], ARGV[1])
return 1
"""

_CONSUME_SCRIPT = """
local payload = redis.call('GET', KEYS[1])
if not payload then
    return nil
end

local used = redis.call('INCR', KEYS[2])
if used >= tonumber(ARGV[1]) then
    redis.call('DEL', KEYS[1])
    redis.call('DEL', KEYS[2])
end

return {payload, used}
"""


class CoorDLStagingStore:
    """Redis-backed short-lived staging area for prepared minibatches."""

    def __init__(self, config: CoorDLRedisConfig, namespace: str) -> None:
        self.namespace = namespace.rstrip(":")
        self.client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            ssl=config.ssl,
            password=config.password or None,
            decode_responses=False,
        )
        self._publish = self.client.register_script(_PUBLISH_SCRIPT)
        self._consume = self.client.register_script(_CONSUME_SCRIPT)

    def ping(self) -> None:
        self.client.ping()

    def publish(self, batch_id: str, payload: bytes) -> None:
        created = int(
            self._publish(
                keys=[self._payload_key(batch_id), self._count_key(batch_id)],
                args=[payload],
            )
        )
        if created != 1:
            raise RuntimeError(f"CoorDL batch already staged: {batch_id}")

    def wait_and_consume(
        self,
        batch_id: str,
        *,
        num_consumers: int,
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> bytes:
        deadline = time.monotonic() + timeout_seconds

        while True:
            result = self._consume(
                keys=[self._payload_key(batch_id), self._count_key(batch_id)],
                args=[num_consumers],
            )

            if result is not None:
                payload, _used = result
                return bytes(payload)

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for staged CoorDL batch {batch_id!r}"
                )

            time.sleep(poll_interval_seconds)

    def exists(self, batch_id: str) -> bool:
        return bool(self.client.exists(self._payload_key(batch_id)))

    def clear_namespace(self) -> int:
        keys = list(self.client.scan_iter(match=f"{self.namespace}:*", count=500))
        return int(self.client.delete(*keys)) if keys else 0

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _payload_key(self, batch_id: str) -> str:
        return f"{self.namespace}:batch:{batch_id}:payload"

    def _count_key(self, batch_id: str) -> str:
        return f"{self.namespace}:batch:{batch_id}:used"
