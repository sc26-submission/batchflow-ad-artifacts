from __future__ import annotations

import redis


class RedisPayloadStore:
    """Shared payload store backed by Redis / Amazon ElastiCache."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 6379,
        db: int = 0,
        ssl: bool = False,
        password: str | None = None,
        key_prefix: str = "batchflow",
    ) -> None:
        if not host:
            raise ValueError("Redis host must be non-empty when Redis is enabled")

        self.host = host
        self.port = int(port)
        self.db = int(db)
        self.ssl = bool(ssl)
        self.key_prefix = key_prefix.strip(":") or "batchflow"

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            ssl=self.ssl,
            password=password or None,
            socket_connect_timeout=5,
            socket_timeout=30,
        )

        # Fail early during startup if the shared cache is unreachable.
        self.client.ping()

    @property
    def location(self) -> str:
        scheme = "rediss" if self.ssl else "redis"
        return f"{scheme}://{self.host}:{self.port}/{self.db}"

    def make_key(self, key: str) -> str:
        """Convert a BatchFlow logical key into the concrete Redis key."""
        return f"{self.key_prefix}:{key}"

    def put(self, *, key: str, payload: bytes) -> str:
        """Store bytes under a logical key and return the concrete Redis key."""
        redis_key = self.make_key(key)
        self.client.set(redis_key, payload)
        return redis_key

    def get(self, *, key: str) -> bytes | None:
        """Read bytes using a BatchFlow logical key."""
        return self.client.get(self.make_key(key))

    def contains(self, *, key: str) -> bool:
        return bool(self.client.exists(self.make_key(key)))

    def size_bytes(self, *, key: str) -> int:
        return int(self.client.strlen(self.make_key(key)))

    def remove(self, *, key: str) -> bool:
        """Delete an object using its BatchFlow logical key."""
        return bool(self.client.delete(self.make_key(key)))

    def remove_concrete_key(self, *, key: str) -> bool:
        """Delete an object using the concrete Redis key stored in metadata."""
        if not key:
            return False
        return bool(self.client.delete(key))

    def close(self) -> None:
        self.client.close()
