from __future__ import annotations

from urllib.parse import urlparse

import redis


class RedisFetchClient:
    """Trainer-side client for fetching prepared payloads directly from Redis."""

    def __init__(self, *, timeout_seconds: float | None = 30.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._clients: dict[str, redis.Redis] = {}

    def fetch_batch(
        self,
        *,
        location: str,
        key: str,
    ) -> bytes:
        if not key:
            raise ValueError("Redis fetch key must be non-empty")

        client = self._get_client(location)
        payload = client.get(key)

        if payload is None:
            raise KeyError(
                f"Redis payload not found location={location!r} key={key!r}"
            )

        return payload

    def _get_client(self, location: str) -> redis.Redis:
        existing = self._clients.get(location)
        if existing is not None:
            return existing

        parsed = urlparse(location)
        if parsed.scheme not in ("redis", "rediss"):
            raise ValueError(f"Unsupported Redis location: {location!r}")

        if not parsed.hostname:
            raise ValueError(f"Redis location is missing a hostname: {location!r}")

        try:
            db = int(parsed.path.lstrip("/") or 0)
        except ValueError as exc:
            raise ValueError(
                f"Invalid Redis database in location: {location!r}"
            ) from exc

        # No environment variables are used. For the artifact's ElastiCache
        # deployment, the endpoint is carried in `location`. If a password is
        # ever needed, a standard Redis URL containing credentials is accepted.
        client = redis.Redis(
            host=parsed.hostname,
            port=parsed.port or 6379,
            db=db,
            ssl=parsed.scheme == "rediss",
            username=parsed.username or None,
            password=parsed.password or None,
            socket_connect_timeout=(
                min(5.0, self.timeout_seconds)
                if self.timeout_seconds is not None
                else 5.0
            ),
            socket_timeout=self.timeout_seconds,
        )

        self._clients[location] = client
        return client

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
