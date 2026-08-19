from __future__ import annotations

import threading
from dataclasses import dataclass

from batchflow.common.core import now_ms


@dataclass
class StoredPayload:
    payload: bytes
    payload_format: str
    created_at_ms: int
    last_accessed_at_ms: int
    evict_after_ms: int = 0

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


class InMemoryPayloadStore:
    """
    Worker-local in-memory payload store.

    Important:
    - key identifies the payload bytes.
    - payload_format identifies how those bytes should be decoded.
    - evict_after_ms means "eligible for cleanup after this time",
      not "invalid after this time".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payloads: dict[str, StoredPayload] = {}

    def put(
        self,
        *,
        key: str,
        payload: bytes,
        payload_format: str,
        ttl_seconds: float | None = None,
    ) -> int:
        current_time_ms = now_ms()
        evict_after_ms = 0

        if ttl_seconds is not None and ttl_seconds > 0:
            evict_after_ms = current_time_ms + int(ttl_seconds * 1000)

        with self._lock:
            self._payloads[key] = StoredPayload(
                payload=payload,
                payload_format=payload_format,
                created_at_ms=current_time_ms,
                last_accessed_at_ms=current_time_ms,
                evict_after_ms=evict_after_ms,
            )

        return evict_after_ms

    def get(self, *, key: str) -> StoredPayload | None:
        current_time_ms = now_ms()

        with self._lock:
            entry = self._payloads.get(key)

            if entry is None:
                return None

            entry.last_accessed_at_ms = current_time_ms
            return entry

    def contains(self, *, key: str) -> bool:
        with self._lock:
            return key in self._payloads

    def remove(self, *, key: str) -> bool:
        with self._lock:
            return self._payloads.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._payloads.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._payloads)

    def total_bytes(self) -> int:
        with self._lock:
            return sum(entry.size_bytes for entry in self._payloads.values())

    def cleanup_expired(self) -> int:
        current_time_ms = now_ms()

        with self._lock:
            expired_keys = [
                key
                for key, entry in self._payloads.items()
                if entry.evict_after_ms
                and entry.evict_after_ms <= current_time_ms
            ]

            for key in expired_keys:
                del self._payloads[key]

            return len(expired_keys)