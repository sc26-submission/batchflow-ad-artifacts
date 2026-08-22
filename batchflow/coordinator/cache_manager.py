from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from batchflow.cache.redis_store import RedisPayloadStore
from batchflow.common.core import Batch, PayloadEntry, StorageClass
from batchflow.coordinator.scheduler import BatchflowScheduler
from batchflow.coordinator.store import CoordinatorStore


LOGGER = logging.getLogger("batchflow.coordinator.cache")


@dataclass(frozen=True)
class PayloadPlacement:
    """Final location selected for a completed materialization."""

    location: str
    storage_class: StorageClass
    expires_at_ms: int | None
    metadata: dict[str, Any]


class ReusableCacheManager:
    """Owns admission and eviction for reusable Redis payloads."""

    def __init__(
        self,
        *,
        store: CoordinatorStore,
        scheduler: BatchflowScheduler,
        redis_store: RedisPayloadStore,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        self.redis_store = redis_store
        self._lock = threading.RLock()

    def admit_reusable_payload(
        self,
        *,
        batch: Batch,
        size_bytes: int,
        location: str,
        expires_at_ms: int | None,
        metadata: dict[str, Any],
    ) -> PayloadPlacement:
        """
        Decide whether a newly materialized Redis payload should be retained.

        The worker has already written the payload to Redis and kept a temporary
        worker-local fallback. This method either retains the Redis object or
        deletes it and returns the transient fallback location.
        """
        with self._lock:
            metadata = dict(metadata)
            incoming_value = self.scheduler.compute_batch_value(batch, self.store)
            metadata["batch_value"] = f"{incoming_value:.12g}"

            victims = self._choose_eviction_victims(
                incoming_batch=batch,
                incoming_size_bytes=size_bytes,
            )
            capacity = int(self.scheduler.config.cache_capacity_bytes)

            if victims is not None:
                for victim in victims:
                    self._evict_reusable_payload(victim)

                if capacity <= 0 or self.store.reusable_cache_bytes() + int(size_bytes) <= capacity:
                    metadata["cache_admission"] = "admitted"
                    LOGGER.debug(
                        "Reusable payload admitted | batch=%s | size=%s | value=%.6f",
                        batch.batch_id,
                        size_bytes,
                        incoming_value,
                    )
                    return PayloadPlacement(location, StorageClass.REUSABLE, expires_at_ms, metadata)

            return self._use_transient_fallback(
                batch=batch,
                location=location,
                expires_at_ms=expires_at_ms,
                metadata=metadata,
            )

    def can_admit_estimated_payload(self, batch: Batch) -> bool:
        """Return whether the reusable cache can plausibly retain this batch."""
        capacity = int(self.scheduler.config.cache_capacity_bytes)
        if capacity <= 0:
            return True

        estimated_size = int(round(self.store.get_payload_size_estimate(batch.dataset_id, default=0.0)))
        if estimated_size <= 0:
            return self.store.reusable_cache_bytes() < capacity

        return self._choose_eviction_victims(
            incoming_batch=batch,
            incoming_size_bytes=estimated_size,
        ) is not None

    def _choose_eviction_victims(
        self,
        *,
        incoming_batch: Batch,
        incoming_size_bytes: int,
    ) -> list[PayloadEntry] | None:
        """Choose lower-value unpinned Redis payloads that make room for a new one."""
        capacity = int(self.scheduler.config.cache_capacity_bytes)
        incoming_size_bytes = max(0, int(incoming_size_bytes))

        if capacity <= 0:
            return []
        if incoming_size_bytes > capacity:
            return None

        required_bytes = self.store.reusable_cache_bytes() + incoming_size_bytes - capacity
        if required_bytes <= 0:
            return []

        incoming_value = self.scheduler.compute_batch_value(incoming_batch, self.store)
        candidates: list[tuple[float, PayloadEntry]] = []

        for entry in self.store.list_reusable_payload_entries():
            if self.store.payload_pin_count(entry.cache_key) > 0:
                continue

            value = self.scheduler.compute_payload_value(entry, self.store)
            if value < incoming_value:
                candidates.append((value, entry))

        candidates.sort(key=lambda item: (item[0], item[1].last_accessed_at_ms, item[1].cache_key))

        victims: list[PayloadEntry] = []
        freed_bytes = 0
        for _, entry in candidates:
            victims.append(entry)
            freed_bytes += max(0, int(entry.size_bytes))
            if freed_bytes >= required_bytes:
                return victims

        return None

    def _evict_reusable_payload(self, victim: PayloadEntry) -> None:
        value = self.scheduler.compute_payload_value(victim, self.store)
        removed = self.store.remove_payload(victim.cache_key)
        if removed is None:
            return

        redis_key = str(removed.metadata.get("fetch_key", ""))
        if redis_key:
            try:
                self.redis_store.remove_concrete_key(key=redis_key)
            except Exception:
                LOGGER.exception(
                    "Failed to delete evicted Redis payload | cache_key=%s | redis_key=%s",
                    removed.cache_key,
                    redis_key,
                )

        LOGGER.info(
            "Reusable cache eviction | batch=%s | size=%s | value=%.6f",
            removed.metadata.get("batch_id", removed.cache_key),
            removed.size_bytes,
            value,
        )

    def _use_transient_fallback(
        self,
        *,
        batch: Batch,
        location: str,
        expires_at_ms: int | None,
        metadata: dict[str, Any],
    ) -> PayloadPlacement:
        redis_key = str(metadata.get("fetch_key", ""))
        if redis_key:
            try:
                self.redis_store.remove_concrete_key(key=redis_key)
            except Exception:
                LOGGER.exception(
                    "Failed to delete rejected Redis payload | batch=%s | redis_key=%s",
                    batch.batch_id,
                    redis_key,
                )

        fallback_host = str(metadata.get("fallback_fetch_host", ""))
        fallback_port = _safe_int(metadata.get("fallback_fetch_port"), default=0)
        fallback_key = str(metadata.get("fallback_fetch_key", ""))
        fallback_expires_at_ms = _safe_int(metadata.get("fallback_expires_at_ms"), default=0)

        if not fallback_host or fallback_port <= 0 or not fallback_key:
            LOGGER.warning(
                "Reusable cache rejection skipped | batch=%s | reason=missing transient fallback",
                batch.batch_id,
            )
            metadata["cache_admission"] = "forced_no_fallback"
            return PayloadPlacement(location, StorageClass.REUSABLE, expires_at_ms, metadata)

        fallback_metadata = dict(metadata)
        fallback_metadata.update(
            {
                "fetch_host": fallback_host,
                "fetch_port": str(fallback_port),
                "fetch_key": fallback_key,
                "cache_admission": "rejected",
            }
        )

        LOGGER.debug(
            "Reusable cache admission rejected | batch=%s | fallback=%s:%s",
            batch.batch_id,
            fallback_host,
            fallback_port,
        )
        return PayloadPlacement(
            location=f"grpc://{fallback_host}:{fallback_port}",
            storage_class=StorageClass.TRANSIENT,
            expires_at_ms=fallback_expires_at_ms or None,
            metadata=fallback_metadata,
        )


def _safe_int(value: object, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
