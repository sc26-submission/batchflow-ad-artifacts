from __future__ import annotations

import math
from dataclasses import replace

from batchflow.common.core import (
    Batch,
    BatchHandle,
    BatchHandleStatus,
    CacheEntry,
    CacheStatus,
    Job,
    MaterializeBatchTask,
    MaterializeBatchTaskStatus,
    StorageClass,
    make_id,
    now_ms,
)
from batchflow.common.payload_formats import PayloadFormat
from batchflow.config.config_types import SchedulerConfig
from batchflow.coordinator.store import CoordinatorStore, MaterializeBatchTaskState


CACHE_RESULT_HIT = "hit"
CACHE_RESULT_IN_FLIGHT = "in_flight"
CACHE_RESULT_MISS = "miss"
CACHE_RESULT_FAILED = "failed"

_HANDLE_FIELD_METADATA_KEYS = {
    "fetch_host",
    "fetch_port",
    "fetch_key",
    "payload_format",
    "dataset_format",
    "fallback_fetch_host",
    "fallback_fetch_port",
    "fallback_fetch_key",
    "fallback_expires_at_ms",
}


class BatchflowScheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    # ------------------------------------------------------------------
    # Online target-depth and scheduling estimates
    # ------------------------------------------------------------------

    def target_lookahead_depth(
        self,
        job: Job,
        store: CoordinatorStore,
    ) -> int:
        """Estimate d_j = ceil(r_j / c_j), with a stable fallback."""
        metrics = store.get_job_metrics(job.job_id)
        prep_time = store.get_preparation_time_estimate(
            job.dataset_id,
            default=0.0,
        )

        if metrics.avg_compute_time_sec > 0.0 and prep_time > 0.0:
            # r_j = 1 / compute_time; c_j = 1 / prep_time.
            depth = int(math.ceil(prep_time / metrics.avg_compute_time_sec))
        else:
            depth = int(self.config.target_ready_batches)

        depth = max(1, depth)

        # The coordinator can only reason over batches it has planned.
        if job.lookahead_batches > 0:
            depth = min(depth, int(job.lookahead_batches))

        return depth

    def compute_job_urgency(self, job: Job, store: CoordinatorStore) -> float:
        target_depth = self.target_lookahead_depth(job, store)
        ready_ahead = store.count_ready_or_inflight_in_window(
            job.job_id,
            epoch=job.progress.epoch,
            start_batch_index=job.progress.next_batch_index,
            limit=target_depth,
        )

        deficit = max(0, target_depth - ready_ahead)
        coverage = min(1.0, float(ready_ahead) / float(target_depth))
        metrics = store.get_job_metrics(job.job_id)

        if metrics.last_commit_at_ms is None:
            stall_seconds = 1.0
        else:
            stall_seconds = max(
                0.0,
                (now_ms() - metrics.last_commit_at_ms) / 1000.0,
            )

        return (
            2.0 * float(deficit)
            + (1.0 - coverage)
            + min(stall_seconds, 10.0)
            + 1.5 * float(metrics.pending_batch_requests)
        )

    def score_batch(
        self,
        batch: Batch,
        *,
        job: Job,
        store: CoordinatorStore,
    ) -> float:
        urgency = self.compute_job_urgency(job, store)

        distance = max(
            0,
            batch.batch_index - job.progress.next_batch_index,
        )
        proximity = 1.0 / (1.0 + distance)

        cache_entry = (
            store.get_cache_entry(batch.cache_key)
            if self.config.reuse_enabled
            else None
        )
        is_cached = (
            cache_entry is not None
            and cache_entry.status == CacheStatus.AVAILABLE
        )

        ready_bonus = self.config.ready_cache_bonus if is_cached else 0.0
        reuse_bonus = self.estimate_reuse_potential(batch, store)
        materialization_cost = self.estimate_materialization_cost(batch, store)

        return (
            ready_bonus
            + self.config.job_urgency_weight * urgency
            + self.config.batch_proximity_weight * proximity
            + self.config.reuse_weight * reuse_bonus
            + self.config.materialization_cost_weight * materialization_cost
        )

    def estimate_reuse_potential(
        self,
        batch: Batch,
        store: CoordinatorStore,
    ) -> float:
        if not self.config.reuse_enabled:
            return 0.0

        compatible_jobs = 0

        for job in store.list_active_jobs():
            if (
                job.dataset_id == batch.dataset_id
                and job.progress.epoch == batch.epoch
            ):
                compatible_jobs += 1

        return float(max(0, compatible_jobs - 1))

    def estimate_materialization_cost(
        self,
        batch: Batch,
        store: CoordinatorStore,
    ) -> float:
        estimate = store.get_preparation_time_estimate(
            batch.dataset_id,
            default=0.0,
        )
        if estimate > 0.0:
            return estimate
        return float(batch.batch_size)

    # ------------------------------------------------------------------
    # Benefit-aware cache value and bounded retention
    # ------------------------------------------------------------------

    def compute_batch_value(
        self,
        batch: Batch,
        store: CoordinatorStore,
    ) -> float:
        return self.compute_cache_key_value(
            cache_key=batch.cache_key,
            dataset_id=batch.dataset_id,
            store=store,
        )

    def compute_cache_entry_value(
        self,
        entry: CacheEntry,
        store: CoordinatorStore,
    ) -> float:
        dataset_id = str(entry.metadata.get("dataset_id", ""))
        if not dataset_id:
            return 0.0

        return self.compute_cache_key_value(
            cache_key=entry.cache_key,
            dataset_id=dataset_id,
            store=store,
        )

    def compute_cache_key_value(
        self,
        *,
        cache_key: str,
        dataset_id: str,
        store: CoordinatorStore,
    ) -> float:
        if not self.config.reuse_enabled:
            return 0.0

        future_demand = 0.0
        input_shortage = 0.0
        multiplier = max(1, int(self.config.extended_lookahead_multiplier))

        for job in store.list_active_jobs():
            if job.dataset_id != dataset_id:
                continue

            target_depth = self.target_lookahead_depth(job, store)
            extended_depth = max(1, multiplier * target_depth)
            upcoming = store.next_uncommitted_batches(job.job_id)[:extended_depth]

            position: int | None = None
            for index, candidate in enumerate(upcoming):
                if candidate.cache_key == cache_key:
                    position = index
                    break

            if position is None:
                continue

            # h_hat(b): batches requested sooner contribute more.
            future_demand += 1.0 / float(position + 1)

            # s_hat(b): jobs with less target coverage contribute more.
            prepared = store.count_ready_or_inflight_in_window(
                job.job_id,
                epoch=job.progress.epoch,
                start_batch_index=job.progress.next_batch_index,
                limit=target_depth,
            )
            coverage = min(1.0, float(prepared) / float(target_depth))
            input_shortage += 1.0 - coverage

        if future_demand <= 0.0:
            return 0.0

        preparation_time = store.get_preparation_time_estimate(
            dataset_id,
            default=1.0,
        )

        return (
            future_demand
            * preparation_time
            * (1.0 + self.config.batch_value_beta * input_shortage)
        )

    def choose_cache_evictions(
        self,
        *,
        incoming_batch: Batch,
        incoming_size_bytes: int,
        store: CoordinatorStore,
    ) -> list[CacheEntry] | None:
        """
        Return lower-value unpinned victims needed to admit `incoming_batch`.

        [] means there is already enough room (or capacity is unbounded).
        None means the incoming object should not be retained in Redis.
        """
        if not self.config.reuse_enabled:
            return None

        capacity = int(self.config.cache_capacity_bytes)
        incoming_size_bytes = max(0, int(incoming_size_bytes))

        if capacity <= 0:
            return []

        if incoming_size_bytes > capacity:
            return None

        used_bytes = store.reusable_cache_bytes()
        required_bytes = used_bytes + incoming_size_bytes - capacity

        if required_bytes <= 0:
            return []

        incoming_value = self.compute_batch_value(incoming_batch, store)
        candidates: list[tuple[float, CacheEntry]] = []

        for entry in store.list_reusable_cache_entries():
            if store.cache_pin_count(entry.cache_key) > 0:
                continue

            value = self.compute_cache_entry_value(entry, store)

            # Retain an incoming object only by replacing strictly lower-value
            # entries, as described by the BatchFlow cache policy.
            if value < incoming_value:
                candidates.append((value, entry))

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].last_accessed_at_ms,
                item[1].cache_key,
            )
        )

        victims: list[CacheEntry] = []
        freed_bytes = 0

        for _, entry in candidates:
            victims.append(entry)
            freed_bytes += max(0, int(entry.size_bytes))

            if freed_bytes >= required_bytes:
                return victims

        return None

    # ------------------------------------------------------------------
    # Opportunistic prefetching
    # ------------------------------------------------------------------

    def score_opportunistic_batch(
        self,
        batch: Batch,
        *,
        store: CoordinatorStore,
    ) -> float:
        """
        Score an extended-lookahead prefetch candidate.

        Opportunistic prefetching is intentionally driven directly by the
        benefit-aware batch value V(b). Normal near-term demand is handled by
        the per-job maintenance pass before this method is used.
        """
        return self.compute_batch_value(batch, store)

    def can_opportunistically_prefetch(
        self,
        batch: Batch,
        *,
        store: CoordinatorStore,
    ) -> bool:
        """Best-effort cache-capacity check before scheduling a prefetch."""
        if not self.config.reuse_enabled:
            return False

        if not self.config.opportunistic_prefetch_enabled:
            return False

        if store.is_batch_ready(batch.batch_id):
            return False

        if store.is_batch_in_progress(batch.batch_id):
            return False

        value = self.compute_batch_value(batch, store)
        if value <= 0.0:
            return False

        capacity = int(self.config.cache_capacity_bytes)
        if capacity <= 0:
            return True

        estimated_size = int(
            round(
                store.get_payload_size_estimate(
                    batch.dataset_id,
                    default=0.0,
                )
            )
        )

        # Before any payload-size observation exists, allow the first candidate
        # only while the cache is not already at its configured logical bound.
        if estimated_size <= 0:
            return store.reusable_cache_bytes() < capacity

        return (
            self.choose_cache_evictions(
                incoming_batch=batch,
                incoming_size_bytes=estimated_size,
                store=store,
            )
            is not None
        )

    def build_opportunistic_task(
        self,
        batch: Batch,
        *,
        store: CoordinatorStore,
    ) -> MaterializeBatchTask:
        """Build a global reusable-cache prefetch task."""
        if not self.config.reuse_enabled:
            raise RuntimeError(
                "cannot build an opportunistic reuse task when reuse is disabled"
            )

        dataset = store.get_dataset(batch.dataset_id)
        value = self.score_opportunistic_batch(batch, store=store)

        return MaterializeBatchTask(
            task_id=make_id("task"),
            batch=batch,
            # This task is global rather than owned by one job. That prevents
            # another job from losing a useful prefetch when one consumer exits.
            job_id="",
            priority=value,
            storage_class=StorageClass.REUSABLE,
            payload_format=dataset.payload_format,
            dataset_format=dataset.dataset_format,
            created_at_ms=now_ms(),
            metadata={
                "opportunistic_prefetch": "true",
                "batch_value": f"{value:.12g}",
            },
        )

    @staticmethod
    def is_opportunistic_task(
        task_state: MaterializeBatchTaskState,
    ) -> bool:
        value = task_state.task.metadata.get("opportunistic_prefetch", "")
        return str(value).lower() == "true"

    # ------------------------------------------------------------------
    # Batch/task selection
    # ------------------------------------------------------------------

    def choose_next_batch_for_job(
        self,
        job_id: str,
        store: CoordinatorStore,
    ) -> Batch | None:
        if not store.is_job_active(job_id):
            return None

        job = store.get_job(job_id)
        candidates = store.next_uncommitted_batches(job_id)

        if not candidates:
            return None

        window = candidates[: max(1, self.config.candidate_window_size)]

        scored = [
            (self.score_batch(batch, job=job, store=store), batch)
            for batch in window
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return scored[0][1]

    def resolve_batch_handle(
        self,
        batch: Batch,
        store: CoordinatorStore,
    ) -> BatchHandle:
        handle = store.get_batch_handle(batch.batch_id)

        if handle is not None:
            return self._with_cache_result(
                handle,
                cache_result=CACHE_RESULT_HIT,
            )

        task_state = store.get_task_for_batch(batch.batch_id)

        if task_state is not None:
            if task_state.status in (
                MaterializeBatchTaskStatus.PENDING,
                MaterializeBatchTaskStatus.RUNNING,
            ):
                return self._pending_handle(
                    batch,
                    store=store,
                    cache_result=CACHE_RESULT_IN_FLIGHT,
                    metadata={"task_id": task_state.task.task_id},
                )

            if task_state.status in (
                MaterializeBatchTaskStatus.FAILED,
                MaterializeBatchTaskStatus.CANCELLED,
            ):
                return self._failed_handle(
                    batch,
                    store=store,
                    cache_result=CACHE_RESULT_FAILED,
                    reason=(
                        task_state.failure_reason
                        or f"task status={task_state.status.value}"
                    ),
                    metadata={"task_id": task_state.task.task_id},
                )

        entry = store.get_cache_entry(batch.cache_key)

        if entry is not None and entry.status == CacheStatus.FAILED:
            return self._failed_handle(
                batch,
                store=store,
                cache_result=CACHE_RESULT_FAILED,
                reason=str(
                    entry.metadata.get(
                        "failure_reason",
                        "cache entry failed",
                    )
                ),
            )

        return self._pending_handle(
            batch,
            store=store,
            cache_result=CACHE_RESULT_MISS,
        )

    def should_materialize(
        self,
        batch: Batch,
        store: CoordinatorStore,
    ) -> bool:
        if store.is_batch_ready(batch.batch_id):
            return False

        if store.is_batch_in_progress(batch.batch_id):
            return False

        return True

    def choose_storage_class(
        self,
        batch: Batch,
        store: CoordinatorStore,
    ) -> StorageClass:
        if not self.config.reuse_enabled:
            return StorageClass.TRANSIENT

        reuse = self.estimate_reuse_potential(batch, store)
        cost = self.estimate_materialization_cost(batch, store)

        if reuse >= self.config.reuse_threshold:
            return StorageClass.REUSABLE

        if cost >= self.config.cache_cost_threshold and reuse > 0:
            return StorageClass.REUSABLE

        return StorageClass.TRANSIENT

    def build_task(
        self,
        batch: Batch,
        *,
        job: Job,
        store: CoordinatorStore,
    ) -> MaterializeBatchTask:
        priority = self.score_batch(batch, job=job, store=store)
        storage_class = self.choose_storage_class(batch, store)
        dataset = store.get_dataset(batch.dataset_id)

        return MaterializeBatchTask(
            task_id=make_id("task"),
            batch=batch,
            job_id=job.job_id,
            priority=priority,
            storage_class=storage_class,
            payload_format=dataset.payload_format,
            dataset_format=dataset.dataset_format,
            created_at_ms=now_ms(),
            metadata={},
        )

    def score_pending_task(
        self,
        task_state: MaterializeBatchTaskState,
        store: CoordinatorStore,
    ) -> float:
        # Kept in the signature for compatibility with existing call sites.
        _ = store
        return float(task_state.task.priority)

    def choose_next_task_for_worker(
        self,
        worker_id: str,
        store: CoordinatorStore,
    ) -> MaterializeBatchTaskState | None:
        # Workers now share one global pool, so worker identity does not
        # constrain which pending task can be selected.
        _ = worker_id
        pending = store.list_pending_tasks()

        if not pending:
            return None

        # All workers share one pool. Near-term demand always wins;
        # opportunistic prefetch only uses otherwise-idle worker capacity.
        normal_pending = [
            state
            for state in pending
            if not self.is_opportunistic_task(state)
        ]
        candidates = normal_pending or pending

        candidates.sort(
            key=lambda task_state: self.score_pending_task(task_state, store),
            reverse=True,
        )

        return candidates[0]

    # ------------------------------------------------------------------
    # Handle construction
    # ------------------------------------------------------------------

    def build_ready_handle(
        self,
        *,
        batch: Batch,
        entry: CacheEntry,
        store: CoordinatorStore,
    ) -> BatchHandle:
        dataset = store.get_dataset(batch.dataset_id)

        metadata = {
            "cache_result": CACHE_RESULT_HIT,
            "worker_id": entry.worker_id or "",
            "storage_class": entry.storage_class.value,
        }

        metadata.update(
            {
                str(key): str(value)
                for key, value in entry.metadata.items()
                if key not in _HANDLE_FIELD_METADATA_KEYS
            }
        )

        payload_format = _payload_format_from_entry(
            entry,
            default=dataset.payload_format,
        )

        return BatchHandle(
            batch_id=batch.batch_id,
            cache_key=entry.cache_key,
            status=BatchHandleStatus.READY,
            fetch_host=str(entry.metadata.get("fetch_host", "")),
            fetch_port=_int_metadata(entry.metadata, "fetch_port", default=0),
            fetch_key=str(entry.metadata.get("fetch_key", batch.batch_id)),
            payload_format=payload_format,
            dataset_format=dataset.dataset_format,
            location=entry.location or "",
            expires_at_ms=entry.expires_at_ms,
            metadata=metadata,
        )

    def _pending_handle(
        self,
        batch: Batch,
        *,
        store: CoordinatorStore,
        cache_result: str,
        metadata: dict[str, str] | None = None,
    ) -> BatchHandle:
        dataset = store.get_dataset(batch.dataset_id)

        handle_metadata = {"cache_result": cache_result}
        handle_metadata.update(metadata or {})

        return BatchHandle(
            batch_id=batch.batch_id,
            cache_key=batch.cache_key,
            status=BatchHandleStatus.PENDING,
            payload_format=dataset.payload_format,
            dataset_format=dataset.dataset_format,
            metadata=handle_metadata,
        )

    def _failed_handle(
        self,
        batch: Batch,
        *,
        store: CoordinatorStore,
        cache_result: str,
        reason: str,
        metadata: dict[str, str] | None = None,
    ) -> BatchHandle:
        dataset = store.get_dataset(batch.dataset_id)

        handle_metadata = {
            "cache_result": cache_result,
            "reason": reason,
        }
        handle_metadata.update(metadata or {})

        return BatchHandle(
            batch_id=batch.batch_id,
            cache_key=batch.cache_key,
            status=BatchHandleStatus.FAILED,
            payload_format=dataset.payload_format,
            dataset_format=dataset.dataset_format,
            metadata=handle_metadata,
        )

    def _with_cache_result(
        self,
        handle: BatchHandle,
        *,
        cache_result: str,
    ) -> BatchHandle:
        metadata = dict(handle.metadata or {})
        metadata["cache_result"] = cache_result

        return replace(
            handle,
            metadata=metadata,
        )


def _payload_format_from_entry(
    entry: CacheEntry,
    *,
    default: PayloadFormat,
) -> PayloadFormat:
    raw_value = entry.metadata.get("payload_format")

    if raw_value is None or raw_value == "":
        return default

    if isinstance(raw_value, PayloadFormat):
        return raw_value

    return PayloadFormat(str(raw_value))


def _int_metadata(
    metadata: dict[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = metadata.get(key)

    if value is None or value == "":
        return default

    return int(value)