from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from batchflow.cache.redis_store import RedisPayloadStore
from batchflow.common.core import (
    Batch,
    BatchHandle,
    BatchHandleStatus,
    CacheEntry,
    CacheStatus,
    Dataset,
    Job,
    JobStatus,
    MaterializeBatchTaskStatus,
    StorageClass,
    now_ms,
)
from batchflow.config.config_types import CoordinatorConfig, RedisConfig
from batchflow.coordinator.batch_coordinator import BatchCoordinator
from batchflow.coordinator.scheduler import BatchflowScheduler
from batchflow.coordinator.store import CoordinatorStore, MaterializeBatchTaskState


LOGGER = logging.getLogger("batchflow.coordinator.service")


@dataclass(frozen=True)
class TrainerRuntimeFeedback:
    data_bottleneck_percent: float = 0.0
    avg_data_time_sec: float = 0.0
    avg_compute_time_sec: float = 0.0
    avg_coordinator_wait_total_time_sec: float = 0.0
    avg_coordinator_pending_polls: float = 0.0


@dataclass(frozen=True)
class StartJobResult:
    job_id: str
    epoch: int
    next_batch_index: int


@dataclass(frozen=True)
class GetNextBatchResult:
    batch_handle: BatchHandle
    epoch: int = 0
    batch_index: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FinishJobResult:
    job_id: str
    status: JobStatus
    reason: str


class CoordinatorService:
    def __init__(
        self,
        *,
        config: CoordinatorConfig,
        redis_config: RedisConfig | None = None,
        maintainer_interval_seconds: float = 0.01,
    ) -> None:
        self.store = CoordinatorStore()
        self.batch_coordinator = BatchCoordinator(default_lookahead=config.default_lookahead)
        self.scheduler = BatchflowScheduler(config=config.scheduler)

        self.redis_payload_store: RedisPayloadStore | None = None
        redis_config = redis_config or RedisConfig()

        if redis_config.enabled:
            self.redis_payload_store = RedisPayloadStore(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                ssl=redis_config.ssl,
                password=redis_config.password or None,
                key_prefix=redis_config.key_prefix,
            )

        self._cache_management_lock = threading.RLock()
        self._maintainer_interval_seconds = maintainer_interval_seconds
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()

        self._maintainer_thread = threading.Thread(
            target=self._maintainer_loop,
            name="batchflow-coordinator-maintainer",
            daemon=True,
        )
        self._maintainer_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._maintainer_thread.join(timeout=5.0)

        if self.redis_payload_store is not None:
            self.redis_payload_store.close()

    def register_dataset(self, dataset: Dataset) -> None:
        self.store.register_dataset(dataset)

    def start_job(
        self,
        *,
        job_id: str,
        dataset_id: str,
        lookahead_batches: int = 32,
        metadata: dict[str, Any] | None = None,
    ) -> StartJobResult:
        job = Job.create(
            job_id=job_id,
            dataset_id=dataset_id,
            lookahead_batches=lookahead_batches,
            metadata=metadata or {},
        )

        self.store.start_job(job)
        self._refresh_planned_window_if_needed(job.job_id, force=True)
        self._maintain_job(job.job_id)
        self._wake_event.set()

        return StartJobResult(
            job_id=job.job_id,
            epoch=job.progress.epoch,
            next_batch_index=job.progress.next_batch_index,
        )

    def get_next_batch(
        self,
        *,
        job_id: str,
        runtime_feedback: TrainerRuntimeFeedback | None = None,
    ) -> GetNextBatchResult:
        if runtime_feedback is not None:
            self.store.update_job_runtime_metrics(
                job_id,
                data_bottleneck_percent=runtime_feedback.data_bottleneck_percent,
                avg_data_time_sec=runtime_feedback.avg_data_time_sec,
                avg_compute_time_sec=runtime_feedback.avg_compute_time_sec,
                avg_coordinator_wait_total_time_sec=runtime_feedback.avg_coordinator_wait_total_time_sec,
                avg_coordinator_pending_polls=runtime_feedback.avg_coordinator_pending_polls,
            )

        job = self.store.get_job(job_id)
        self.store.note_get_next_batch_call(job_id)
        dataset = self.store.get_dataset(job.dataset_id)

        if job.status != JobStatus.ACTIVE:
            handle = BatchHandle(
                batch_id="",
                cache_key="",
                status=BatchHandleStatus.FAILED,
                payload_format=dataset.payload_format,
                dataset_format=dataset.dataset_format,
                metadata={"reason": f"job is not active: {job.status.value}"},
            )
            return GetNextBatchResult(
                batch_handle=handle,
                epoch=job.progress.epoch,
                batch_index=None,
                metadata={"done": True},
            )

        batch = self.store.get_batch(
            job_id=job.job_id,
            epoch=job.progress.epoch,
            batch_index=job.progress.next_batch_index,
        )

        if batch is None:
            self._refresh_planned_window_if_needed(job.job_id, force=True)
            batch = self.store.get_batch(
                job_id=job.job_id,
                epoch=job.progress.epoch,
                batch_index=job.progress.next_batch_index,
            )

        if batch is None:
            handle = BatchHandle(
                batch_id="",
                cache_key="",
                status=BatchHandleStatus.FAILED,
                payload_format=dataset.payload_format,
                dataset_format=dataset.dataset_format,
                metadata={"reason": "no planned batch available"},
            )
            return GetNextBatchResult(
                batch_handle=handle,
                epoch=job.progress.epoch,
                batch_index=None,
                metadata={"done": True},
            )

        handle = self.scheduler.resolve_batch_handle(batch, self.store)
        self.store.note_batch_handle_served(job_id, handle)

        if handle.is_ready:
            self.store.touch_cache_entry(handle.cache_key)
            self.store.pin_cache_entry(handle.cache_key, job_id)

            job = self.store.issue_batch_to_job(
                job_id=job_id,
                batch_id=batch.batch_id,
                epoch=batch.epoch,
                batch_index=batch.batch_index,
            )

            dataset = self.store.get_dataset(job.dataset_id)

            if job.progress.next_batch_index >= dataset.batches_per_epoch:
                job = self.store.update_job_progress(
                    job.job_id,
                    job.progress.advance_to_next_epoch(),
                )
                self.store.set_planned_window(job.job_id, [])

            self._refresh_planned_window_if_needed(job.job_id, force=True)

        self._wake_event.set()

        return GetNextBatchResult(
            batch_handle=handle,
            epoch=batch.epoch,
            batch_index=batch.batch_index,
            metadata={"job_id": job_id},
        )

    def commit_batch(
        self,
        *,
        job_id: str,
        batch_id: str,
        epoch: int,
        batch_index: int,
    ) -> Job:
        job = self.store.get_job(job_id)

        if epoch > job.progress.epoch:
            raise ValueError(
                f"commit epoch is ahead of job progress: "
                f"job_id={job_id} current={job.progress.epoch} got={epoch}"
            )

        job = self.store.commit_batch(
            job_id=job_id,
            batch_id=batch_id,
            batch_index=batch_index,
        )

        self.store.unpin_cache_entry(batch_id, job_id)
        self._wake_event.set()
        return job

    def finish_job(
        self,
        *,
        job_id: str,
        reason: str = "job finished",
        status: JobStatus = JobStatus.COMPLETED,
    ) -> FinishJobResult:
        job = self.store.finish_job(job_id, reason=reason, status=status)

        self.store.release_job_cache_pins(job_id)
        self.store.set_planned_window(job_id, [])
        self._wake_event.set()

        return FinishJobResult(
            job_id=job.job_id,
            status=job.status,
            reason=job.closed_reason,
        )

    def register_worker(
        self,
        *,
        worker_id: str,
        hostname: str,
        metadata: dict[str, Any] | None = None,
    ):
        return self.store.register_worker(
            worker_id=worker_id,
            hostname=hostname,
            metadata=metadata,
        )

    def heartbeat_worker(self, worker_id: str):
        return self.store.heartbeat_worker(worker_id)

    def poll_materialize_batch_task(
        self,
        *,
        worker_id: str,
    ) -> MaterializeBatchTaskState | None:
        self.store.heartbeat_worker(worker_id)

        task_state = self.scheduler.choose_next_task_for_worker(worker_id, self.store)
        if task_state is None:
            return None

        if task_state.status == MaterializeBatchTaskStatus.PENDING:
            task_state = self.store.assign_task_to_worker(
                task_state.task.task_id,
                worker_id,
            )

        return task_state

    def complete_materialize_batch_task(
        self,
        *,
        task_id: str,
        worker_id: str,
        location: str,
        size_bytes: int,
        storage_class: StorageClass,
        expires_at_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MaterializeBatchTaskState:
        task_state = self.store.get_task(task_id)
        batch = task_state.task.batch
        metadata = dict(metadata or {})

        raw_materialize_time = metadata.get("materialize_time_sec")
        if raw_materialize_time not in (None, ""):
            try:
                self.store.update_preparation_time(
                    batch.dataset_id,
                    float(raw_materialize_time),
                    alpha=self.scheduler.config.preparation_time_ema_alpha,
                )
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Ignoring invalid materialize_time_sec task_id=%s value=%r",
                    task_id,
                    raw_materialize_time,
                )

        self.store.update_payload_size_estimate(
            batch.dataset_id,
            int(size_bytes),
            alpha=self.scheduler.config.preparation_time_ema_alpha,
        )

        final_location = location
        final_storage_class = storage_class
        final_expires_at_ms = expires_at_ms
        final_metadata = metadata

        is_shared_redis_result = (
            storage_class == StorageClass.REUSABLE
            and location.startswith(("redis://", "rediss://"))
            and self.redis_payload_store is not None
        )

        if is_shared_redis_result:
            with self._cache_management_lock:
                (
                    final_location,
                    final_storage_class,
                    final_expires_at_ms,
                    final_metadata,
                ) = self._apply_cache_admission(
                    batch=batch,
                    size_bytes=size_bytes,
                    location=location,
                    expires_at_ms=expires_at_ms,
                    metadata=metadata,
                )

        entry = CacheEntry(
            cache_key=batch.cache_key,
            status=CacheStatus.AVAILABLE,
            storage_class=final_storage_class,
            location=final_location,
            size_bytes=size_bytes,
            worker_id=worker_id,
            created_at_ms=now_ms(),
            last_accessed_at_ms=now_ms(),
            expires_at_ms=final_expires_at_ms,
            metadata={
                "batch_id": batch.batch_id,
                "dataset_id": batch.dataset_id,
                "epoch": batch.epoch,
                "batch_index": batch.batch_index,
                **final_metadata,
            },
        )

        handle = self.scheduler.build_ready_handle(batch=batch, entry=entry, store=self.store)
        task_state = self.store.complete_task(
            task_id,
            cache_entry=entry,
            batch_handle=handle,
        )

        self._wake_event.set()
        return task_state

    def _apply_cache_admission(
        self,
        *,
        batch: Batch,
        size_bytes: int,
        location: str,
        expires_at_ms: int | None,
        metadata: dict[str, Any],
    ) -> tuple[str, StorageClass, int | None, dict[str, Any]]:
        assert self.redis_payload_store is not None

        metadata = dict(metadata)
        incoming_value = self.scheduler.compute_batch_value(batch, self.store)
        metadata["batch_value"] = f"{incoming_value:.12g}"

        victims = self.scheduler.choose_cache_evictions(
            incoming_batch=batch,
            incoming_size_bytes=size_bytes,
            store=self.store,
        )

        capacity = int(self.scheduler.config.cache_capacity_bytes)

        if victims is not None:
            for victim in victims:
                evicted = self.store.evict_batch(victim.cache_key)
                if evicted is None:
                    continue

                redis_key = str(evicted.metadata.get("fetch_key", ""))
                if redis_key:
                    try:
                        self.redis_payload_store.remove_concrete_key(key=redis_key)
                    except Exception:
                        LOGGER.exception(
                            "Failed to delete evicted Redis payload cache_key=%s key=%s",
                            evicted.cache_key,
                            redis_key,
                        )

                LOGGER.info(
                    "Evicted reusable cache entry cache_key=%s size_bytes=%s value=%.6f",
                    evicted.cache_key,
                    evicted.size_bytes,
                    self.scheduler.compute_cache_entry_value(evicted, self.store),
                )

            if capacity <= 0 or self.store.reusable_cache_bytes() + int(size_bytes) <= capacity:
                metadata["cache_admission"] = "admitted"
                return location, StorageClass.REUSABLE, expires_at_ms, metadata

        redis_key = str(metadata.get("fetch_key", ""))
        if redis_key:
            try:
                self.redis_payload_store.remove_concrete_key(key=redis_key)
            except Exception:
                LOGGER.exception(
                    "Failed to delete rejected Redis payload batch_id=%s key=%s",
                    batch.batch_id,
                    redis_key,
                )

        fallback_host = str(metadata.get("fallback_fetch_host", ""))
        fallback_port = _safe_int(metadata.get("fallback_fetch_port"), default=0)
        fallback_key = str(metadata.get("fallback_fetch_key", ""))
        fallback_expires_at_ms = _safe_int(
            metadata.get("fallback_expires_at_ms"),
            default=0,
        )

        if not fallback_host or fallback_port <= 0 or not fallback_key:
            LOGGER.warning(
                "Cannot reject Redis cache admission because worker fallback "
                "metadata is missing batch_id=%s; retaining object",
                batch.batch_id,
            )
            metadata["cache_admission"] = "forced_no_fallback"
            return location, StorageClass.REUSABLE, expires_at_ms, metadata

        fallback_metadata = dict(metadata)
        fallback_metadata.update(
            {
                "fetch_host": fallback_host,
                "fetch_port": str(fallback_port),
                "fetch_key": fallback_key,
                "cache_admission": "rejected",
            }
        )

        return (
            f"grpc://{fallback_host}:{fallback_port}",
            StorageClass.TRANSIENT,
            fallback_expires_at_ms or None,
            fallback_metadata,
        )

    def fail_materialize_batch_task(
        self,
        *,
        task_id: str,
        reason: str,
    ) -> MaterializeBatchTaskState:
        task_state = self.store.fail_task(task_id, reason)
        self._wake_event.set()
        return task_state

    def _maintainer_loop(self) -> None:
        while not self._stop_event.is_set():
            did_work = self._cleanup_expired_cache_entries()

            for job in self.store.list_active_jobs():
                try:
                    if self._maintain_job(job.job_id):
                        did_work = True
                except Exception:
                    LOGGER.exception(
                        "Coordinator maintainer failed for job_id=%s",
                        job.job_id,
                    )

            try:
                if self._schedule_opportunistic_prefetch():
                    did_work = True
            except Exception:
                LOGGER.exception("Coordinator opportunistic-prefetch pass failed")

            if did_work:
                continue

            self._wake_event.wait(timeout=self._maintainer_interval_seconds)
            self._wake_event.clear()

    def _cleanup_expired_cache_entries(self) -> bool:
        current_ms = now_ms()
        removed = False

        for entry in self.store.list_cache_entries():
            if not entry.is_expired(current_ms):
                continue
            if self.store.cache_pin_count(entry.cache_key) > 0:
                continue

            if self.store.evict_batch(entry.cache_key) is not None:
                removed = True

        return removed

    def _maintain_job(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)

        if job.status != JobStatus.ACTIVE:
            self.store.set_planned_window(job_id, [])
            return False

        self._refresh_planned_window_if_needed(job_id)
        planned = self.store.get_planned_window(job_id)

        if not planned:
            return False

        target_depth = self.scheduler.target_lookahead_depth(job, self.store)
        considered = 0
        did_work = False

        for batch in planned:
            if batch.epoch != job.progress.epoch:
                continue
            if batch.batch_index < job.progress.next_batch_index:
                continue
            if considered >= target_depth:
                break

            considered += 1

            if self.store.is_batch_ready(batch.batch_id):
                continue
            if self.store.is_batch_in_progress(batch.batch_id):
                continue

            if self.scheduler.should_materialize(batch, self.store):
                task = self.scheduler.build_task(batch, job=job, store=self.store)
                self.store.create_task(task)
                did_work = True

        return did_work

    def _schedule_opportunistic_prefetch(self) -> bool:
        if not self.scheduler.config.reuse_enabled:
            return False
        if not self.scheduler.config.opportunistic_prefetch_enabled:
            return False
        if self.redis_payload_store is None:
            return False

        active_jobs = self.store.list_active_jobs()
        if not active_jobs:
            return False

        # Immediate trainer demand must be covered before spare workers prefetch.
        for job in active_jobs:
            target_depth = self.scheduler.target_lookahead_depth(job, self.store)
            covered = self.store.count_ready_or_inflight_in_window(
                job.job_id,
                epoch=job.progress.epoch,
                start_batch_index=job.progress.next_batch_index,
                limit=target_depth,
            )
            if covered < target_depth:
                return False

        idle_workers = self.store.count_idle_workers()
        if idle_workers <= 0:
            return False

        spare_slots = max(0, idle_workers - self.store.count_pending_tasks())
        if spare_slots <= 0:
            return False

        multiplier = max(1, int(self.scheduler.config.extended_lookahead_multiplier))
        candidates_by_cache_key: dict[str, tuple[float, Batch]] = {}

        for job in active_jobs:
            target_depth = self.scheduler.target_lookahead_depth(job, self.store)
            extended_depth = max(target_depth, multiplier * target_depth)
            upcoming = self.store.next_uncommitted_batches(job.job_id)

            for batch in upcoming[target_depth:extended_depth]:
                if not self.scheduler.can_opportunistically_prefetch(
                    batch,
                    store=self.store,
                ):
                    continue

                if batch.cache_key in candidates_by_cache_key:
                    continue

                value = self.scheduler.score_opportunistic_batch(batch, store=self.store)
                if value <= 0.0:
                    continue

                candidates_by_cache_key[batch.cache_key] = (value, batch)

        if not candidates_by_cache_key:
            return False

        candidates = sorted(
            candidates_by_cache_key.values(),
            key=lambda item: (item[0], item[1].cache_key),
            reverse=True,
        )

        scheduled = 0

        for value, batch in candidates:
            if scheduled >= spare_slots:
                break

            if not self.scheduler.can_opportunistically_prefetch(
                batch,
                store=self.store,
            ):
                continue

            task = self.scheduler.build_opportunistic_task(batch, store=self.store)
            state = self.store.create_task(task)

            if state.task.task_id != task.task_id:
                continue

            scheduled += 1
            LOGGER.debug(
                "Scheduled opportunistic prefetch batch_id=%s cache_key=%s "
                "value=%.6f slot=%s/%s",
                batch.batch_id,
                batch.cache_key,
                value,
                scheduled,
                spare_slots,
            )

        return scheduled > 0

    def _refresh_planned_window_if_needed(
        self,
        job_id: str,
        *,
        force: bool = False,
    ) -> None:
        job = self.store.get_job(job_id)

        if job.status != JobStatus.ACTIVE:
            self.store.set_planned_window(job_id, [])
            return

        lookahead = max(
            1,
            job.lookahead_batches or self.batch_coordinator.default_lookahead,
        )

        planned = self.store.get_planned_window(job_id)
        future_batches = [
            batch
            for batch in planned
            if batch.epoch == job.progress.epoch
            and batch.batch_index >= job.progress.next_batch_index
        ]

        if not force and len(future_batches) >= lookahead:
            return

        dataset = self.store.get_dataset(job.dataset_id)

        new_batches = self.batch_coordinator.get_batches_for_job(
            dataset=dataset,
            job=job,
            reuse_enabled=self.scheduler.config.reuse_enabled,
        )

        self.store.set_planned_window(job_id, new_batches)


def _safe_int(value: object, *, default: int) -> int:
    if value in (None, ""):
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default