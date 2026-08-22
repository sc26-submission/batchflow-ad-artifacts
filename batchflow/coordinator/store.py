from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any

from batchflow.common.core import (
    Batch,
    BatchHandle,
    BatchHandleStatus,
    PayloadEntry,
    PayloadStatus,
    Dataset,
    Job,
    JobProgress,
    JobStatus,
    MaterializeBatchTask,
    MaterializeBatchTaskStatus,
    Worker,
    now_ms,
)


@dataclass
class MaterializeBatchTaskState:
    task: MaterializeBatchTask
    status: MaterializeBatchTaskStatus = MaterializeBatchTaskStatus.PENDING
    assigned_worker_id: str | None = None
    created_at_ms: int = field(default_factory=now_ms)
    assigned_at_ms: int | None = None
    completed_at_ms: int | None = None
    failed_at_ms: int | None = None
    failure_reason: str = ""

    def mark_running(self, worker_id: str) -> None:
        self.status = MaterializeBatchTaskStatus.RUNNING
        self.assigned_worker_id = worker_id
        self.assigned_at_ms = now_ms()

    def mark_completed(self) -> None:
        self.status = MaterializeBatchTaskStatus.COMPLETED
        self.completed_at_ms = now_ms()

    def mark_failed(self, reason: str) -> None:
        self.status = MaterializeBatchTaskStatus.FAILED
        self.failed_at_ms = now_ms()
        self.failure_reason = reason

    def mark_cancelled(self, reason: str) -> None:
        self.status = MaterializeBatchTaskStatus.CANCELLED
        self.failed_at_ms = now_ms()
        self.failure_reason = reason


@dataclass
class JobMetrics:
    get_next_batch_calls: int = 0
    pending_batch_requests: int = 0
    ready_responses: int = 0
    pending_responses: int = 0
    failed_responses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_in_flight: int = 0
    cache_failures: int = 0
    materializations: int = 0
    last_acknowledged_at_ms: int | None = None

    data_bottleneck_percent: float = 0.0
    avg_data_time_sec: float = 0.0
    avg_compute_time_sec: float = 0.0
    avg_coordinator_wait_total_time_sec: float = 0.0
    avg_coordinator_pending_polls: float = 0.0
    last_runtime_metrics_update_ms: int = 0


class CoordinatorStore:
    """In-memory control-plane state for jobs, workers, tasks, and payload metadata."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._datasets: dict[str, Dataset] = {}
        self._jobs: dict[str, Job] = {}
        self._planned_batches: dict[str, list[Batch]] = {}

        self._tasks: dict[str, MaterializeBatchTaskState] = {}
        self._tasks_by_batch_id: dict[str, str] = {}

        self._payload_entries: dict[str, PayloadEntry] = {}
        self._batch_handles: dict[str, BatchHandle] = {}

        self._workers: dict[str, Worker] = {}
        self._job_metrics: dict[str, JobMetrics] = {}

        # Online preparation-time and payload-size estimates used by the
        # batch-value and opportunistic-prefetch policies.
        self._preparation_time_ema: dict[str, float] = {}
        self._payload_size_ema: dict[str, float] = {}

        # A payload stays pinned while a client is fetching it. Once the client
        # acknowledges the fetch, the payload can participate in eviction again.
        self._payload_pins: dict[str, set[str]] = {}
        self._job_payload_pins: dict[str, set[str]] = {}

    def register_dataset(self, dataset: Dataset) -> None:
        with self._lock:
            self._datasets[dataset.dataset_id] = dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        with self._lock:
            if dataset_id not in self._datasets:
                raise KeyError(f"unknown dataset: {dataset_id}")
            return self._datasets[dataset_id]

    def list_datasets(self) -> list[Dataset]:
        with self._lock:
            return list(self._datasets.values())

    def start_job(self, job: Job) -> None:
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"job already exists: {job.job_id}")

            if job.dataset_id not in self._datasets:
                raise KeyError(f"unknown dataset: {job.dataset_id}")

            self._jobs[job.job_id] = job
            self._planned_batches[job.job_id] = []
            self._job_metrics[job.job_id] = JobMetrics()

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job: {job_id}")
            return self._jobs[job_id]

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def list_active_jobs(self) -> list[Job]:
        with self._lock:
            return [
                job
                for job in self._jobs.values()
                if job.status == JobStatus.ACTIVE
            ]

    def is_job_active(self, job_id: str) -> bool:
        with self._lock:
            return self.get_job(job_id).status == JobStatus.ACTIVE

    def finish_job(
        self,
        job_id: str,
        *,
        reason: str,
        status: JobStatus = JobStatus.COMPLETED,
    ) -> Job:
        with self._lock:
            job = self.get_job(job_id)

            if job.status != JobStatus.ACTIVE:
                return job

            job.status = status
            job.closed_reason = reason

            for task_state in self._tasks.values():
                if task_state.task.job_id != job_id:
                    continue

                if task_state.status == MaterializeBatchTaskStatus.PENDING:
                    task_state.mark_cancelled(f"job finished: {reason}")

            return job

    def update_job_progress(
        self,
        job_id: str,
        progress: JobProgress,
    ) -> Job:
        with self._lock:
            job = self.get_job(job_id)
            job.progress = progress
            return job

    def issue_batch_to_job(
        self,
        *,
        job_id: str,
        batch_id: str,
        epoch: int,
        batch_index: int,
    ) -> Job:
        """
        Mark a READY batch as issued to a job.

        This advances next_batch_index when GetNextBatch returns a READY handle.
        CommitBatch should only confirm consumption, not advance next_batch_index.
        """
        with self._lock:
            job = self.get_job(job_id)

            if job.status != JobStatus.ACTIVE:
                return job

            if job.progress.epoch != epoch:
                return job

            if job.progress.next_batch_index != batch_index:
                return job

            job.progress = replace(
                job.progress,
                next_batch_index=batch_index + 1,
            )

            return job

    def get_job_metrics(self, job_id: str) -> JobMetrics:
        with self._lock:
            if job_id not in self._job_metrics:
                raise KeyError(f"unknown job metrics: {job_id}")
            return self._job_metrics[job_id]

    def note_get_next_batch_call(self, job_id: str) -> None:
        with self._lock:
            metrics = self.get_job_metrics(job_id)
            metrics.get_next_batch_calls += 1
            metrics.pending_batch_requests += 1

    def note_batch_handle_served(
        self,
        job_id: str,
        handle: BatchHandle,
    ) -> None:
        with self._lock:
            metrics = self.get_job_metrics(job_id)
            metrics.pending_batch_requests = max(
                0,
                metrics.pending_batch_requests - 1,
            )

            if handle.status == BatchHandleStatus.READY:
                metrics.ready_responses += 1
            elif handle.status == BatchHandleStatus.PENDING:
                metrics.pending_responses += 1
            elif handle.status == BatchHandleStatus.FAILED:
                metrics.failed_responses += 1

            cache_result = str(handle.metadata.get("cache_result", ""))

            if cache_result == "hit":
                metrics.cache_hits += 1
            elif cache_result == "miss":
                metrics.cache_misses += 1
            elif cache_result == "in_flight":
                metrics.cache_in_flight += 1
            elif cache_result == "failed":
                metrics.cache_failures += 1

    def note_materialization_completed(self, job_id: str) -> None:
        with self._lock:
            metrics = self.get_job_metrics(job_id)
            metrics.materializations += 1

    def update_job_runtime_metrics(
        self,
        job_id: str,
        *,
        data_bottleneck_percent: float,
        avg_data_time_sec: float,
        avg_compute_time_sec: float,
        avg_coordinator_wait_total_time_sec: float,
        avg_coordinator_pending_polls: float,
    ) -> None:
        with self._lock:
            metrics = self.get_job_metrics(job_id)
            metrics.data_bottleneck_percent = float(data_bottleneck_percent)
            metrics.avg_data_time_sec = float(avg_data_time_sec)
            metrics.avg_compute_time_sec = float(avg_compute_time_sec)
            metrics.avg_coordinator_wait_total_time_sec = float(
                avg_coordinator_wait_total_time_sec
            )
            metrics.avg_coordinator_pending_polls = float(
                avg_coordinator_pending_polls
            )
            metrics.last_runtime_metrics_update_ms = now_ms()

    def update_preparation_time(
        self,
        dataset_id: str,
        duration_sec: float,
        *,
        alpha: float = 0.2,
    ) -> None:
        if duration_sec <= 0:
            return

        alpha = min(1.0, max(0.0, float(alpha)))

        with self._lock:
            previous = self._preparation_time_ema.get(dataset_id)

            if previous is None:
                self._preparation_time_ema[dataset_id] = float(duration_sec)
            else:
                self._preparation_time_ema[dataset_id] = (
                    alpha * float(duration_sec)
                    + (1.0 - alpha) * previous
                )

    def get_preparation_time_estimate(
        self,
        dataset_id: str,
        *,
        default: float = 1.0,
    ) -> float:
        with self._lock:
            return float(self._preparation_time_ema.get(dataset_id, default))

    def update_payload_size_estimate(
        self,
        dataset_id: str,
        size_bytes: int,
        *,
        alpha: float = 0.2,
    ) -> None:
        if size_bytes <= 0:
            return

        alpha = min(1.0, max(0.0, float(alpha)))

        with self._lock:
            previous = self._payload_size_ema.get(dataset_id)

            if previous is None:
                self._payload_size_ema[dataset_id] = float(size_bytes)
            else:
                self._payload_size_ema[dataset_id] = (
                    alpha * float(size_bytes)
                    + (1.0 - alpha) * previous
                )

    def get_payload_size_estimate(
        self,
        dataset_id: str,
        *,
        default: float = 0.0,
    ) -> float:
        with self._lock:
            return float(self._payload_size_ema.get(dataset_id, default))

    def set_planned_window(
        self,
        job_id: str,
        batches: list[Batch],
    ) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job: {job_id}")
            self._planned_batches[job_id] = list(batches)

    def get_planned_window(self, job_id: str) -> list[Batch]:
        with self._lock:
            if job_id not in self._planned_batches:
                raise KeyError(f"unknown job: {job_id}")
            return list(self._planned_batches[job_id])

    def get_batch(
        self,
        job_id: str,
        epoch: int,
        batch_index: int,
    ) -> Batch | None:
        with self._lock:
            for batch in self._planned_batches.get(job_id, []):
                if batch.epoch == epoch and batch.batch_index == batch_index:
                    return batch
            return None

    def upcoming_batches(self, job_id: str) -> list[Batch]:
        with self._lock:
            job = self.get_job(job_id)

            return [
                batch
                for batch in self._planned_batches.get(job_id, [])
                if batch.epoch == job.progress.epoch
                and batch.batch_index >= job.progress.next_batch_index
            ]

    def count_ready_or_inflight_from_index(
        self,
        job_id: str,
        *,
        epoch: int,
        start_batch_index: int,
    ) -> int:
        with self._lock:
            count = 0

            for batch in self._planned_batches.get(job_id, []):
                if batch.epoch != epoch:
                    continue

                if batch.batch_index < start_batch_index:
                    continue

                if self.is_batch_ready(batch.batch_id) or self.is_batch_in_progress(
                    batch.batch_id,
                ):
                    count += 1

            return count

    def count_ready_or_inflight_in_window(
        self,
        job_id: str,
        *,
        epoch: int,
        start_batch_index: int,
        limit: int,
    ) -> int:
        if limit <= 0:
            return 0

        with self._lock:
            count = 0
            considered = 0

            for batch in self._planned_batches.get(job_id, []):
                if batch.epoch != epoch:
                    continue
                if batch.batch_index < start_batch_index:
                    continue
                if considered >= limit:
                    break

                considered += 1

                if self.is_batch_ready(batch.batch_id) or self.is_batch_in_progress(
                    batch.batch_id
                ):
                    count += 1

            return count

    def acknowledge_batch(
        self,
        *,
        job_id: str,
        batch_id: str,
        batch_index: int,
    ) -> Job:
        """
        Record that the client successfully fetched a previously issued batch.

        This does not advance next_batch_index. Progress advances when a READY
        handle is issued by GetNextBatch; acknowledgement only releases the pin.
        """
        with self._lock:
            job = self.get_job(job_id)

            job.progress = replace(
                job.progress,
                last_batch_id=batch_id,
                last_batch_index=batch_index,
            )

            metrics = self.get_job_metrics(job_id)
            metrics.last_acknowledged_at_ms = now_ms()

            return job

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        metadata: dict[str, Any] | None = None,
    ) -> Worker:
        with self._lock:
            worker = Worker(
                worker_id=worker_id,
                hostname=hostname,
                metadata=metadata or {},
            )
            self._workers[worker_id] = worker
            return worker

    def heartbeat_worker(self, worker_id: str) -> Worker:
        with self._lock:
            if worker_id not in self._workers:
                raise KeyError(f"unknown worker: {worker_id}")

            worker = self._workers[worker_id]
            worker.touch()
            return worker

    def get_worker(self, worker_id: str) -> Worker:
        with self._lock:
            if worker_id not in self._workers:
                raise KeyError(f"unknown worker: {worker_id}")
            return self._workers[worker_id]

    def list_workers(self) -> list[Worker]:
        with self._lock:
            return list(self._workers.values())

    def count_idle_workers(self) -> int:
        """Return workers that are not currently executing a task."""
        with self._lock:
            return sum(
                1
                for worker in self._workers.values()
                if not worker.active_task_ids
            )

    def count_pending_tasks(self) -> int:
        """Return materialization tasks waiting for a worker."""
        with self._lock:
            return sum(
                1
                for task_state in self._tasks.values()
                if task_state.status == MaterializeBatchTaskStatus.PENDING
            )

    def create_task(
        self,
        task: MaterializeBatchTask,
    ) -> MaterializeBatchTaskState:
        with self._lock:
            job_id = task.job_id

            if job_id and not self.is_job_active(job_id):
                state = MaterializeBatchTaskState(task=task)
                state.mark_cancelled("job is not active")
                return state

            existing_task_id = self._tasks_by_batch_id.get(task.batch.batch_id)

            if existing_task_id is not None:
                existing_state = self._tasks[existing_task_id]

                if existing_state.status in (
                    MaterializeBatchTaskStatus.PENDING,
                    MaterializeBatchTaskStatus.RUNNING,
                    MaterializeBatchTaskStatus.COMPLETED,
                ):
                    return existing_state

                if existing_state.status in (
                    MaterializeBatchTaskStatus.FAILED,
                    MaterializeBatchTaskStatus.CANCELLED,
                ):
                    del self._tasks_by_batch_id[task.batch.batch_id]

            state = MaterializeBatchTaskState(task=task)

            self._tasks[task.task_id] = state
            self._tasks_by_batch_id[task.batch.batch_id] = task.task_id

            if task.batch.cache_key not in self._payload_entries:
                self._payload_entries[task.batch.cache_key] = PayloadEntry(
                    cache_key=task.batch.cache_key,
                    status=PayloadStatus.PENDING,
                    storage_class=task.storage_class,
                    metadata={
                        "batch_id": task.batch.batch_id,
                        "job_id": job_id,
                        "dataset_id": task.batch.dataset_id,
                        "epoch": task.batch.epoch,
                        "batch_index": task.batch.batch_index,
                    },
                )
            else:
                entry = self._payload_entries[task.batch.cache_key]

                if entry.status == PayloadStatus.FAILED:
                    entry.status = PayloadStatus.PENDING
                    entry.metadata.pop("failure_reason", None)

            return state

    def get_task(self, task_id: str) -> MaterializeBatchTaskState:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id}")
            return self._tasks[task_id]

    def get_task_for_batch(self, batch_id: str) -> MaterializeBatchTaskState | None:
        with self._lock:
            task_id = self._tasks_by_batch_id.get(batch_id)
            if task_id is None:
                return None
            return self._tasks[task_id]

    def list_pending_tasks(self) -> list[MaterializeBatchTaskState]:
        with self._lock:
            return [
                task_state
                for task_state in self._tasks.values()
                if task_state.status == MaterializeBatchTaskStatus.PENDING
            ]

    def assign_task_to_worker(
        self,
        task_id: str,
        worker_id: str,
    ) -> MaterializeBatchTaskState:
        with self._lock:
            task_state = self.get_task(task_id)
            worker = self.get_worker(worker_id)

            if task_state.status != MaterializeBatchTaskStatus.PENDING:
                return task_state

            job_id = task_state.task.job_id

            if job_id and not self.is_job_active(job_id):
                task_state.mark_cancelled("job closed before assignment")
                return task_state

            task_state.mark_running(worker_id)
            worker.active_task_ids.add(task_id)
            return task_state

    def complete_task(
        self,
        task_id: str,
        *,
        payload_entry: PayloadEntry,
        batch_handle: BatchHandle,
    ) -> MaterializeBatchTaskState:
        with self._lock:
            task_state = self.get_task(task_id)
            task_state.mark_completed()

            if task_state.assigned_worker_id is not None:
                worker = self._workers.get(task_state.assigned_worker_id)
                if worker is not None:
                    worker.active_task_ids.discard(task_id)

            self._payload_entries[payload_entry.cache_key] = payload_entry
            self._batch_handles[task_state.task.batch.batch_id] = batch_handle

            job_id = task_state.task.job_id
            if job_id:
                self.note_materialization_completed(job_id)

            return task_state

    def fail_task(self, task_id: str, reason: str) -> MaterializeBatchTaskState:
        with self._lock:
            task_state = self.get_task(task_id)
            task_state.mark_failed(reason)

            if task_state.assigned_worker_id is not None:
                worker = self._workers.get(task_state.assigned_worker_id)
                if worker is not None:
                    worker.active_task_ids.discard(task_id)

            cache_key = task_state.task.batch.cache_key
            entry = self._payload_entries.get(cache_key)

            if entry is not None:
                entry.status = PayloadStatus.FAILED
                entry.metadata["failure_reason"] = reason

            return task_state

    def list_payload_entries(self) -> list[PayloadEntry]:
        with self._lock:
            return list(self._payload_entries.values())

    def list_reusable_payload_entries(self) -> list[PayloadEntry]:
        with self._lock:
            return [
                entry
                for entry in self._payload_entries.values()
                if entry.status == PayloadStatus.AVAILABLE
                and entry.storage_class.value == "reusable"
                and (entry.location or "").startswith(("redis://", "rediss://"))
            ]

    def reusable_cache_bytes(self) -> int:
        with self._lock:
            return sum(
                max(0, int(entry.size_bytes))
                for entry in self._payload_entries.values()
                if entry.status == PayloadStatus.AVAILABLE
                and entry.storage_class.value == "reusable"
                and (entry.location or "").startswith(("redis://", "rediss://"))
            )

    def pin_payload(self, cache_key: str, job_id: str) -> None:
        with self._lock:
            if cache_key not in self._payload_entries:
                return

            self._payload_pins.setdefault(cache_key, set()).add(job_id)
            self._job_payload_pins.setdefault(job_id, set()).add(cache_key)

    def unpin_payload(self, cache_key: str, job_id: str) -> None:
        with self._lock:
            jobs = self._payload_pins.get(cache_key)
            if jobs is not None:
                jobs.discard(job_id)
                if not jobs:
                    self._payload_pins.pop(cache_key, None)

            keys = self._job_payload_pins.get(job_id)
            if keys is not None:
                keys.discard(cache_key)
                if not keys:
                    self._job_payload_pins.pop(job_id, None)

    def payload_pin_count(self, cache_key: str) -> int:
        with self._lock:
            return len(self._payload_pins.get(cache_key, set()))

    def release_job_payload_pins(self, job_id: str) -> None:
        with self._lock:
            cache_keys = list(self._job_payload_pins.pop(job_id, set()))

            for cache_key in cache_keys:
                jobs = self._payload_pins.get(cache_key)
                if jobs is None:
                    continue
                jobs.discard(job_id)
                if not jobs:
                    self._payload_pins.pop(cache_key, None)

    def remove_payload(self, cache_key: str) -> PayloadEntry | None:
        """Remove coordinator state for a payload so it can be materialized again."""
        with self._lock:
            if self.payload_pin_count(cache_key) > 0:
                return None

            entry = self._payload_entries.pop(cache_key, None)
            if entry is None:
                return None

            batch_id = str(entry.metadata.get("batch_id", cache_key))
            self._batch_handles.pop(batch_id, None)

            task_id = self._tasks_by_batch_id.pop(batch_id, None)
            if task_id is not None:
                self._tasks.pop(task_id, None)

            self._payload_pins.pop(cache_key, None)
            for job_id, keys in list(self._job_payload_pins.items()):
                keys.discard(cache_key)
                if not keys:
                    self._job_payload_pins.pop(job_id, None)

            return entry

    def put_payload_entry(self, entry: PayloadEntry) -> None:
        with self._lock:
            self._payload_entries[entry.cache_key] = entry

    def get_payload_entry(self, cache_key: str) -> PayloadEntry | None:
        with self._lock:
            return self._payload_entries.get(cache_key)

    def touch_payload(self, cache_key: str) -> None:
        with self._lock:
            entry = self._payload_entries.get(cache_key)
            if entry is not None:
                entry.touch()

    def delete_payload_entry(self, cache_key: str) -> None:
        with self._lock:
            self._payload_entries.pop(cache_key, None)

    def put_batch_handle(self, batch_id: str, handle: BatchHandle) -> None:
        with self._lock:
            self._batch_handles[batch_id] = handle

    def get_batch_handle(self, batch_id: str) -> BatchHandle | None:
        with self._lock:
            return self._batch_handles.get(batch_id)

    def delete_batch_handle(self, batch_id: str) -> None:
        with self._lock:
            self._batch_handles.pop(batch_id, None)

    def is_batch_ready(self, batch_id: str) -> bool:
        with self._lock:
            return batch_id in self._batch_handles

    def is_batch_in_progress(self, batch_id: str) -> bool:
        with self._lock:
            task = self.get_task_for_batch(batch_id)
            return task is not None and task.status in (
                MaterializeBatchTaskStatus.PENDING,
                MaterializeBatchTaskStatus.RUNNING,
            )