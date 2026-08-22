from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from batchflow.clients.coordinator_client import CoordinatorGrpcClient
from batchflow.clients.redis_client import RedisFetchClient
from batchflow.clients.worker_client import WorkerFetchClient
from batchflow.integrations.pytorch.config import (
    BatchFlowTorchConfig,
    TrainerRuntimeMetricsBuffer,
)
from batchflow.integrations.pytorch.decoding import decode_payload, pin_memory_batch
from batchflow.proto import batchflow_pb2


LOGGER = logging.getLogger("batchflow.integrations.pytorch.prefetcher")


@dataclass
class BatchItem:
    batch: dict[str, Any]


@dataclass
class ErrorItem:
    error: BaseException


@dataclass
class EndItem:
    pass


@dataclass(frozen=True)
class FetchTask:
    sequence: int
    job_id: str
    batch_id: str
    cache_key: str
    epoch: int
    batch_index: int

    location: str
    fetch_host: str
    fetch_port: int
    fetch_key: str
    payload_format: str
    dataset_format: str

    handle_status: str
    cache_result: str
    client_cache_result: str

    coordinator_wait_total_time_sec: float
    coordinator_rpc_time_sec: float
    coordinator_sleep_time_sec: float
    coordinator_pending_polls: int
    coordinator_miss_polls: int
    coordinator_in_flight_polls: int


@dataclass
class FetchedTaskResult:
    task: FetchTask
    batch: dict[str, Any]
    payload_bytes: int
    fetch_time_sec: float
    decode_time_sec: float
    pin_time_sec: float


class MultiThreadBatchFlowPrefetcher:
    def __init__(
        self,
        config: BatchFlowTorchConfig,
        *,
        runtime_metrics: TrainerRuntimeMetricsBuffer | None = None,
    ) -> None:
        config.validate()

        self.config = config
        self.runtime_metrics = runtime_metrics

        queue_size = max(1, config.max_ready_batches)

        self.ready_queue: queue.Queue[BatchItem | ErrorItem | EndItem] = queue.Queue(
            maxsize=queue_size,
        )
        self.task_queue: queue.Queue[FetchTask | None] = queue.Queue(
            maxsize=queue_size,
        )
        self.fetched_queue: queue.Queue[FetchedTaskResult | ErrorItem | None] = (
            queue.Queue(maxsize=queue_size)
        )

        self._stop_event = threading.Event()

        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            name="batchflow-pytorch-coordinator",
            daemon=True,
        )
        self._publish_thread = threading.Thread(
            target=self._ordered_publish_loop,
            name="batchflow-pytorch-publish",
            daemon=True,
        )
        self._fetch_threads = [
            threading.Thread(
                target=self._fetch_worker_loop,
                name=f"batchflow-pytorch-fetch-{index}",
                daemon=True,
            )
            for index in range(max(1, config.parallel_fetch_workers))
        ]

        self._coordinator_client: CoordinatorGrpcClient | None = None
        self._job_id: str | None = None
        self._completed_normally = False

        self._coordinator_done = False
        self._end_sent = False

        self._scheduled_batches = 0
        self._published_batches = 0
        self._produced_batches = 0

        self._last_status_log_time = time.time()

    def start(self) -> None:
        for thread in self._fetch_threads:
            thread.start()

        self._publish_thread.start()
        self._coordinator_thread.start()

    def shutdown(self) -> None:
        self.stop()

        if self.config.finish_job_on_close:
            self.close_job()

    def stop(self) -> None:
        self._stop_event.set()

        for _ in self._fetch_threads:
            self._put_task_sentinel()

        self._coordinator_thread.join(timeout=5.0)

        for thread in self._fetch_threads:
            thread.join(timeout=5.0)

        self._put_fetched_sentinel()
        self._publish_thread.join(timeout=5.0)

    def close_job(self) -> None:
        try:
            if self._coordinator_client is not None and self._job_id:
                status = (
                    batchflow_pb2.JOB_STATUS_COMPLETED
                    if self._completed_normally
                    else batchflow_pb2.JOB_STATUS_CANCELLED
                )
                reason = (
                    "torch iterable dataset completed"
                    if self._completed_normally
                    else "torch iterable dataset closed"
                )

                self._coordinator_client.finish_job(
                    job_id=self._job_id,
                    reason=reason,
                    status=status,
                )
        except Exception:
            LOGGER.exception("Failed to finish BatchFlow job job_id=%s", self._job_id)
        finally:
            if self._coordinator_client is not None:
                self._coordinator_client.close()

    def get_item(self, timeout_seconds: float) -> BatchItem | ErrorItem | EndItem:
        return self.ready_queue.get(timeout=timeout_seconds)

    def qsize(self) -> int:
        return self.ready_queue.qsize()

    def maxsize(self) -> int:
        return self.ready_queue.maxsize

    def update_runtime_metrics(
        self,
        *,
        data_bottleneck_percent: float,
        avg_data_time_sec: float,
        avg_compute_time_sec: float,
        avg_coordinator_wait_total_time_sec: float,
        avg_coordinator_pending_polls: float,
    ) -> None:
        if self.runtime_metrics is None:
            return

        self.runtime_metrics.update(
            data_bottleneck_percent=data_bottleneck_percent,
            avg_data_time_sec=avg_data_time_sec,
            avg_compute_time_sec=avg_compute_time_sec,
            avg_coordinator_wait_total_time_sec=(
                avg_coordinator_wait_total_time_sec
            ),
            avg_coordinator_pending_polls=avg_coordinator_pending_polls,
        )

    def _coordinator_loop(self) -> None:
        coordinator_client = CoordinatorGrpcClient(self.config.coordinator_address)
        self._coordinator_client = coordinator_client

        try:
            coordinator_client.connect()

            job_id = self.config.job_id or f"torch-job-{uuid.uuid4().hex[:8]}"

            start_response = coordinator_client.start_job(
                job_id=job_id,
                dataset_id=self.config.dataset_id,
                lookahead_batches=self.config.lookahead_batches,
                metadata={"job_index": str(self.config.job_index)},
            )

            self._job_id = start_response.job_id

            LOGGER.info(
                "Started BatchFlow torch prefetch job job_id=%s dataset_id=%s "
                "lookahead_batches=%s max_ready_batches=%s fetch_workers=%s",
                start_response.job_id,
                self.config.dataset_id,
                self.config.lookahead_batches,
                self.config.max_ready_batches,
                len(self._fetch_threads),
            )

            while not self._stop_event.is_set():
                if self._scheduled_batches >= self.config.max_batches:
                    self._coordinator_done = True
                    self._completed_normally = True
                    self._maybe_send_end()
                    return

                task = self._get_next_fetch_task(coordinator_client)

                if task is None:
                    self._coordinator_done = True
                    self._completed_normally = True
                    self._maybe_send_end()
                    return

                self._put_task(task)
                self._scheduled_batches += 1

        except BaseException as exc:
            if not self._stop_event.is_set():
                LOGGER.exception("BatchFlow torch coordinator loop failed")
                self._put_ready_item(ErrorItem(error=exc))
                self._stop_event.set()
        finally:
            for _ in self._fetch_threads:
                self._put_task_sentinel()

    def _get_next_fetch_task(
        self,
        coordinator_client: CoordinatorGrpcClient,
    ) -> FetchTask | None:
        assert self._job_id is not None

        coordinator_wait_start = time.perf_counter()
        coordinator_rpc_time_sec = 0.0
        coordinator_sleep_time_sec = 0.0

        pending_polls = 0
        miss_polls = 0
        in_flight_polls = 0

        while not self._stop_event.is_set():
            rpc_start = time.perf_counter()

            metrics_snapshot = (
                self.runtime_metrics.snapshot()
                if self.runtime_metrics is not None
                else None
            )

            response = coordinator_client.get_next_batch(
                self._job_id,
                runtime_feedback=metrics_snapshot,
                timeout_seconds=self.config.coordinator_timeout_seconds,
            )

            coordinator_rpc_time_sec += time.perf_counter() - rpc_start

            if response.done:
                return None

            handle = response.batch_handle
            metadata = _metadata_to_dict(handle.metadata)
            cache_result = metadata.get("cache_result", "")

            if handle.status == batchflow_pb2.BATCH_HANDLE_STATUS_PENDING:
                pending_polls += 1

                if cache_result == "miss":
                    miss_polls += 1
                elif cache_result == "in_flight":
                    in_flight_polls += 1

                if (
                    self.config.log_pending_batch_every_n_polls > 0
                    and pending_polls
                    % self.config.log_pending_batch_every_n_polls
                    == 0
                ):
                    LOGGER.info(
                        "Waiting for BatchFlow batch job_id=%s "
                        "pending_polls=%s cache_result=%s",
                        self._job_id,
                        pending_polls,
                        cache_result,
                    )

                sleep_start = time.perf_counter()
                time.sleep(self.config.request_poll_interval_seconds)
                coordinator_sleep_time_sec += time.perf_counter() - sleep_start
                continue

            if handle.status == batchflow_pb2.BATCH_HANDLE_STATUS_FAILED:
                raise RuntimeError(f"Coordinator returned failed batch handle: {handle}")

            if handle.status != batchflow_pb2.BATCH_HANDLE_STATUS_READY:
                raise RuntimeError(
                    f"Unexpected BatchFlow handle status={handle.status}"
                )

            if not handle.fetch_key:
                raise RuntimeError(f"Missing fetch_key in handle: {handle}")

            if handle.location.startswith(("redis://", "rediss://")):
                pass
            elif handle.location.startswith("grpc://") or not handle.location:
                if not handle.fetch_host or handle.fetch_port <= 0:
                    raise RuntimeError(
                        f"Missing worker fetch info in handle: "
                        f"fetch_host={handle.fetch_host!r} "
                        f"fetch_port={handle.fetch_port!r} "
                        f"fetch_key={handle.fetch_key!r}"
                    )
            else:
                raise RuntimeError(
                    f"Unsupported batch location={handle.location!r}"
                )

            if not handle.payload_format:
                raise RuntimeError(f"Missing payload_format in handle: {handle}")

            if not handle.dataset_format:
                raise RuntimeError(f"Missing dataset_format in handle: {handle}")

            coordinator_wait_total_time_sec = (
                time.perf_counter() - coordinator_wait_start
            )

            client_cache_result = "hot" if pending_polls == 0 else "miss"

            return FetchTask(
                sequence=self._scheduled_batches,
                job_id=self._job_id,
                batch_id=handle.batch_id,
                cache_key=handle.cache_key,
                epoch=int(response.epoch),
                batch_index=(
                    int(response.batch_index)
                    if response.has_batch_index
                    else -1
                ),
                location=handle.location,
                fetch_host=handle.fetch_host,
                fetch_port=int(handle.fetch_port),
                fetch_key=handle.fetch_key,
                payload_format=handle.payload_format,
                dataset_format=handle.dataset_format,
                handle_status=_handle_status_name(handle.status),
                cache_result=cache_result,
                client_cache_result=client_cache_result,
                coordinator_wait_total_time_sec=float(coordinator_wait_total_time_sec),
                coordinator_rpc_time_sec=float(coordinator_rpc_time_sec),
                coordinator_sleep_time_sec=float(coordinator_sleep_time_sec),
                coordinator_pending_polls=int(pending_polls),
                coordinator_miss_polls=int(miss_polls),
                coordinator_in_flight_polls=int(in_flight_polls),
            )

        return None

    def _fetch_worker_loop(self) -> None:
        worker_fetch_client = WorkerFetchClient()
        redis_fetch_client = RedisFetchClient(
            timeout_seconds=self.config.fetch_timeout_seconds
        )

        try:
            while not self._stop_event.is_set():
                try:
                    task = self.task_queue.get(timeout=0.2)
                except queue.Empty:
                    self._maybe_send_end()
                    continue

                if task is None:
                    self.task_queue.task_done()
                    self._maybe_send_end()
                    return

                try:
                    result = self._fetch_and_decode(
                        worker_fetch_client,
                        redis_fetch_client,
                        task,
                    )
                    self._put_fetched_result(result)
                except BaseException as exc:
                    if not self._stop_event.is_set():
                        LOGGER.exception("BatchFlow torch fetch worker failed")
                        self._put_ready_item(ErrorItem(error=exc))
                        self._stop_event.set()
                        return
                finally:
                    self.task_queue.task_done()
                    self._maybe_send_end()
        finally:
            redis_fetch_client.close()
            worker_fetch_client.close()

    def _fetch_and_decode(
        self,
        worker_fetch_client: WorkerFetchClient,
        redis_fetch_client: RedisFetchClient,
        task: FetchTask,
    ) -> FetchedTaskResult:
        fetch_start = time.perf_counter()

        if task.location.startswith(("redis://", "rediss://")):
            payload = redis_fetch_client.fetch_batch(
                location=task.location,
                key=task.fetch_key,
            )
        else:
            payload = worker_fetch_client.fetch_batch(
                host=task.fetch_host,
                port=task.fetch_port,
                key=task.fetch_key,
                timeout_seconds=self.config.fetch_timeout_seconds,
            )

        fetch_time_sec = time.perf_counter() - fetch_start

        decode_start = time.perf_counter()
        batch = decode_payload(
            payload,
            payload_format=task.payload_format,
        )
        decode_time_sec = time.perf_counter() - decode_start

        pin_time_sec = 0.0

        if self.config.pin_memory:
            pin_start = time.perf_counter()
            batch = pin_memory_batch(batch)
            pin_time_sec = time.perf_counter() - pin_start

        return FetchedTaskResult(
            task=task,
            batch=batch,
            payload_bytes=len(payload),
            fetch_time_sec=float(fetch_time_sec),
            decode_time_sec=float(decode_time_sec),
            pin_time_sec=float(pin_time_sec),
        )

    def _ordered_publish_loop(self) -> None:
        coordinator_client = CoordinatorGrpcClient(self.config.coordinator_address)
        next_sequence_to_publish = 0
        buffered: dict[int, FetchedTaskResult] = {}

        try:
            coordinator_client.connect()

            while not self._stop_event.is_set():
                try:
                    item = self.fetched_queue.get(timeout=0.2)
                except queue.Empty:
                    self._maybe_send_end()
                    continue

                if item is None:
                    self.fetched_queue.task_done()
                    self._maybe_send_end()
                    return

                if isinstance(item, ErrorItem):
                    self._put_ready_item(item)
                    self._stop_event.set()
                    self.fetched_queue.task_done()
                    return

                buffered[item.task.sequence] = item
                self.fetched_queue.task_done()

                while next_sequence_to_publish in buffered:
                    result = buffered.pop(next_sequence_to_publish)
                    self._acknowledge_and_publish(coordinator_client, result)
                    next_sequence_to_publish += 1

                self._maybe_send_end()

        except BaseException as exc:
            if not self._stop_event.is_set():
                LOGGER.exception("BatchFlow torch ordered publish loop failed")
                self._put_ready_item(ErrorItem(error=exc))
                self._stop_event.set()
        finally:
            coordinator_client.close()

    def _acknowledge_and_publish(
        self,
        coordinator_client: CoordinatorGrpcClient,
        result: FetchedTaskResult,
    ) -> None:
        task = result.task

        coordinator_client.acknowledge_batch(
            job_id=task.job_id,
            batch_id=task.batch_id,
            epoch=task.epoch,
            batch_index=task.batch_index,
            timeout_seconds=self.config.coordinator_timeout_seconds,
        )

        batch = result.batch

        batch["job_id"] = task.job_id
        batch["dataset_id"] = self.config.dataset_id
        batch["batch_id"] = task.batch_id
        batch["cache_key"] = task.cache_key
        batch["epoch"] = task.epoch
        batch["batch_index"] = task.batch_index

        batch["handle_status"] = task.handle_status
        batch["dataset_format"] = task.dataset_format
        batch["payload_format"] = task.payload_format
        batch["fetch_location"] = task.location
        batch["cache_result"] = task.cache_result
        batch["client_cache_result"] = task.client_cache_result

        batch["pending_polls_before_batch"] = task.coordinator_pending_polls
        batch["miss_polls_before_batch"] = task.coordinator_miss_polls
        batch["in_flight_polls_before_batch"] = task.coordinator_in_flight_polls

        batch["coordinator_wait_total_time_sec"] = (
            task.coordinator_wait_total_time_sec
        )
        batch["coordinator_rpc_time_sec"] = task.coordinator_rpc_time_sec
        batch["coordinator_sleep_time_sec"] = task.coordinator_sleep_time_sec

        batch["fetch_time_sec"] = result.fetch_time_sec
        batch["trainer_decode_time_sec"] = result.decode_time_sec
        batch["trainer_pin_time_sec"] = result.pin_time_sec
        batch["payload_bytes"] = result.payload_bytes
        batch["prefetch_queue_size_before_put"] = int(self.ready_queue.qsize())

        self._put_ready_item(BatchItem(batch=batch))

        self._published_batches += 1
        self._produced_batches += 1

        if (
            self.config.log_every_n_batches > 0
            and self._produced_batches % self.config.log_every_n_batches == 0
        ):
            self._log_status(result)

    def _log_status(self, result: FetchedTaskResult | None = None) -> None:
        if self.config.log_interval_seconds <= 0:
            return

        now = time.time()

        if now - self._last_status_log_time < self.config.log_interval_seconds:
            return

        extra = ""

        if result is not None:
            extra = (
                f" last_batch_id={result.task.batch_id}"
                f" last_cache={result.task.client_cache_result}"
                f" last_fetch_sec={result.fetch_time_sec:.4f}"
                f" last_decode_sec={result.decode_time_sec:.4f}"
            )

        LOGGER.info(
            "BatchFlow torch prefetch status scheduled=%s published=%s "
            "ready_queue=%s/%s task_queue=%s/%s fetched_queue=%s/%s%s",
            self._scheduled_batches,
            self._published_batches,
            self.ready_queue.qsize(),
            self.ready_queue.maxsize,
            self.task_queue.qsize(),
            self.task_queue.maxsize,
            self.fetched_queue.qsize(),
            self.fetched_queue.maxsize,
            extra,
        )

        self._last_status_log_time = now

    def _put_task(self, task: FetchTask) -> None:
        while not self._stop_event.is_set():
            try:
                self.task_queue.put(task, timeout=0.2)
                return
            except queue.Full:
                continue

    def _put_task_sentinel(self) -> None:
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass

    def _put_fetched_result(self, result: FetchedTaskResult) -> None:
        while not self._stop_event.is_set():
            try:
                self.fetched_queue.put(result, timeout=0.2)
                return
            except queue.Full:
                continue

    def _put_fetched_sentinel(self) -> None:
        try:
            self.fetched_queue.put_nowait(None)
        except queue.Full:
            pass

    def _put_ready_item(self, item: BatchItem | ErrorItem | EndItem) -> None:
        while not self._stop_event.is_set():
            try:
                self.ready_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _maybe_send_end(self) -> None:
        if self._end_sent:
            return

        if not self._coordinator_done:
            return

        if not self.task_queue.empty():
            return

        if not self.fetched_queue.empty():
            return

        if self._published_batches < self._scheduled_batches:
            return

        self._end_sent = True
        self._put_ready_item(EndItem())


def _metadata_to_dict(items) -> dict[str, str]:
    return {item.key: item.value for item in items}


def _handle_status_name(value: int) -> str:
    mapping = {
        batchflow_pb2.BATCH_HANDLE_STATUS_PENDING: "pending",
        batchflow_pb2.BATCH_HANDLE_STATUS_READY: "ready",
        batchflow_pb2.BATCH_HANDLE_STATUS_FAILED: "failed",
    }

    return mapping.get(value, f"unknown({value})")