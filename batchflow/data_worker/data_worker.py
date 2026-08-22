from __future__ import annotations

import logging
import threading
import time

import grpc

from batchflow.cache.redis_store import RedisPayloadStore
from batchflow.clients.coordinator_client import CoordinatorGrpcClient
from batchflow.config.config_types import RedisConfig
from batchflow.data_worker.materializers import BatchMaterializer
from batchflow.data_worker.payload_store import InMemoryPayloadStore
from batchflow.data_worker.worker_grpc_service import serve_worker_data_grpc
from batchflow.proto import batchflow_pb2


LOGGER = logging.getLogger("batchflow.worker")


class DataWorker:
    def __init__(
        self,
        *,
        worker_id: str,
        coordinator_address: str,
        hostname: str,
        fetch_host: str,
        fetch_port: int,
        poll_interval_seconds: float,
        heartbeat_interval_seconds: float,
        s3_fetch_threads: int,
        decode_threads: int,
        transient_ttl: int,
        redis_config: RedisConfig,
        payload_cleanup_interval_seconds: float = 1000.0,
        log_every_n_tasks: int = 10,
    ) -> None:
        self.worker_id = worker_id
        self.coordinator_address = coordinator_address
        self.hostname = hostname
        self.fetch_host = fetch_host
        self.fetch_port = fetch_port
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.transient_ttl = transient_ttl
        self.payload_cleanup_interval_seconds = payload_cleanup_interval_seconds
        self.log_every_n_tasks = log_every_n_tasks

        self.coordinator_client = CoordinatorGrpcClient(coordinator_address)

        # Transient payloads are kept in worker-local memory and served
        # through this worker's gRPC endpoint.
        self.local_payload_store = InMemoryPayloadStore()

        # Reusable payloads use the shared Redis/ElastiCache store when
        # reuse is enabled for the current BatchFlow run.
        self.redis_payload_store: RedisPayloadStore | None = None
        if redis_config.enabled:
            self.redis_payload_store = RedisPayloadStore(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                ssl=redis_config.ssl,
                password=redis_config.password or None,
                key_prefix=redis_config.key_prefix,
            )

        self.materializer = BatchMaterializer(
            s3_fetch_threads=s3_fetch_threads,
            decode_threads=decode_threads,
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_heartbeat = 0.0
        self._last_payload_cleanup = 0.0
        self._data_server: grpc.Server | None = None

        self._completed_task_count = 0
        self._materialized_task_count = 0
        self._reused_task_count = 0

    def start(self) -> None:
        LOGGER.info(
            "Worker runtime starting | id=%s | coordinator=%s | fetch=%s:%s | reuse=%s",
            self.worker_id,
            self.coordinator_address,
            self.fetch_host,
            self.fetch_port,
            "enabled" if self.redis_payload_store is not None else "disabled",
        )

        # Listen on all local interfaces. fetch_host is the address advertised
        # to trainers and the coordinator, not the local bind address.
        self._data_server = serve_worker_data_grpc(
            payload_store=self.local_payload_store,
            host="0.0.0.0",
            port=self.fetch_port,
        )

        self.coordinator_client.connect()

        self.coordinator_client.register_worker(
            worker_id=self.worker_id,
            hostname=self.hostname,
            metadata={
                "fetch_host": self.fetch_host,
                "fetch_port": str(self.fetch_port),
            },
        )

        self._thread = threading.Thread(
            target=self._run_loop,
            name=self.worker_id,
            daemon=True,
        )
        self._thread.start()

        LOGGER.info(
            "Worker runtime ready | id=%s | fetch=%s:%s",
            self.worker_id,
            self.fetch_host,
            self.fetch_port,
        )

    def stop(self) -> None:
        LOGGER.info("Worker runtime stopping | id=%s", self.worker_id)

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._data_server is not None:
            self._data_server.stop(0)

        if self.redis_payload_store is not None:
            self.redis_payload_store.close()

        self.coordinator_client.close()

        LOGGER.info(
            "Worker runtime stopped | id=%s | tasks=%s | materialized=%s | reused=%s",
            self.worker_id,
            self._completed_task_count,
            self._materialized_task_count,
            self._reused_task_count,
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            task_id: str | None = None
            batch_id: str | None = None

            try:
                self._maybe_heartbeat()
                self._maybe_cleanup_payload_store()

                resp = self.coordinator_client.poll_materialization_task(self.worker_id)

                if not resp.has_task:
                    time.sleep(self.poll_interval_seconds)
                    continue

                task = resp.task
                batch = task.batch
                task_id = task.task_id
                batch_id = batch.batch_id
                fetch_key = batch.batch_id

                if task.storage_class not in (
                    batchflow_pb2.STORAGE_CLASS_REUSABLE,
                    batchflow_pb2.STORAGE_CLASS_TRANSIENT,
                ):
                    raise ValueError(f"unknown storage_class={task.storage_class}")

                if not task.payload_format:
                    raise ValueError(
                        f"materialize task missing payload_format task_id={task_id}"
                    )

                if not task.dataset_format:
                    raise ValueError(
                        f"materialize task missing dataset_format task_id={task_id}"
                    )

                LOGGER.debug(
                    "Picked materialization task worker_id=%s task_id=%s "
                    "job_id=%s batch_id=%s dataset_id=%s epoch=%s "
                    "batch_index=%s storage_class=%s payload_format=%s dataset_format=%s",
                    self.worker_id,
                    task_id,
                    task.job_id,
                    batch.batch_id,
                    batch.dataset_id,
                    batch.epoch,
                    batch.batch_index,
                    task.storage_class,
                    task.payload_format,
                    task.dataset_format,
                )

                use_redis = (
                    task.storage_class == batchflow_pb2.STORAGE_CLASS_REUSABLE
                    and self.redis_payload_store is not None
                )

                if use_redis:
                    assert self.redis_payload_store is not None

                    if self.redis_payload_store.contains(key=fetch_key):
                        redis_key = self.redis_payload_store.make_key(fetch_key)
                        payload_bytes = self.redis_payload_store.size_bytes(key=fetch_key)

                        self.coordinator_client.complete_materialization(
                            task_id=task_id,
                            worker_id=self.worker_id,
                            location=self.redis_payload_store.location,
                            size_bytes=payload_bytes,
                            storage_class=task.storage_class,
                            expires_at_ms=0,
                            metadata={
                                "fetch_key": redis_key,
                                "payload_format": task.payload_format,
                                "reused_redis_payload": "true",
                            },
                        )

                        self._record_completed_task(
                            batch_id=batch.batch_id,
                            action="reused",
                            destination="redis",
                            payload_bytes=payload_bytes,
                            storage_class=task.storage_class,
                            payload_format=task.payload_format,
                        )
                        continue

                else:
                    existing_entry = self.local_payload_store.get(key=fetch_key)

                    if existing_entry is not None:
                        self.coordinator_client.complete_materialization(
                            task_id=task_id,
                            worker_id=self.worker_id,
                            location=f"grpc://{self.fetch_host}:{self.fetch_port}",
                            size_bytes=existing_entry.size_bytes,
                            storage_class=task.storage_class,
                            expires_at_ms=existing_entry.evict_after_ms,
                            metadata={
                                "fetch_host": self.fetch_host,
                                "fetch_port": str(self.fetch_port),
                                "fetch_key": fetch_key,
                                "payload_format": existing_entry.payload_format,
                                "reused_local_payload": "true",
                                "evict_after_ms": str(existing_entry.evict_after_ms),
                            },
                        )

                        self._record_completed_task(
                            batch_id=batch.batch_id,
                            action="reused",
                            destination="local",
                            payload_bytes=existing_entry.size_bytes,
                            storage_class=task.storage_class,
                            payload_format=existing_entry.payload_format,
                        )
                        continue

                materialize_start = time.perf_counter()

                materialized = self.materializer.materialize(
                    batch,
                    payload_format=task.payload_format,
                    dataset_format=task.dataset_format,
                )

                materialize_time_sec = time.perf_counter() - materialize_start

                if use_redis:
                    assert self.redis_payload_store is not None

                    redis_key = self.redis_payload_store.put(
                        key=fetch_key,
                        payload=materialized.payload,
                    )

                    # Keep a temporary local fallback in case Redis retention
                    # is rejected by the bounded cache policy.
                    fallback_expires_at_ms = self.local_payload_store.put(
                        key=fetch_key,
                        payload=materialized.payload,
                        payload_format=materialized.payload_format,
                        ttl_seconds=self.transient_ttl,
                    )

                    LOGGER.debug(
                        "Stored reusable payload in Redis "
                        "worker_id=%s task_id=%s batch_id=%s fetch_key=%s "
                        "storage_class=%s payload_bytes=%s payload_format=%s "
                        "materialize_time_sec=%.6f fallback_expires_at_ms=%s",
                        self.worker_id,
                        task_id,
                        batch.batch_id,
                        redis_key,
                        task.storage_class,
                        len(materialized.payload),
                        materialized.payload_format,
                        materialize_time_sec,
                        fallback_expires_at_ms,
                    )

                    self.coordinator_client.complete_materialization(
                        task_id=task_id,
                        worker_id=self.worker_id,
                        location=self.redis_payload_store.location,
                        size_bytes=len(materialized.payload),
                        storage_class=task.storage_class,
                        expires_at_ms=0,
                        metadata={
                            "fetch_key": redis_key,
                            "payload_format": materialized.payload_format,
                            "materialize_time_sec": f"{materialize_time_sec:.6f}",
                            "reused_redis_payload": "false",
                            "fallback_fetch_host": self.fetch_host,
                            "fallback_fetch_port": str(self.fetch_port),
                            "fallback_fetch_key": fetch_key,
                            "fallback_expires_at_ms": str(fallback_expires_at_ms),
                        },
                    )

                else:
                    # Non-reusable payloads stay in worker-local memory and
                    # expire after the configured transient TTL.
                    evict_after_ms = self.local_payload_store.put(
                        key=fetch_key,
                        payload=materialized.payload,
                        payload_format=materialized.payload_format,
                        ttl_seconds=self.transient_ttl,
                    )

                    LOGGER.debug(
                        "Stored worker-local batch payload "
                        "worker_id=%s task_id=%s batch_id=%s fetch_key=%s "
                        "storage_class=%s payload_bytes=%s payload_format=%s "
                        "materialize_time_sec=%.6f evict_after_ms=%s",
                        self.worker_id,
                        task_id,
                        batch.batch_id,
                        fetch_key,
                        task.storage_class,
                        len(materialized.payload),
                        materialized.payload_format,
                        materialize_time_sec,
                        evict_after_ms,
                    )

                    self.coordinator_client.complete_materialization(
                        task_id=task_id,
                        worker_id=self.worker_id,
                        location=f"grpc://{self.fetch_host}:{self.fetch_port}",
                        size_bytes=len(materialized.payload),
                        storage_class=task.storage_class,
                        expires_at_ms=evict_after_ms,
                        metadata={
                            "fetch_host": self.fetch_host,
                            "fetch_port": str(self.fetch_port),
                            "fetch_key": fetch_key,
                            "payload_format": materialized.payload_format,
                            "materialize_time_sec": f"{materialize_time_sec:.6f}",
                            "evict_after_ms": str(evict_after_ms),
                            "reused_local_payload": "false",
                        },
                    )

                self._record_completed_task(
                    batch_id=batch.batch_id,
                    action="materialized",
                    destination="redis" if use_redis else "local",
                    payload_bytes=len(materialized.payload),
                    storage_class=task.storage_class,
                    payload_format=materialized.payload_format,
                    duration_sec=materialize_time_sec,
                )

            except Exception as exc:
                LOGGER.exception(
                    "Task failed | worker=%s | task=%s | batch=%s | error=%s",
                    self.worker_id,
                    task_id or "-",
                    batch_id or "-",
                    exc,
                )

                if task_id is not None:
                    try:
                        self.coordinator_client.fail_materialization(
                            task_id=task_id,
                            worker_id=self.worker_id,
                            reason=str(exc),
                        )
                    except Exception as report_exc:
                        LOGGER.exception(
                            "Could not report task failure | worker=%s | task=%s | error=%s",
                            self.worker_id,
                            task_id,
                            report_exc,
                        )

                time.sleep(self.poll_interval_seconds)

    def _record_completed_task(
        self,
        *,
        batch_id: str,
        action: str,
        destination: str,
        payload_bytes: int,
        storage_class: int,
        payload_format: str,
        duration_sec: float | None = None,
    ) -> None:
        self._completed_task_count += 1

        if action == "materialized":
            self._materialized_task_count += 1
        elif action == "reused":
            self._reused_task_count += 1

        if not self._should_log_task_progress():
            return

        duration = f" | time={duration_sec:.3f}s" if duration_sec is not None else ""

        LOGGER.info(
            "Progress | worker=%s | tasks=%s (%s materialized, %s reused) | "
            "last=%s | batch=%s | target=%s/%s | size=%s%s",
            self.worker_id,
            self._completed_task_count,
            self._materialized_task_count,
            self._reused_task_count,
            action,
            batch_id,
            destination,
            _storage_class_name(storage_class),
            _format_bytes(payload_bytes),
            duration,
        )

        LOGGER.debug(
            "Last task details | worker=%s | batch=%s | payload_format=%s | storage_class=%s",
            self.worker_id,
            batch_id,
            payload_format,
            storage_class,
        )

    def _should_log_task_progress(self) -> bool:
        if self.log_every_n_tasks <= 0:
            return False
        return self._completed_task_count % self.log_every_n_tasks == 0

    def _maybe_heartbeat(self) -> None:
        current_time = time.time()

        if current_time - self._last_heartbeat >= self.heartbeat_interval_seconds:
            self.coordinator_client.heartbeat_worker(self.worker_id)
            self._last_heartbeat = current_time

    def _maybe_cleanup_payload_store(self) -> None:
        """Clean expired worker-local transient payloads."""

        current_time = time.time()

        if current_time - self._last_payload_cleanup < self.payload_cleanup_interval_seconds:
            return

        removed_count = self.local_payload_store.cleanup_expired()
        self._last_payload_cleanup = current_time

        if removed_count > 0:
            LOGGER.info(
                "Transient cache cleanup | worker=%s | removed=%s | remaining=%s | memory=%s",
                self.worker_id,
                removed_count,
                self.local_payload_store.size(),
                _format_bytes(self.local_payload_store.total_bytes()),
            )


def _storage_class_name(storage_class: int) -> str:
    if storage_class == batchflow_pb2.STORAGE_CLASS_REUSABLE:
        return "reusable"
    if storage_class == batchflow_pb2.STORAGE_CLASS_TRANSIENT:
        return "transient"
    return str(storage_class)


def _format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size_bytes} B"