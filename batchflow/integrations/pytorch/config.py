from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerRuntimeMetrics:
    data_bottleneck_percent: float = 0.0
    avg_data_time_sec: float = 0.0
    avg_compute_time_sec: float = 0.0
    avg_coordinator_wait_total_time_sec: float = 0.0
    avg_coordinator_pending_polls: float = 0.0


class TrainerRuntimeMetricsBuffer:
    """
    Thread-safe holder for the latest trainer runtime metrics.

    The training loop can update this while the BatchFlow prefetch thread
    reads snapshots before GetNextBatch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = TrainerRuntimeMetrics()

    def update(
        self,
        *,
        data_bottleneck_percent: float,
        avg_data_time_sec: float,
        avg_compute_time_sec: float,
        avg_coordinator_wait_total_time_sec: float,
        avg_coordinator_pending_polls: float,
    ) -> None:
        with self._lock:
            self._metrics = TrainerRuntimeMetrics(
                data_bottleneck_percent=float(data_bottleneck_percent),
                avg_data_time_sec=float(avg_data_time_sec),
                avg_compute_time_sec=float(avg_compute_time_sec),
                avg_coordinator_wait_total_time_sec=float(
                    avg_coordinator_wait_total_time_sec
                ),
                avg_coordinator_pending_polls=float(avg_coordinator_pending_polls),
            )

    def reset(self) -> None:
        with self._lock:
            self._metrics = TrainerRuntimeMetrics()

    def snapshot(self) -> TrainerRuntimeMetrics:
        with self._lock:
            return self._metrics


@dataclass(frozen=True)
class BatchFlowTorchConfig:
    # Coordinator endpoint used by the PyTorch integration.
    coordinator_address: str = "127.0.0.1:50051"

    # Dataset registered with the BatchFlow coordinator.
    dataset_id: str = "synthetic-dataset"

    # Maximum number of batches this iterator should yield before finishing the job.
    # Required because the coordinator currently keeps advancing epochs indefinitely.
    max_batches: int = 100

    # Optional name for this training job. If empty, the iterator creates one.
    job_id: str = ""

    # Stable index of this job in the workload. The static-allocation ablation
    # uses this to bind fixed worker partitions to jobs without relying on
    # process start order.
    job_index: int = 0

    # Optional per-job lookahead override for debugging/experiments.
    # The coordinator may ignore or adapt this in future versions.
    lookahead_batches: int = 16

    # How long to sleep before polling again when GetNextBatch returns pending.
    request_poll_interval_seconds: float = 0.2

    # Timeout for fetching payload bytes from a worker.
    fetch_timeout_seconds: float | None = 30.0

    # Timeout for coordinator RPCs. None means no explicit timeout.
    coordinator_timeout_seconds: float | None = None

    # Maximum number of decoded batches to buffer for the trainer.
    max_ready_batches: int = 4

    # Number of trainer-side threads used to fetch/decode worker payloads.
    parallel_fetch_workers: int = 4

    # How long the trainer waits for a ready batch before logging/checking again.
    ready_queue_timeout_seconds: float = 0.5

    # Whether to pin returned tensors before yielding to the trainer.
    pin_memory: bool = True

    # Whether closing the iterator should also finish/cancel the BatchFlow job.
    finish_job_on_close: bool = True

    # Minimum seconds between progress/status log lines.
    log_interval_seconds: float = 2.0

    # Emit progress every N produced batches. 0 disables batch-count progress logs.
    log_every_n_batches: int = 20

    # Whether to log when the trainer waits because no decoded batch is ready.
    log_trainer_waits: bool = False

    # Log every N pending coordinator polls. 0 disables pending-wait logs.
    log_pending_batch_every_n_polls: int = 50

    def validate(self) -> None:
        if not self.coordinator_address:
            raise ValueError("coordinator_address must be non-empty")

        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")

        if self.max_batches <= 0:
            raise ValueError(f"max_batches must be > 0, got {self.max_batches}")

        if self.job_index < 0:
            raise ValueError(f"job_index must be >= 0, got {self.job_index}")

        if self.lookahead_batches <= 0:
            raise ValueError(
                f"lookahead_batches must be > 0, got {self.lookahead_batches}"
            )

        if self.request_poll_interval_seconds <= 0:
            raise ValueError(
                "request_poll_interval_seconds must be > 0, "
                f"got {self.request_poll_interval_seconds}"
            )

        if self.fetch_timeout_seconds is not None and self.fetch_timeout_seconds <= 0:
            raise ValueError(
                "fetch_timeout_seconds must be > 0 when set, "
                f"got {self.fetch_timeout_seconds}"
            )

        if (
            self.coordinator_timeout_seconds is not None
            and self.coordinator_timeout_seconds <= 0
        ):
            raise ValueError(
                "coordinator_timeout_seconds must be > 0 when set, "
                f"got {self.coordinator_timeout_seconds}"
            )

        if self.max_ready_batches <= 0:
            raise ValueError(
                f"max_ready_batches must be > 0, got {self.max_ready_batches}"
            )

        if self.parallel_fetch_workers <= 0:
            raise ValueError(
                "parallel_fetch_workers must be > 0, "
                f"got {self.parallel_fetch_workers}"
            )

        if self.ready_queue_timeout_seconds <= 0:
            raise ValueError(
                "ready_queue_timeout_seconds must be > 0, "
                f"got {self.ready_queue_timeout_seconds}"
            )

        if self.log_interval_seconds < 0:
            raise ValueError(
                f"log_interval_seconds must be >= 0, got {self.log_interval_seconds}"
            )

        if self.log_every_n_batches < 0:
            raise ValueError(
                f"log_every_n_batches must be >= 0, got {self.log_every_n_batches}"
            )

        if self.log_pending_batch_every_n_polls < 0:
            raise ValueError(
                "log_pending_batch_every_n_polls must be >= 0, "
                f"got {self.log_pending_batch_every_n_polls}"
            )