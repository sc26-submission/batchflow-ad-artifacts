from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

import torch

from batchflow.common.utils import ResourceMonitor, SmoothedMeter
from experiments.common.training import TrainingComponents


def configure_process_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def resolve_device_and_amp(
    device: str,
    use_amp: bool | None,
    *,
    job_index: int,
) -> tuple[torch.device, bool]:
    requested = device.lower()

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        device_count = torch.cuda.device_count()

        if job_index >= device_count:
            raise RuntimeError(
                f"Job {job_index} requires a GPU, but only "
                f"{device_count} CUDA device(s) are available."
            )

        resolved = torch.device(f"cuda:{job_index}")

    elif requested == "auto":
        resolved = torch.device("cpu")

    else:
        resolved = torch.device(device)

    amp_enabled = (
        resolved.type == "cuda"
        if use_amp is None
        else bool(use_amp) and resolved.type == "cuda"
    )

    return resolved, amp_enabled


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_scalar(
    batch: dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    value = batch.get(key)

    if value is None:
        return default

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default

        return float(value.sum().item())

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_batch_metrics(
    batch: dict[str, Any],
) -> dict[str, float]:
    keys = {
        "baseline_io_time_sec": "io_time_sec",
        "baseline_decode_time_sec": "decode_time_sec",
        "baseline_transform_time_sec": "transform_time_sec",
        "batchflow_worker_io_time_sec": "worker_io_time_sec",
        "batchflow_worker_decode_time_sec": "worker_decode_time_sec",
        "batchflow_worker_transform_time_sec": "worker_transform_time_sec",
        "batchflow_worker_stack_time_sec": "worker_stack_time_sec",
        "batchflow_worker_serialize_time_sec": "worker_serialize_time_sec",
        "batchflow_fetch_time_sec": "fetch_time_sec",
        "batchflow_coordinator_rpc_time_sec": "coordinator_rpc_time_sec",
        "batchflow_coordinator_sleep_time_sec": "coordinator_sleep_time_sec",
        "batchflow_coordinator_wait_total_time_sec": (
            "coordinator_wait_total_time_sec"
        ),
        "batchflow_coordinator_pending_polls": "pending_polls_before_batch",
        "trainer_decode_time_sec": "trainer_decode_time_sec",
        "trainer_pin_time_sec": "trainer_pin_time_sec",
        "prefetch_queue_size_before_put": "prefetch_queue_size_before_put",
        "trainer_queue_size_after_get": "trainer_queue_size_after_get",
        "trainer_queue_empty_events": "trainer_queue_empty_events",
        "tensorsocket_wait_time_sec": "tensorsocket_wait_time_sec",
        "tensorsocket_cache_hit": "tensorsocket_cache_hit",
        "coordl_wait_time_sec": "coordl_wait_time_sec",
        "coordl_deserialize_time_sec": "coordl_deserialize_time_sec",
        "coordl_io_time_sec": "coordl_io_time_sec",
        "coordl_decode_time_sec": "coordl_decode_time_sec",
        "coordl_transform_time_sec": "coordl_transform_time_sec",
        "coordl_prep_time_sec": "coordl_prep_time_sec",
        "coordl_payload_bytes": "coordl_payload_bytes",
        "coordl_is_owner": "coordl_is_owner",
    }

    return {
        output_name: _batch_scalar(batch, batch_name)
        for output_name, batch_name in keys.items()
    }


def _bottleneck_percent(
    data_time: float,
    compute_time: float,
) -> float:
    total = data_time + compute_time
    return 100.0 * data_time / max(total, 1e-12)


def _update_batchflow_metrics(
    batch_iter: Any,
    *,
    data_meter: SmoothedMeter,
    compute_meter: SmoothedMeter,
    batch_metrics: dict[str, float],
) -> None:
    update = getattr(batch_iter, "update_runtime_metrics", None)

    if not callable(update):
        return

    update(
        data_bottleneck_percent=_bottleneck_percent(
            data_meter.avg,
            compute_meter.avg,
        ),
        avg_data_time_sec=data_meter.avg,
        avg_compute_time_sec=compute_meter.avg,
        avg_coordinator_wait_total_time_sec=(
            batch_metrics["batchflow_coordinator_wait_total_time_sec"]
        ),
        avg_coordinator_pending_polls=(
            batch_metrics["batchflow_coordinator_pending_polls"]
        ),
    )


def _should_log_batch(
    batch: int,
    total_batches: int,
    log_every_batches: int,
) -> bool:
    return (
        batch == 1
        or batch == total_batches
        or (
            log_every_batches > 0
            and batch % log_every_batches == 0
        )
    )


def _log_warmup_progress(
    logger: logging.Logger,
    *,
    mode: str,
    batch: int,
    warmup_batches: int,
    data_time: float,
    compute_time: float,
) -> None:
    logger.info(
        f"{mode} | "
        f"warmup={batch}/{warmup_batches} | "
        f"data={data_time:.4f}s | "
        f"compute={compute_time:.4f}s"
    )


def _log_progress(
    logger: logging.Logger,
    *,
    mode: str,
    batch: int,
    num_batches: int,
    total_samples: int,
    total_data_time: float,
    total_compute_time: float,
    total_batch_time: float,
) -> None:
    avg_data_time = total_data_time / batch
    avg_compute_time = total_compute_time / batch

    batches_per_sec = batch / max(total_batch_time, 1e-12)
    samples_per_sec = total_samples / max(total_batch_time, 1e-12)

    logger.info(
        f"{mode} | "
        f"batch={batch}/{num_batches} | "
        f"data={avg_data_time:.4f}s | "
        f"compute={avg_compute_time:.4f}s | "
        f"throughput={batches_per_sec:.2f} batches/s | "
        f"samples={samples_per_sec:.2f}/s"
    )


def run_training_loop(
    *,
    mode: str,
    job_id: str,
    batch_iter: Iterable[dict[str, Any]],
    training: TrainingComponents,
    num_batches: int,
    warmup_batches: int,
    device: torch.device,
    use_amp: bool,
    on_batch_end: Callable[[dict[str, Any]], None],
    logger: logging.Logger,
    log_every_batches: int = 10,
) -> None:
    """Run warmup batches followed by exactly num_batches training batches."""

    # Rolling measurements used for BatchFlow runtime feedback.
    data_meter = SmoothedMeter(window_size=20)
    compute_meter = SmoothedMeter(window_size=20)
    batch_meter = SmoothedMeter(window_size=20)
    throughput_meter = SmoothedMeter(window_size=20)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    iterator = iter(batch_iter)
    total_loop_batches = warmup_batches + num_batches
    run_start = time.perf_counter()

    # Cumulative values exclude warmup.
    completed_batches = 0
    total_samples = 0
    total_data_time = 0.0
    total_compute_time = 0.0
    total_batch_time = 0.0

    monitor = ResourceMonitor(
        sample_interval_seconds=0.25,
        logger=logger,
    )
    monitor.start()

    try:
        training.model.train()

        for loop_batch in range(total_loop_batches):
            is_warmup = loop_batch < warmup_batches
            batch_start = time.perf_counter()

            # Time spent waiting for the next batch.
            data_start = time.perf_counter()
            batch = next(iterator)
            data_time = time.perf_counter() - data_start

            batch_metrics = _extract_batch_metrics(batch)

            result = training.run_batch(
                batch,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )

            loss = result.loss
            batch_size = result.batch_size
            compute_time = result.compute_time_sec
            h2d_time = result.h2d_time_sec
            forward_time = result.forward_time_sec
            backward_time = result.backward_time_sec
            optimizer_time = result.optimizer_step_time_sec

            batch_time = time.perf_counter() - batch_start
            samples_per_sec = batch_size / max(batch_time, 1e-12)

            # Rolling values include the most recent batches and are used
            # for runtime feedback.
            data_meter.update(data_time)
            compute_meter.update(compute_time)
            batch_meter.update(batch_time)
            throughput_meter.update(samples_per_sec)

            _update_batchflow_metrics(
                iterator,
                data_meter=data_meter,
                compute_meter=compute_meter,
                batch_metrics=batch_metrics,
            )

            # Warmup batches are excluded from the cumulative run statistics.
            if not is_warmup:
                completed_batches += 1
                total_samples += batch_size
                total_data_time += data_time
                total_compute_time += compute_time
                total_batch_time += batch_time

            row = {
                "mode": mode,
                "batch": loop_batch,
                "warmup": int(is_warmup),
                "batch_size": batch_size,
                "loss": loss,
                "batch_time_sec": batch_time,
                "data_time_sec": data_time,
                "compute_time_sec": compute_time,
                "h2d_time_sec": h2d_time,
                "forward_time_sec": forward_time,
                "backward_time_sec": backward_time,
                "optimizer_step_time_sec": optimizer_time,
                "samples_per_sec": samples_per_sec,
                "rolling_samples_per_sec": throughput_meter.avg,
                "avg_batch_time_sec": batch_meter.avg,
                "avg_data_time_sec": data_meter.avg,
                "avg_compute_time_sec": compute_meter.avg,
                "data_bottleneck_percent": _bottleneck_percent(
                    data_time,
                    compute_time,
                ),
                "avg_data_bottleneck_percent": _bottleneck_percent(
                    data_meter.avg,
                    compute_meter.avg,
                ),
                **batch_metrics,
                "batch_index": int(
                    batch.get("batch_index", -1)
                ),
                "batch_id": str(
                    batch.get("batch_id", "")
                ),
                "job_id": str(
                    batch.get("job_id") or job_id
                ),
                "elapsed_time_sec": (
                    time.perf_counter() - run_start
                ),
                "device": str(device),
                "use_amp": int(amp_enabled),
            }

            on_batch_end(row)

            if is_warmup:
                warmup_batch = loop_batch + 1

                if _should_log_batch(
                    warmup_batch,
                    warmup_batches,
                    log_every_batches,
                ):
                    _log_warmup_progress(
                        logger,
                        mode=mode,
                        batch=warmup_batch,
                        warmup_batches=warmup_batches,
                        data_time=data_time,
                        compute_time=compute_time,
                    )

                continue

            if _should_log_batch(
                completed_batches,
                num_batches,
                log_every_batches,
            ):
                _log_progress(
                    logger,
                    mode=mode,
                    batch=completed_batches,
                    num_batches=num_batches,
                    total_samples=total_samples,
                    total_data_time=total_data_time,
                    total_compute_time=total_compute_time,
                    total_batch_time=total_batch_time,
                )

    finally:
        close = getattr(iterator, "close", None)

        if callable(close):
            close()

        monitor.stop()