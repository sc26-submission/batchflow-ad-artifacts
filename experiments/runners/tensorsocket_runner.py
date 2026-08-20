from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Empty
from typing import Any

from torch.utils.data import DataLoader

from batchflow.config.config_types import DatasetConfig
from experiments.baselines.tensorsocket import TensorConsumer, TensorProducer
from experiments.common.reporting import PerbatchMetricsWriter, per_batch_metrics_path_for_job
from experiments.common.training import build_training_components
from experiments.config.types import JobConfig, TensorSocketSystemConfig
from experiments.runners.runner_core import (
    configure_process_logging,
    resolve_device_and_amp,
    run_training_loop,
    set_seed,
)
from experiments.torch_datasets.factory import build_torch_dataset

LOGGER = logging.getLogger("experiments.runners.tensorsocket_runner")


def _build_producer_dataloader(
    dataset: DatasetConfig,
    system: TensorSocketSystemConfig,
) -> DataLoader:
    if not dataset.prefix_uri.startswith("s3://"):
        raise ValueError(
            f"TensorSocket experiment datasets must use an S3 URI, got {dataset.prefix_uri!r}"
        )

    return DataLoader(
        build_torch_dataset(dataset),
        batch_size=dataset.batch_size,
        shuffle=dataset.shuffle,
        drop_last=dataset.drop_last,
        num_workers=system.num_workers,
        persistent_workers=system.persistent_workers and system.num_workers > 0,
        pin_memory=False,
    )


def _run_producer(
    *,
    dataset: DatasetConfig,
    system: TensorSocketSystemConfig,
    expected_consumers: int,
    stop_event: Any,
    status_queue: Any,
) -> None:
    configure_process_logging()
    logger = logging.getLogger("experiments.runners.tensorsocket_runner.producer")
    producer = None

    try:
        set_seed(dataset.seed)
        producer = TensorProducer(
            _build_producer_dataloader(dataset, system),
            expected_consumers=expected_consumers,
            port=system.port,
            control_port=system.control_port,
            consumer_buffer_size=system.consumer_buffer_size,
            max_lag_batches=system.max_lag_batches,
        )
        status_queue.put(("ready", ""))
        logger.info(
            "TensorSocket producer started workers=%d batch_size=%d",
            system.num_workers,
            dataset.batch_size,
        )
        producer.run(stop_event)
    except BaseException as exc:
        status_queue.put(("error", repr(exc)))
        logger.exception("TensorSocket producer failed")
        raise
    finally:
        if producer is not None:
            producer.close()
        logger.info("TensorSocket producer stopped")


def _run_job(
    *,
    job_index: int,
    job: JobConfig,
    dataset: DatasetConfig,
    system: TensorSocketSystemConfig,
    output_dir: Path,
) -> None:
    configure_process_logging()
    logger = logging.getLogger(f"experiments.runners.tensorsocket_runner.{job.name}")

    set_seed(job.seed)
    device, use_amp = resolve_device_and_amp(job.device, job.use_amp, job_index=job_index)
    consumer = TensorConsumer(
        host=system.host,
        port=system.port,
        control_port=system.control_port,
        buffer_size=system.consumer_buffer_size,
        receive_timeout_seconds=system.receive_timeout_seconds,
    )
    training = build_training_components(job=job, dataset=dataset, device=device)
    writer = PerbatchMetricsWriter(
        per_batch_metrics_path_for_job(output_dir, job.name),
        flush_every_batches=10,
    )

    logger.info("Starting TensorSocket job task=%s device=%s", job.task, device)

    try:
        run_training_loop(
            mode="tensorsocket",
            job_id=job.name,
            batch_iter=consumer,
            training=training,
            num_batches=job.num_batches,
            warmup_batches=job.warmup_batches,
            device=device,
            use_amp=use_amp,
            on_batch_end=writer.write_batch,
            logger=logger,
        )
        logger.info("TensorSocket job complete")
    except Exception as exc:
            logger.exception("TensorSocket job failed", exc_info=exc)
    finally:
        consumer.close()
        writer.close()


def run_tensorsocket_jobs(
    *,
    job_configs: tuple[JobConfig, ...],
    dataset_config: DatasetConfig,
    tensorsocket_config: TensorSocketSystemConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    stop_event = context.Event()
    status_queue = context.Queue()
    producer = context.Process(
        target=_run_producer,
        kwargs={
            "dataset": dataset_config,
            "system": tensorsocket_config,
            "expected_consumers": len(job_configs),
            "stop_event": stop_event,
            "status_queue": status_queue,
        },
        name="tensorsocket-producer",
    )
    producer.start()

    try:
        try:
            status, message = status_queue.get(
                timeout=tensorsocket_config.startup_timeout_seconds
            )
        except Empty as exc:
            raise RuntimeError("TensorSocket producer did not start in time") from exc

        if status != "ready":
            raise RuntimeError(f"TensorSocket producer failed to start: {message}")

        errors: list[str] = []
        with ProcessPoolExecutor(max_workers=len(job_configs), mp_context=context) as executor:
            futures = {
                executor.submit(
                    _run_job,
                    job_index=index,
                    job=job,
                    dataset=dataset_config,
                    system=tensorsocket_config,
                    output_dir=output_dir,
                ): job.name
                for index, job in enumerate(job_configs)
            }

            for future in as_completed(futures):
                job_name = futures[future]
                try:
                    future.result()
                    LOGGER.info("TensorSocket job finished job=%s", job_name)
                except Exception as exc:
                    LOGGER.exception("TensorSocket job failed job=%s", job_name)
                    errors.append(f"{job_name}: {exc!r}")

        if errors:
            raise RuntimeError("one or more TensorSocket jobs failed:\n" + "\n".join(errors))

    
    finally:
        stop_event.set()
        producer.join(timeout=10.0)

        if producer.is_alive():
            LOGGER.warning("TensorSocket producer did not stop cleanly; terminating")
            producer.terminate()
            producer.join(timeout=5.0)

        status_queue.close()
        status_queue.join_thread()
