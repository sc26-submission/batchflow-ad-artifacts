from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from batchflow.config.config_types import DatasetConfig
from batchflow.integrations.pytorch.config import BatchFlowTorchConfig
from batchflow.integrations.pytorch.iterable_dataset import BatchFlowIterableDataset
from experiments.common.reporting import PerbatchMetricsWriter, per_batch_metrics_path_for_job
from experiments.common.training import build_training_components
from experiments.config.types import JobConfig
from experiments.runners.runner_core import (
    configure_process_logging,
    resolve_device_and_amp,
    run_training_loop,
    set_seed,
)

LOGGER = logging.getLogger("experiments.runners.batchflow_runner")


def _run_job(
    *,
    job_index: int,
    job: JobConfig,
    dataset: DatasetConfig,
    client_config: BatchFlowTorchConfig,
    output_dir: Path,
) -> None:
    configure_process_logging()
    logger = logging.getLogger(f"experiments.runners.batchflow_runner.{job.name}")

    set_seed(job.seed)
    device, use_amp = resolve_device_and_amp(job.device, job.use_amp, job_index=job_index)

    config = replace(
        client_config,
        dataset_id=dataset.dataset_id,
        job_id=job.name,
        job_index=job_index,
        max_batches=job.warmup_batches + job.num_batches,
    )
    config.validate()

    batchflow_dataset = BatchFlowIterableDataset(config=config)
    training = build_training_components(job=job, dataset=dataset, device=device)
    writer = PerbatchMetricsWriter(
        per_batch_metrics_path_for_job(output_dir, job.name),
        flush_every_batches=10,
    )

    logger.info("Starting BatchFlow job task=%s device=%s", job.task, device)

    try:
        run_training_loop(
            mode="batchflow",
            job_id=job.name,
            batch_iter=batchflow_dataset,
            training=training,
            num_batches=job.num_batches,
            warmup_batches=job.warmup_batches,
            device=device,
            use_amp=use_amp,
            on_batch_end=writer.write_batch,
            logger=logger,
        )
        logger.info("BatchFlow job complete")
    except Exception as exc:
            logger.exception("BatchFlow job failed", exc_info=exc)
            raise exc
    finally:
        writer.close()


def run_batchflow_jobs(
    *,
    job_configs: tuple[JobConfig, ...],
    dataset_config: DatasetConfig,
    batchflow_config: BatchFlowTorchConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(job_configs) == 1:
        _run_job(
            job_index=0,
            job=job_configs[0],
            dataset=dataset_config,
            client_config=batchflow_config,
            output_dir=output_dir,
        )
        return

    context = mp.get_context("spawn")
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=len(job_configs), mp_context=context) as executor:
        futures = {
            executor.submit(
                _run_job,
                job_index=index,
                job=job,
                dataset=dataset_config,
                client_config=batchflow_config,
                output_dir=output_dir,
            ): job.name
            for index, job in enumerate(job_configs)
        }

        for future in as_completed(futures):
            job_name = futures[future]
            try:
                future.result()
                LOGGER.info("BatchFlow job finished job=%s", job_name)
            except Exception as exc:
                LOGGER.exception("BatchFlow job failed job=%s", job_name)
                errors.append(f"{job_name}: {exc!r}")

    if errors:
        raise RuntimeError("one or more BatchFlow jobs failed:\n" + "\n".join(errors))
