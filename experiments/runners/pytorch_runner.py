from __future__ import annotations

import logging
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from batchflow.config.config_types import DatasetConfig
from experiments.common.reporting import PerbatchMetricsWriter, per_batch_metrics_path_for_job
from experiments.common.training import build_training_components
from experiments.config.types import JobConfig, PytorchSystemConfig
from experiments.runners.runner_core import (
    configure_process_logging,
    resolve_device_and_amp,
    run_training_loop,
    set_seed,
)
from experiments.torch_datasets.factory import build_torch_dataset

LOGGER = logging.getLogger("experiments.runners.pytorch_runner")


def _build_dataloader(
    dataset: DatasetConfig,
    system: PytorchSystemConfig,
    *,
    pin_memory: bool,
) -> DataLoader:
    if not dataset.prefix_uri.startswith("s3://"):
        raise ValueError(
            f"PyTorch experiment datasets must use an S3 URI, got {dataset.prefix_uri!r}"
        )

    torch_dataset = build_torch_dataset(dataset)


    if dataset.shuffle:
        if dataset.seed is None:
            sampler = RandomSampler(torch_dataset)
        else:
            sampler = RandomSampler(torch_dataset, generator=torch.Generator().manual_seed(dataset.seed))
    else:
        sampler = SequentialSampler(torch_dataset)

    return DataLoader(
        dataset=torch_dataset,
        sampler=sampler,
        batch_size=dataset.batch_size,
        # shuffle=dataset.shuffle,
        drop_last=dataset.drop_last,
        num_workers=system.num_workers,
        persistent_workers=system.persistent_workers and system.num_workers > 0,
        pin_memory=pin_memory,
    
    )


def _run_job(
    *,
    job_index: int,
    job: JobConfig,
    dataset: DatasetConfig,
    system: PytorchSystemConfig,
    output_dir: Path,
) -> None:

    
    configure_process_logging()
    logger = logging.getLogger(f"experiments.runners.pytorch_runner.{job.name}")

    set_seed(job.seed)
    device, use_amp = resolve_device_and_amp(job.device, job.use_amp, job_index=job_index)
    dataloader = _build_dataloader(dataset, system, pin_memory=device.type == "cuda")
    training = build_training_components(job=job, dataset=dataset, device=device)
    writer = PerbatchMetricsWriter(
        per_batch_metrics_path_for_job(output_dir, job.name),
        flush_every_batches=10,
    )

    logger.info("Starting PyTorch job task=%s device=%s", job.task, device)

    try:
        run_training_loop(
            mode="pytorch",
            job_id=job.name,
            batch_iter=dataloader,
            training=training,
            num_batches=job.num_batches,
            warmup_batches=job.warmup_batches,
            device=device,
            use_amp=use_amp,
            on_batch_end=writer.write_batch,
            logger=logger,
        )
        logger.info("PyTorch job complete")
    except Exception as exc:
        logger.exception("PyTorch job failed", exc_info=exc)
        raise exc
    finally:
        writer.close()


def run_pytorch_jobs(
    *,
    job_configs: tuple[JobConfig, ...],
    dataset_config: DatasetConfig,
    pytorch_config: PytorchSystemConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(job_configs) == 1:
        _run_job(
            job_index=0,
            job=job_configs[0],
            dataset=dataset_config,
            system=pytorch_config,
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
                system=pytorch_config,
                output_dir=output_dir,
            ): job.name
            for index, job in enumerate(job_configs)
        }

        for future in as_completed(futures):
            job_name = futures[future]
            try:
                future.result()
                LOGGER.info("PyTorch job finished job=%s", job_name)
            except Exception as exc:
                LOGGER.exception("PyTorch job failed job=%s", job_name)
                print(f"PyTorch job failed job={job_name} exception={exc!r}")
                errors.append(f"{job_name}: {exc!r}")

    if errors:
        raise RuntimeError("one or more PyTorch jobs failed:\n" + "\n".join(errors))
