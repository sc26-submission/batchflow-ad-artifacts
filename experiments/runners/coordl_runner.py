from __future__ import annotations

import logging
import multiprocessing as mp
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Empty

from batchflow.config.config_types import DatasetConfig
from experiments.baselines.coordl import (
    CoorDLBatchIterator,
    CoorDLStagingStore,
    build_coordl_batch_plan,
    dataset_sample_count,
    run_preparation_owner,
)
from experiments.common.reporting import PerbatchMetricsWriter, per_batch_metrics_path_for_job
from experiments.common.training import build_training_components
from experiments.config.types import CoorDLSystemConfig, JobConfig
from experiments.runners.runner_core import (
    configure_process_logging,
    resolve_device_and_amp,
    run_training_loop,
    set_seed,
)


LOGGER = logging.getLogger("experiments.runners.coordl_runner")


def _validate_jobs(job_configs: tuple[JobConfig, ...]) -> int:
    if not job_configs:
        raise ValueError("CoorDL requires at least one job")

    expected = (job_configs[0].warmup_batches, job_configs[0].num_batches)
    for job in job_configs[1:]:
        current = (job.warmup_batches, job.num_batches)
        if current != expected:
            raise ValueError(
                "CoorDL coordinated prep requires the same warmup_batches and "
                f"num_batches for every job; expected {expected}, got {current} for {job.name!r}"
            )

    return sum(expected)


def _run_job(
    *,
    job_index: int,
    job: JobConfig,
    dataset: DatasetConfig,
    system: CoorDLSystemConfig,
    plan,
    namespace: str,
    num_jobs: int,
    output_dir: Path,
) -> None:
    configure_process_logging()
    logger = logging.getLogger(f"experiments.runners.coordl_runner.{job.name}")

    set_seed(job.seed)
    device, use_amp = resolve_device_and_amp(
        job.device, job.use_amp, job_index=job_index
    )

    batch_iter = CoorDLBatchIterator(
        plan=plan,
        system=system,
        namespace=namespace,
        job_index=job_index,
        job_id=job.name,
        num_jobs=num_jobs,
    )

    training = build_training_components(job=job, dataset=dataset, device=device)

    writer = PerbatchMetricsWriter(
        per_batch_metrics_path_for_job(output_dir, job.name),
        flush_every_batches=10,
    )

    logger.info("Starting CoorDL job task=%s device=%s", job.task, device)

    try:
        run_training_loop(
            mode="coordl",
            job_id=job.name,
            batch_iter=batch_iter,
            training=training,
            num_batches=job.num_batches,
            warmup_batches=job.warmup_batches,
            device=device,
            use_amp=use_amp,
            on_batch_end=writer.write_batch,
            logger=logger,
        )
        logger.info("CoorDL job complete")
    except Exception as exc:
        logger.exception("CoorDL job failed", exc_info=exc)
        raise exc
    finally:
        writer.close()


def _start_preparation_owners(
    *,
    context,
    plan,
    dataset: DatasetConfig,
    system: CoorDLSystemConfig,
    namespace: str,
    num_jobs: int,
):
    stop_event = context.Event()
    status_queue = context.Queue()
    processes = []

    for owner_index in range(num_jobs):
        process = context.Process(
            target=run_preparation_owner,
            kwargs={
                "owner_index": owner_index,
                "plan": plan,
                "dataset": dataset,
                "system": system,
                "namespace": namespace,
                "stop_event": stop_event,
                "status_queue": status_queue,
            },
            name=f"coordl-owner-{owner_index}",
        )
        process.start()
        processes.append(process)

    ready: set[int] = set()
    try:
        while len(ready) < num_jobs:
            try:
                status, owner_index, message = status_queue.get(
                    timeout=system.startup_timeout_seconds
                )
            except Empty as exc:
                raise RuntimeError("CoorDL preparation owners did not start in time") from exc

            if status == "error":
                raise RuntimeError(
                    f"CoorDL preparation owner {owner_index} failed to start: {message}"
                )

            if status == "ready":
                ready.add(int(owner_index))

        return processes, stop_event, status_queue

    except Exception:
        stop_event.set()
        for process in processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        status_queue.close()
        status_queue.join_thread()
        raise


def run_coordl_jobs(
    *,
    job_configs: tuple[JobConfig, ...],
    dataset_config: DatasetConfig,
    coordl_config: CoorDLSystemConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_batches = _validate_jobs(job_configs)
    num_jobs = len(job_configs)

    sample_count = dataset_sample_count(dataset_config)
    plan = build_coordl_batch_plan(
        num_samples=sample_count,
        batch_size=dataset_config.batch_size,
        total_batches=total_batches,
        num_jobs=num_jobs,
        shuffle=dataset_config.shuffle,
        drop_last=dataset_config.drop_last,
        seed=dataset_config.seed,
    )

    namespace = f"coordl:{uuid.uuid4().hex}"
    store = CoorDLStagingStore(coordl_config.redis, namespace)
    store.ping()
    store.clear_namespace()
    store.close()

    context = mp.get_context("spawn")
    owners, stop_event, status_queue = _start_preparation_owners(
        context=context,
        plan=plan,
        dataset=dataset_config,
        system=coordl_config,
        namespace=namespace,
        num_jobs=num_jobs,
    )

    LOGGER.info(
        "CoorDL coordinated prep started jobs=%d batches=%d staging_window=%d max_staged_batches=%d",
        num_jobs,
        len(plan),
        coordl_config.staging_window_per_owner,
        num_jobs * coordl_config.staging_window_per_owner,
    )

    try:
        errors: list[str] = []

        with ProcessPoolExecutor(max_workers=num_jobs, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _run_job,
                    job_index=index,
                    job=job,
                    dataset=dataset_config,
                    system=coordl_config,
                    plan=plan,
                    namespace=namespace,
                    num_jobs=num_jobs,
                    output_dir=output_dir,
                ): job.name
                for index, job in enumerate(job_configs)
            }

            for future in as_completed(futures):
                job_name = futures[future]
                try:
                    future.result()
                    LOGGER.info("CoorDL job finished job=%s", job_name)
                except Exception as exc:
                    LOGGER.exception("CoorDL job failed job=%s", job_name)
                    errors.append(f"{job_name}: {exc!r}")

        if errors:
            raise RuntimeError("one or more CoorDL jobs failed:\n" + "\n".join(errors))

    finally:
        stop_event.set()

        for process in owners:
            process.join(timeout=10.0)
            if process.is_alive():
                LOGGER.warning(
                    "CoorDL preparation owner did not stop cleanly; terminating pid=%s",
                    process.pid,
                )
                process.terminate()
                process.join(timeout=5.0)

        status_queue.close()
        status_queue.join_thread()

        cleanup_store = CoorDLStagingStore(coordl_config.redis, namespace)
        try:
            deleted = cleanup_store.clear_namespace()
            if deleted:
                LOGGER.info("Removed %d leftover CoorDL Redis key(s)", deleted)
        finally:
            cleanup_store.close()
