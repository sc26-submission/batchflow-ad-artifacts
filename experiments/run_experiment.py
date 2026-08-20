from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from batchflow.config.config_types import DatasetConfig
from batchflow.config.loaders import load_dataset_config
from batchflow.coordinator.dataset_builder import build_dataset_from_config
from batchflow.integrations.pytorch.config import BatchFlowTorchConfig
from experiments.common.reporting import ExperimentReporter
from experiments.config.types import (
    CoorDLRedisConfig,
    CoorDLSystemConfig,
    JobConfig,
    PytorchSystemConfig,
    TensorSocketSystemConfig,
)
from experiments.runners.batchflow_runner import run_batchflow_jobs
from experiments.runners.coordl_runner import run_coordl_jobs
from experiments.runners.pytorch_runner import run_pytorch_jobs
from experiments.runners.runner_core import configure_process_logging
from experiments.runners.tensorsocket_runner import run_tensorsocket_jobs


LOGGER = logging.getLogger("experiments.run_experiment")


def _resolve_dataset_metadata(dataset: DatasetConfig) -> DatasetConfig:
    """Resolve counts using the canonical dataset builder."""
    if not dataset.prefix_uri.startswith("s3://"):
        return dataset

    is_retrieval = dataset.dataset_format == "coco_retrieval"
    if dataset.num_samples > 0 and (is_retrieval or dataset.num_classes > 0):
        return dataset

    discovered = build_dataset_from_config(dataset)
    if is_retrieval:
        return replace(dataset, num_samples=discovered.sample_count)

    num_classes = int(discovered.metadata.get("num_classes", dataset.num_classes))
    if num_classes <= 0:
        raise ValueError(
            f"could not determine num_classes for dataset {dataset.dataset_id!r}"
        )

    return replace(
        dataset,
        num_samples=discovered.sample_count,
        num_classes=num_classes,
    )


def _plain_dict(cfg: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _build_jobs(cfg: DictConfig) -> tuple[JobConfig, ...]:
    defaults = _plain_dict(cfg.workload.get("job_defaults"))
    jobs: list[JobConfig] = []

    for job_cfg in cfg.workload.jobs:
        values = {**defaults, **_plain_dict(job_cfg)}
        jobs.append(
            JobConfig(
                name=str(values["name"]),
                model_name=str(values["model_name"]),
                num_batches=int(values["num_batches"]),
                warmup_batches=int(values["warmup_batches"]),
                learning_rate=float(values["learning_rate"]),
                task=str(values.get("task", "classification")),
                weight_decay=float(values.get("weight_decay", 0.0)),
                alpha=float(values.get("alpha", 0.4)),
                seed=int(values.get("seed", 123)),
                device=str(values.get("device", "auto")),
                use_amp=values.get("use_amp"),
            )
        )

    if not jobs:
        raise ValueError("workload must contain at least one job")

    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"job names must be unique, got {names}")

    return tuple(jobs)



def _build_batchflow_config(cfg: DictConfig) -> BatchFlowTorchConfig:
    return BatchFlowTorchConfig(**_plain_dict(cfg.system.get("client")))


def _build_pytorch_config(cfg: DictConfig) -> PytorchSystemConfig:
    values = _plain_dict(cfg.system)
    return PytorchSystemConfig(
        num_workers=int(values.get("num_workers", 4)),
        persistent_workers=bool(values.get("persistent_workers", True)),
    )


def _build_tensorsocket_config(cfg: DictConfig) -> TensorSocketSystemConfig:
    values = _plain_dict(cfg.system)
    return TensorSocketSystemConfig(
        num_workers=int(values.get("num_workers", 8)),
        persistent_workers=bool(values.get("persistent_workers", True)),
        host=str(values.get("host", "127.0.0.1")),
        port=int(values.get("port", 5555)),
        control_port=int(values.get("control_port", 5556)),
        consumer_buffer_size=int(values.get("consumer_buffer_size", 8)),
        max_lag_batches=int(values.get("max_lag_batches", 64)),
        startup_timeout_seconds=float(values.get("startup_timeout_seconds", 60.0)),
        receive_timeout_seconds=float(values.get("receive_timeout_seconds", 120.0)),
    )


def _build_coordl_config(cfg: DictConfig) -> CoorDLSystemConfig:
    values = _plain_dict(cfg.system)
    redis_values = _plain_dict(cfg.system.get("redis"))

    return CoorDLSystemConfig(
        num_workers=int(values.get("num_workers", 4)),
        persistent_workers=bool(values.get("persistent_workers", True)),
        staging_window_per_owner=int(values.get("staging_window_per_owner", 1)),
        poll_interval_seconds=float(values.get("poll_interval_seconds", 0.01)),
        wait_timeout_seconds=float(values.get("wait_timeout_seconds", 120.0)),
        startup_timeout_seconds=float(values.get("startup_timeout_seconds", 120.0)),
        redis=CoorDLRedisConfig(
            host=str(redis_values.get("host", "127.0.0.1")),
            port=int(redis_values.get("port", 6379)),
            db=int(redis_values.get("db", 0)),
            ssl=bool(redis_values.get("ssl", False)),
            password=str(redis_values.get("password", "")),
        ),
    )


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mp.freeze_support()
    configure_process_logging()

    system_name = str(cfg.system.name)
    workload_name = str(cfg.workload.name)
    # _validate_ablation_config(cfg, runner_name=runner_name)
    jobs = _build_jobs(cfg)
    dataset = load_dataset_config(str(cfg.workload.dataset))
    dataset = _resolve_dataset_metadata(dataset)

    OmegaConf.update(cfg, "resolved_dataset", asdict(dataset), force_add=True)

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    reporter = ExperimentReporter(
        results_dir=Path(cfg.workload.results_dir),
        workload_name=workload_name,
        system_name=system_name,
        dataset_config=dataset,
        cfg=cfg,
        run_id=run_id,
        run_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    reporter.save_resolved_config()

    LOGGER.info(
        "Starting experiment system=%s workload=%s dataset=%s jobs=%d",
        system_name, workload_name, dataset.dataset_id, len(jobs),
    )

    if system_name == "batchflow":
        run_batchflow_jobs(
            job_configs=jobs,
            dataset_config=dataset,
            batchflow_config=_build_batchflow_config(cfg),
            output_dir=reporter.run_dir,
        )
    elif system_name == "pytorch":
        run_pytorch_jobs(
            job_configs=jobs,
            dataset_config=dataset,
            pytorch_config=_build_pytorch_config(cfg),
            output_dir=reporter.run_dir,
        )
    elif system_name == "tensorsocket":
        run_tensorsocket_jobs(
            job_configs=jobs,
            dataset_config=dataset,
            tensorsocket_config=_build_tensorsocket_config(cfg),
            output_dir=reporter.run_dir,
        )
    elif system_name == "coordl":
        run_coordl_jobs(
            job_configs=jobs,
            dataset_config=dataset,
            coordl_config=_build_coordl_config(cfg),
            output_dir=reporter.run_dir,
        )
    else:
        raise ValueError(
            f"unsupported system={system_name!r} (expected one of 'batchflow', 'pytorch', 'tensorsocket', 'coordl')")

    reporter.write_summaries(jobs=jobs, mode=system_name)
    LOGGER.info("Experiment complete results=%s", reporter.run_dir.resolve())


if __name__ == "__main__":
    main()
