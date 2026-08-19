from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import grpc
import hydra
from omegaconf import DictConfig

from batchflow.common.core import Dataset
from batchflow.config.config_types import (
    CoordinatorConfig,
    DatasetConfig,
    DeploymentConfig,
    RedisConfig,
    WorkerConfig,
    WorkerHostConfig,
    make_worker_config,
    parse_dataset_config,
    parse_deployment_config,
)
from batchflow.coordinator.dataset_builder import build_dataset_from_config
from batchflow.coordinator.grpc_service import serve_grpc
from batchflow.coordinator.service import CoordinatorService
from batchflow.data_worker.data_worker import DataWorker


LOGGER = logging.getLogger("batchflow.launch")


ROLE_ALL = "all"
ROLE_COORDINATOR = "coordinator"
ROLE_WORKER = "worker"
VALID_ROLES = {ROLE_ALL, ROLE_COORDINATOR, ROLE_WORKER}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    return getattr(logging, str(level).upper(), logging.INFO)


def _reenable_batchflow_loggers(log_level: int) -> None:
    logging.disable(logging.NOTSET)

    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if not name.startswith("batchflow"):
            continue

        if isinstance(logger_obj, logging.Logger):
            logger_obj.disabled = False
            logger_obj.setLevel(log_level)
            logger_obj.propagate = True

    batchflow_logger = logging.getLogger("batchflow")
    batchflow_logger.disabled = False
    batchflow_logger.setLevel(log_level)
    batchflow_logger.propagate = True


def setup_logging(
    base_dir: str | Path = "logs",
    level: str | int = "INFO",
) -> Path:
    run_dir = Path(base_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "batchflow.log"
    log_level = _log_level(level)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    _reenable_batchflow_loggers(log_level)

    LOGGER.info("Logging initialized log_file=%s level=%s", log_file, level)

    return run_dir


def setup_worker_logging(
    worker_id: str,
    run_dir: str | Path,
    level: str | int = "INFO",
) -> None:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / f"worker-{worker_id}.log"
    log_level = _log_level(level)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    _reenable_batchflow_loggers(log_level)

    logging.getLogger("batchflow.worker_process").info(
        "Worker logging initialized worker_id=%s log_file=%s level=%s",
        worker_id,
        log_file,
        level,
    )


# ---------------------------------------------------------------------------
# Runtime handles
# ---------------------------------------------------------------------------


class RuntimeHandle(Protocol):
    def stop(self) -> None:
        ...


@dataclass
class LocalCoordinatorHandle:
    server: Any
    coordinator_service: CoordinatorService
    _stopped: bool = False

    def stop(self) -> None:
        if self._stopped:
            return

        self._stopped = True
        LOGGER.info("Stopping coordinator")
        self.coordinator_service.shutdown()
        self.server.stop(0)
        LOGGER.info("Coordinator stopped")


@dataclass
class ManagedProcessHandle:
    name: str
    process: mp.Process
    _stopped: bool = False

    def stop(self) -> None:
        if self._stopped:
            return

        self._stopped = True
        LOGGER.info("Stopping worker %s", self.name)

        if self.process.is_alive():
            self.process.terminate()

        self.process.join(timeout=5.0)

        if self.process.is_alive():
            LOGGER.warning("Worker did not stop cleanly, killing %s", self.name)
            self.process.kill()
            self.process.join(timeout=5.0)

        LOGGER.info(
            "Worker stopped name=%s exitcode=%s",
            self.name,
            self.process.exitcode,
        )


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------


def build_registered_dataset_from_config(
    dataset_config: DatasetConfig,
    force_rebuild: bool = False,
) -> Dataset:
    LOGGER.info("Preparing dataset %s", dataset_config.dataset_id)
    LOGGER.info("  source:     %s", dataset_config.prefix_uri)
    LOGGER.info("  split:      %s", dataset_config.split)
    LOGGER.info("  format:     %s", dataset_config.dataset_format)
    LOGGER.info("  transform:  %s", dataset_config.transform_name)
    LOGGER.info("  batch size: %s", dataset_config.batch_size)
    LOGGER.info("  drop last:  %s", dataset_config.drop_last)
    LOGGER.info("  shuffle:    %s", dataset_config.shuffle)
    LOGGER.info("  seed:       %s", dataset_config.seed)

    return build_dataset_from_config(
        dataset_config,
        force_rebuild=force_rebuild,
    )


# ---------------------------------------------------------------------------
# Coordinator launch
# ---------------------------------------------------------------------------


def start_coordinator(
    coordinator_config: CoordinatorConfig,
    dataset_config: DatasetConfig,
    redis_config: RedisConfig,
) -> LocalCoordinatorHandle:
    LOGGER.info("Starting coordinator")
    LOGGER.info("  bind address:       %s", coordinator_config.bind_addr)
    LOGGER.info("  advertised address: %s", coordinator_config.addr)
    LOGGER.info("  grpc max workers:   %s", coordinator_config.grpc_max_workers)
    LOGGER.info(
        "  scheduler:          %s",
        coordinator_config.scheduler.scheduling_strategy,
    )
    LOGGER.info(
        "  worker assignment:  %s",
        coordinator_config.scheduler.worker_assignment_mode,
    )
    LOGGER.info(
        "  cross-job batches:  %s",
        "shared"
        if coordinator_config.scheduler.share_batches_across_jobs
        else "private",
    )
    LOGGER.info(
        "  reusable cache:     %s",
        "enabled" if coordinator_config.scheduler.cache_enabled else "disabled",
    )

    coordinator_service = CoordinatorService(
        config=coordinator_config,
        redis_config=redis_config,
    )

    dataset = build_registered_dataset_from_config(dataset_config)
    coordinator_service.register_dataset(dataset)

    LOGGER.info(
        "Dataset ready id=%s samples=%s batch_size=%s "
        "dataset_format=%s payload_format=%s",
        dataset.dataset_id,
        dataset.sample_count,
        dataset.batch_size,
        dataset.dataset_format,
        dataset.payload_format.value,
    )

    server = serve_grpc(
        coordinator_service,
        host=coordinator_config.host,
        port=coordinator_config.port,
        grpc_max_workers=coordinator_config.grpc_max_workers,
    )

    LOGGER.info("Coordinator ready")

    return LocalCoordinatorHandle(
        server=server,
        coordinator_service=coordinator_service,
    )


def launch_coordinator_only(
    *,
    deployment_config: DeploymentConfig,
    dataset_config: DatasetConfig,
) -> list[RuntimeHandle]:
    handle = start_coordinator(
        coordinator_config=deployment_config.coordinator,
        dataset_config=dataset_config,
        redis_config=deployment_config.redis,
    )

    return [handle]


# ---------------------------------------------------------------------------
# Worker launch
# ---------------------------------------------------------------------------


def worker_process_entry(
    worker_config: WorkerConfig,
    run_dir: str,
    log_level: str | int = "INFO",
) -> None:
    setup_worker_logging(
        worker_id=worker_config.worker_id,
        run_dir=run_dir,
        level=log_level,
    )

    logger = logging.getLogger("batchflow.worker_process")

    logger.info("Starting worker process")
    logger.info("  worker_id:   %s", worker_config.worker_id)
    logger.info("  coordinator: %s", worker_config.coordinator_address)
    logger.info("  hostname:    %s", worker_config.hostname)
    logger.info(
        "  fetch_addr:  %s:%s",
        worker_config.fetch_host,
        worker_config.fetch_port,
    )
    logger.info(
        "  static_job:  %s",
        worker_config.static_job_index
        if worker_config.static_job_index is not None
        else "shared",
    )
    logger.info(
        "  redis:       %s",
        "enabled" if worker_config.redis.enabled else "disabled",
    )

    if worker_config.redis.enabled:
        logger.info(
            "  redis_addr:  %s:%s db=%s ssl=%s",
            worker_config.redis.host,
            worker_config.redis.port,
            worker_config.redis.db,
            worker_config.redis.ssl,
        )

    worker = DataWorker(
        worker_id=worker_config.worker_id,
        coordinator_address=worker_config.coordinator_address,
        hostname=worker_config.hostname,
        fetch_host=worker_config.fetch_host,
        fetch_port=worker_config.fetch_port,
        static_job_index=worker_config.static_job_index,
        poll_interval_seconds=worker_config.poll_interval_seconds,
        heartbeat_interval_seconds=worker_config.heartbeat_interval_seconds,
        s3_fetch_threads=worker_config.s3_fetch_threads,
        decode_threads=worker_config.decode_threads,
        transient_ttl=worker_config.transient_ttl,
        redis_config=worker_config.redis,
    )

    try:
        worker.start()
        logger.info("Worker ready worker_id=%s", worker_config.worker_id)

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info(
            "Worker received KeyboardInterrupt worker_id=%s",
            worker_config.worker_id,
        )

    finally:
        logger.info("Worker shutting down worker_id=%s", worker_config.worker_id)
        worker.stop()
        logger.info(
            "Worker shutdown complete worker_id=%s",
            worker_config.worker_id,
        )


def launch_local_worker_process(
    *,
    worker_config: WorkerConfig,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> ManagedProcessHandle:
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=worker_process_entry,
        args=(worker_config, str(run_dir), log_level),
        name=worker_config.worker_id,
        daemon=True,
    )

    process.start()

    return ManagedProcessHandle(
        name=worker_config.worker_id,
        process=process,
    )


def _wait_for_grpc_endpoint(
    address: str,
    *,
    timeout_seconds: float,
) -> None:
    LOGGER.info(
        "Waiting for coordinator connectivity address=%s timeout=%.1fs",
        address,
        timeout_seconds,
    )

    channel = grpc.insecure_channel(address)

    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_seconds)
    except grpc.FutureTimeoutError as exc:
        raise RuntimeError(
            f"Coordinator is not reachable at {address} after "
            f"{timeout_seconds:.1f} seconds"
        ) from exc
    finally:
        channel.close()

    LOGGER.info("Coordinator reachable address=%s", address)


def _resolve_worker_host(
    deployment_config: DeploymentConfig,
    worker_host_id: str | None,
) -> WorkerHostConfig:
    hosts = deployment_config.worker_hosts

    if worker_host_id:
        for host in hosts:
            if host.id == worker_host_id:
                return host

        available = ", ".join(host.id for host in hosts) or "<none>"
        raise ValueError(
            f"Unknown worker_host_id={worker_host_id!r}. "
            f"Available worker hosts: {available}"
        )

    if len(hosts) == 1:
        return hosts[0]

    if not hosts:
        raise ValueError("No worker_hosts are configured")

    available = ", ".join(host.id for host in hosts)
    raise ValueError(
        "Multiple worker hosts are configured. Specify worker_host_id on the "
        f"worker machine. Available worker hosts: {available}"
    )


def launch_worker_host(
    *,
    deployment_config: DeploymentConfig,
    worker_host: WorkerHostConfig,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> list[RuntimeHandle]:
    coordinator_address = deployment_config.coordinator.addr

    if deployment_config.verify_connectivity:
        _wait_for_grpc_endpoint(
            coordinator_address,
            timeout_seconds=deployment_config.startup_timeout_seconds,
        )

    LOGGER.info("Starting worker host id=%s", worker_host.id)
    LOGGER.info("  advertised hostname: %s", worker_host.hostname)
    LOGGER.info("  workers:             %s", worker_host.worker_count)
    if (
        str(deployment_config.coordinator.scheduler.worker_assignment_mode)
        .strip()
        .lower()
        == "static"
    ):
        static_job_count = int(
            deployment_config.coordinator.scheduler.static_job_count
        )
        LOGGER.info(
            "  static workers/job:  %s",
            worker_host.worker_count // static_job_count,
        )
    LOGGER.info("  coordinator:         %s", coordinator_address)

    handles: list[RuntimeHandle] = []

    for worker_index in range(worker_host.worker_count):
        worker_config = make_worker_config(
            node=worker_host,
            worker_index=worker_index,
            coordinator_address=coordinator_address,
            scheduler_config=deployment_config.coordinator.scheduler,
            redis_config=deployment_config.redis,
        )

        worker_handle = launch_local_worker_process(
            worker_config=worker_config,
            run_dir=run_dir,
            log_level=log_level,
        )
        handles.append(worker_handle)

    return handles


def launch_workers_only(
    *,
    deployment_config: DeploymentConfig,
    worker_host_id: str | None,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> list[RuntimeHandle]:
    worker_host = _resolve_worker_host(
        deployment_config,
        worker_host_id,
    )

    return launch_worker_host(
        deployment_config=deployment_config,
        worker_host=worker_host,
        run_dir=run_dir,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# Co-located launch
# ---------------------------------------------------------------------------


def launch_colocated_batchflow(
    *,
    deployment_config: DeploymentConfig,
    dataset_config: DatasetConfig,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> list[RuntimeHandle]:
    run_dir.mkdir(parents=True, exist_ok=True)

    handles: list[RuntimeHandle] = []

    coordinator_handle = start_coordinator(
        coordinator_config=deployment_config.coordinator,
        dataset_config=dataset_config,
        redis_config=deployment_config.redis,
    )
    handles.append(coordinator_handle)

    # Give the local server a brief moment to begin accepting connections.
    time.sleep(0.5)

    worker_count = sum(
        host.worker_count for host in deployment_config.worker_hosts
    )
    LOGGER.info("Starting %s co-located worker(s)", worker_count)

    for worker_host in deployment_config.worker_hosts:
        worker_handles = launch_worker_host(
            deployment_config=deployment_config,
            worker_host=worker_host,
            run_dir=run_dir,
            log_level=log_level,
        )
        handles.extend(worker_handles)

    return handles


# ---------------------------------------------------------------------------
# Launch validation / shutdown
# ---------------------------------------------------------------------------


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")

    if normalized == "colocated":
        normalized = "co-located"

    return normalized


def _validate_launch_request(
    *,
    deployment_config: DeploymentConfig,
    role: str,
) -> None:
    if role not in VALID_ROLES:
        raise ValueError(
            f"Unsupported role={role!r}. Expected one of: "
            f"{', '.join(sorted(VALID_ROLES))}"
        )

    mode = _normalize_mode(deployment_config.mode)

    if mode not in {"co-located", "disaggregated"}:
        raise ValueError(
            f"Unsupported deployment mode={deployment_config.mode!r}. "
            "Expected 'co-located' or 'disaggregated'."
        )

    if mode == "disaggregated" and role == ROLE_ALL:
        raise ValueError(
            "A disaggregated deployment must be launched by role. "
            "Use role=coordinator on the coordinator/training machine and "
            "role=worker on each data-worker machine."
        )

    assignment_mode = str(
        deployment_config.coordinator.scheduler.worker_assignment_mode
    ).strip().lower()

    if assignment_mode == "static":
        static_job_count = int(
            deployment_config.coordinator.scheduler.static_job_count
        )
        if static_job_count <= 0:
            raise ValueError(
                "static worker assignment requires scheduler.static_job_count > 0"
            )

        for worker_host in deployment_config.worker_hosts:
            if worker_host.worker_count % static_job_count != 0:
                raise ValueError(
                    f"worker_count must be divisible by static_job_count for "
                    f"host={worker_host.id!r}: "
                    f"worker_count={worker_host.worker_count} "
                    f"static_job_count={static_job_count}"
                )

    if role == ROLE_WORKER:
        coordinator_host = (
            deployment_config.coordinator.advertised_host
            or deployment_config.coordinator.host
        )

        if coordinator_host in {"0.0.0.0", "::", "[::]"}:
            raise ValueError(
                "Workers cannot connect to coordinator host "
                f"{coordinator_host!r}. Set coordinator.advertised_host to "
                "the coordinator machine's reachable private IP or DNS name."
            )


def stop_all(handles: list[RuntimeHandle]) -> None:
    if not handles:
        return

    LOGGER.info("Stopping BatchFlow")

    for handle in reversed(handles):
        try:
            handle.stop()
        except Exception:
            LOGGER.exception("Failed to stop runtime handle %s", handle)

    LOGGER.info("BatchFlow stopped")


def _install_signal_handlers(handles: list[RuntimeHandle]) -> None:
    def _handle_signal(signum, frame) -> None:  # noqa: ARG001
        LOGGER.info("Received signal=%s, stopping BatchFlow", signum)
        stop_all(handles)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    run_dir_value = (
        cfg.get("run_dir", None)
        or cfg.get("runtime_dir", None)
        or "logs/batchflow"
    )

    logging_cfg = cfg.get("logging", {})
    log_level = (
        logging_cfg.get("level", "INFO")
        if logging_cfg is not None
        else "INFO"
    )

    run_dir = setup_logging(
        base_dir=run_dir_value,
        level=log_level,
    )

    deployment_config = parse_deployment_config(cfg)
    dataset_config = parse_dataset_config(cfg)

    role = str(cfg.get("role", ROLE_ALL)).strip().lower()

    raw_worker_host_id = cfg.get("worker_host_id", None)
    worker_host_id = (
        str(raw_worker_host_id).strip()
        if raw_worker_host_id not in (None, "", "null")
        else None
    )

    _validate_launch_request(
        deployment_config=deployment_config,
        role=role,
    )

    LOGGER.info("BatchFlow launch configuration")
    LOGGER.info("  deployment mode: %s", deployment_config.mode)
    LOGGER.info("  role:            %s", role)

    if worker_host_id is not None:
        LOGGER.info("  worker host id:  %s", worker_host_id)

    handles: list[RuntimeHandle] = []

    try:
        if role == ROLE_ALL:
            handles = launch_colocated_batchflow(
                deployment_config=deployment_config,
                dataset_config=dataset_config,
                run_dir=run_dir,
                log_level=log_level,
            )

        elif role == ROLE_COORDINATOR:
            handles = launch_coordinator_only(
                deployment_config=deployment_config,
                dataset_config=dataset_config,
            )

        elif role == ROLE_WORKER:
            handles = launch_workers_only(
                deployment_config=deployment_config,
                worker_host_id=worker_host_id,
                run_dir=run_dir,
                log_level=log_level,
            )

        _install_signal_handlers(handles)

        LOGGER.info("BatchFlow running. Press Ctrl+C to stop.")

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        LOGGER.info("BatchFlow received KeyboardInterrupt")

    except Exception:
        LOGGER.exception("BatchFlow deployment failed")
        raise

    finally:
        stop_all(handles)


if __name__ == "__main__":
    mp.freeze_support()
    main()
