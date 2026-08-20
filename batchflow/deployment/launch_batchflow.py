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
    NodeConfig,
    RedisConfig,
    SchedulerConfig,
    TopologyConfig,
    WorkerConfig,
    coordinator_runtime_config,
    make_worker_config,
    parse_dataset_config,
    parse_scheduler_config,
    parse_topology_config,
    runtime_redis_config,
)
from batchflow.coordinator.dataset_builder import build_dataset_from_config
from batchflow.coordinator.grpc_service import serve_grpc
from batchflow.coordinator.service import CoordinatorService
from batchflow.data_worker.data_worker import DataWorker


LOGGER = logging.getLogger("batchflow.launch")


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
        if name.startswith("batchflow") and isinstance(logger_obj, logging.Logger):
            logger_obj.disabled = False
            logger_obj.setLevel(log_level)
            logger_obj.propagate = True

    batchflow_logger = logging.getLogger("batchflow")
    batchflow_logger.disabled = False
    batchflow_logger.setLevel(log_level)
    batchflow_logger.propagate = True


def _configure_logging(log_file: Path, level: str | int) -> None:
    log_level = _log_level(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    _reenable_batchflow_loggers(log_level)


def setup_logging(
    base_dir: str | Path = "logs",
    level: str | int = "INFO",
) -> Path:
    run_dir = Path(base_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "batchflow.log"
    _configure_logging(log_file, level)

    LOGGER.info(f"Logging initialized | file={log_file} | level={level}")
    return run_dir


def setup_worker_logging(
    worker_id: str,
    run_dir: str | Path,
    level: str | int = "INFO",
) -> None:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / f"worker-{worker_id}.log"
    _configure_logging(log_file, level)

    logging.getLogger("batchflow.worker_process").info(
        f"Worker logging initialized | worker={worker_id} | "
        f"file={log_file} | level={level}"
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
        LOGGER.info(f"Stopping worker {self.name}")

        if self.process.is_alive():
            self.process.terminate()

        self.process.join(timeout=5.0)

        if self.process.is_alive():
            LOGGER.warning(f"Worker did not stop cleanly, killing {self.name}")
            self.process.kill()
            self.process.join(timeout=5.0)

        LOGGER.info(
            f"Worker stopped | name={self.name} | exitcode={self.process.exitcode}"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_registered_dataset_from_config(
    dataset_config: DatasetConfig,
    force_rebuild: bool = False,
) -> Dataset:
    LOGGER.info(f"Preparing dataset {dataset_config.dataset_id}")
    LOGGER.info(f"  source:     {dataset_config.prefix_uri}")
    LOGGER.info(f"  split:      {dataset_config.split}")
    LOGGER.info(f"  format:     {dataset_config.dataset_format}")
    LOGGER.info(f"  transform:  {dataset_config.transform_name}")
    LOGGER.info(f"  batch size: {dataset_config.batch_size}")
    LOGGER.info(f"  drop last:  {dataset_config.drop_last}")
    LOGGER.info(f"  shuffle:    {dataset_config.shuffle}")
    LOGGER.info(f"  seed:       {dataset_config.seed}")

    return build_dataset_from_config(
        dataset_config,
        force_rebuild=force_rebuild,
    )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def start_coordinator(
    coordinator_config: CoordinatorConfig,
    dataset_config: DatasetConfig,
    redis_config: RedisConfig,
) -> LocalCoordinatorHandle:
    LOGGER.info("Starting coordinator")
    LOGGER.info(f"  bind address:      0.0.0.0:{coordinator_config.port}")
    LOGGER.info(f"  grpc max workers:  {coordinator_config.grpc_max_workers}")
    LOGGER.info(f"  scheduler:         {coordinator_config.scheduler.scheduling_strategy}")
    LOGGER.info(
        f"  worker assignment: {coordinator_config.scheduler.worker_assignment_mode}"
    )
    LOGGER.info(
        "  cross-job batches: "
        + (
            "shared"
            if coordinator_config.scheduler.share_batches_across_jobs
            else "private"
        )
    )
    LOGGER.info(
        "  reusable cache:    "
        + ("enabled" if coordinator_config.scheduler.cache_enabled else "disabled")
    )

    if redis_config.enabled:
        LOGGER.info(
            f"  redis:             {redis_config.host}:{redis_config.port} "
            f"db={redis_config.db} ssl={redis_config.ssl}"
        )

    coordinator_service = CoordinatorService(
        config=coordinator_config,
        redis_config=redis_config,
    )

    dataset = build_registered_dataset_from_config(dataset_config)
    coordinator_service.register_dataset(dataset)

    LOGGER.info(
        f"Dataset ready | id={dataset.dataset_id} | samples={dataset.sample_count} | "
        f"batch_size={dataset.batch_size} | format={dataset.dataset_format} | "
        f"payload={dataset.payload_format.value}"
    )

    # Binding is an implementation detail. Other nodes connect using
    # TopologyConfig.coordinator_address().
    server = serve_grpc(
        coordinator_service,
        host="0.0.0.0",
        port=coordinator_config.port,
        grpc_max_workers=coordinator_config.grpc_max_workers,
    )

    LOGGER.info("Coordinator ready")
    return LocalCoordinatorHandle(
        server=server,
        coordinator_service=coordinator_service,
    )


# ---------------------------------------------------------------------------
# Workers
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

    logger.info(f"Starting worker {worker_config.worker_id}")
    logger.info(f"  coordinator: {worker_config.coordinator_address}")
    logger.info(f"  hostname:    {worker_config.hostname}")
    logger.info(f"  fetch addr:  {worker_config.fetch_host}:{worker_config.fetch_port}")
    logger.info(
        "  assignment:  "
        + (
            str(worker_config.static_job_index)
            if worker_config.static_job_index is not None
            else "shared"
        )
    )
    logger.info(
        "  redis:       "
        + ("enabled" if worker_config.redis.enabled else "disabled")
    )

    if worker_config.redis.enabled:
        logger.info(
            f"  redis addr:  {worker_config.redis.host}:{worker_config.redis.port} "
            f"db={worker_config.redis.db} ssl={worker_config.redis.ssl}"
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
        logger.info(f"Worker ready {worker_config.worker_id}")

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info(f"Worker received KeyboardInterrupt {worker_config.worker_id}")

    finally:
        logger.info(f"Worker shutting down {worker_config.worker_id}")
        worker.stop()
        logger.info(f"Worker shutdown complete {worker_config.worker_id}")


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
        f"Waiting for coordinator | address={address} | "
        f"timeout={timeout_seconds:.1f}s"
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

    LOGGER.info(f"Coordinator reachable | address={address}")


def launch_workers(
    *,
    node_id: str,
    node: NodeConfig,
    coordinator_address: str,
    scheduler_config: SchedulerConfig,
    redis_config: RedisConfig,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> list[RuntimeHandle]:
    if node.worker_count <= 0:
        return []

    LOGGER.info(
        f"Starting workers | node={node_id} | host={node.host} | "
        f"workers={node.worker_count} | coordinator={coordinator_address}"
    )

    if scheduler_config.worker_assignment_mode.strip().lower() == "static":
        LOGGER.info(
            f"  static jobs:        {scheduler_config.static_job_count}"
        )
        LOGGER.info(
            f"  workers/job:        "
            f"{node.worker_count // scheduler_config.static_job_count}"
        )

    handles: list[RuntimeHandle] = []

    for worker_index in range(node.worker_count):
        worker_config = make_worker_config(
            node_id=node_id,
            node=node,
            worker_index=worker_index,
            coordinator_address=coordinator_address,
            scheduler_config=scheduler_config,
            redis_config=redis_config,
        )

        handles.append(
            launch_local_worker_process(
                worker_config=worker_config,
                run_dir=run_dir,
                log_level=log_level,
            )
        )

    return handles


# ---------------------------------------------------------------------------
# Node launch
# ---------------------------------------------------------------------------


def _validate_node_launch(
    *,
    topology: TopologyConfig,
    scheduler: SchedulerConfig,
    node_id: str,
) -> NodeConfig:
    if topology.coordinator.node not in topology.nodes:
        available = ", ".join(topology.nodes) or "<none>"
        raise ValueError(
            f"Coordinator node {topology.coordinator.node!r} is not defined. "
            f"Available nodes: {available}"
        )

    if node_id not in topology.nodes:
        available = ", ".join(topology.nodes) or "<none>"
        raise ValueError(
            f"Unknown node_id={node_id!r}. Available nodes: {available}"
        )

    node = topology.nodes[node_id]

    if node.worker_count < 0:
        raise ValueError(
            f"worker_count must be >= 0 for node={node_id!r}, "
            f"got {node.worker_count}"
        )

    if not node.host.strip():
        raise ValueError(f"Node {node_id!r} must define a reachable host")

    if not 1 <= topology.coordinator.port <= 65535:
        raise ValueError(
            f"Invalid coordinator port={topology.coordinator.port}"
        )

    if node.worker_count > 0:
        last_worker_port = node.worker_port(node.worker_count - 1)

        if not 1 <= node.worker_port_start <= 65535 or last_worker_port > 65535:
            raise ValueError(
                f"Invalid worker port range for node={node_id!r}: "
                f"{node.worker_port_start}-{last_worker_port}"
            )

    assignment_mode = scheduler.worker_assignment_mode.strip().lower()

    if assignment_mode not in {"shared", "static"}:
        raise ValueError(
            f"Unknown worker_assignment_mode={assignment_mode!r}; "
            "expected 'shared' or 'static'"
        )

    if assignment_mode == "static" and node.worker_count > 0:
        if scheduler.static_job_count <= 0:
            raise ValueError(
                "static worker assignment requires static_job_count > 0"
            )

        if node.worker_count % scheduler.static_job_count != 0:
            raise ValueError(
                f"worker_count must be divisible by static_job_count for "
                f"node={node_id!r}: worker_count={node.worker_count}, "
                f"static_job_count={scheduler.static_job_count}"
            )

    owns_coordinator = topology.coordinator.node == node_id

    if not owns_coordinator and node.worker_count == 0:
        raise ValueError(
            f"Node {node_id!r} has no BatchFlow services configured: "
            "it is not the coordinator node and worker_count=0"
        )

    return node


def launch_node(
    *,
    topology: TopologyConfig,
    scheduler: SchedulerConfig,
    dataset: DatasetConfig,
    node_id: str,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> list[RuntimeHandle]:
    node = _validate_node_launch(
        topology=topology,
        scheduler=scheduler,
        node_id=node_id,
    )

    coordinator_address = topology.coordinator_address()
    redis_config = runtime_redis_config(topology, scheduler)
    owns_coordinator = topology.coordinator.node == node_id

    LOGGER.info(f"Launching node {node_id}")
    LOGGER.info(f"  host:        {node.host}")
    LOGGER.info(f"  coordinator: {coordinator_address}")
    LOGGER.info(f"  owns coord:  {owns_coordinator}")
    LOGGER.info(f"  workers:     {node.worker_count}")

    handles: list[RuntimeHandle] = []

    if owns_coordinator:
        coordinator_config = coordinator_runtime_config(topology, scheduler)

        handles.append(
            start_coordinator(
                coordinator_config=coordinator_config,
                dataset_config=dataset,
                redis_config=redis_config,
            )
        )

    # Wait before spawning workers. This handles both:
    #   1. local/co-located coordinator + workers
    #   2. workers connecting to a coordinator on another node
    if node.worker_count > 0 and (owns_coordinator or topology.verify_connectivity):
        _wait_for_grpc_endpoint(
            coordinator_address,
            timeout_seconds=topology.startup_timeout_seconds,
        )

    handles.extend(
        launch_workers(
            node_id=node_id,
            node=node,
            coordinator_address=coordinator_address,
            scheduler_config=scheduler,
            redis_config=redis_config,
            run_dir=run_dir,
            log_level=log_level,
        )
    )

    return handles


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def stop_all(handles: list[RuntimeHandle]) -> None:
    if not handles:
        return

    LOGGER.info("Stopping BatchFlow")

    for handle in reversed(handles):
        try:
            handle.stop()
        except Exception:
            LOGGER.exception(f"Failed to stop runtime handle {handle}")

    LOGGER.info("BatchFlow stopped")


def _install_signal_handlers(handles: list[RuntimeHandle]) -> None:
    def _handle_signal(signum, frame) -> None:  # noqa: ARG001
        LOGGER.info(f"Received signal={signum}, stopping BatchFlow")
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
    run_dir = setup_logging(
        base_dir=cfg.logging.dir,
        level=cfg.logging.level,
    )

    node_id = str(cfg.node_id).strip()
    topology = parse_topology_config(cfg)
    scheduler = parse_scheduler_config(cfg)
    dataset = parse_dataset_config(cfg)

    LOGGER.info("BatchFlow launch configuration")
    LOGGER.info(f"  node:        {node_id}")
    LOGGER.info(f"  coordinator: {topology.coordinator_address()}")
    LOGGER.info(f"  policy:      {scheduler.scheduling_strategy}")
    LOGGER.info(
        f"  cache:       {'enabled' if scheduler.cache_enabled else 'disabled'}"
    )

    handles: list[RuntimeHandle] = []

    try:
        handles = launch_node(
            topology=topology,
            scheduler=scheduler,
            dataset=dataset,
            node_id=node_id,
            run_dir=run_dir,
            log_level=cfg.logging.level,
        )

        _install_signal_handlers(handles)

        LOGGER.info("BatchFlow running. Press Ctrl+C to stop.")

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        LOGGER.info("BatchFlow received KeyboardInterrupt")

    except Exception:
        LOGGER.exception("BatchFlow launch failed")
        raise

    finally:
        stop_all(handles)


if __name__ == "__main__":
    mp.freeze_support()
    main()