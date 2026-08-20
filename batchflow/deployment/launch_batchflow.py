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
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def _file_handler(path: Path, level: int) -> logging.FileHandler:
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _stream_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _clear_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def _enable_batchflow_loggers() -> None:
    """Undo any logger disabling performed by Hydra."""
    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if name.startswith("batchflow") and isinstance(logger_obj, logging.Logger):
            logger_obj.disabled = False


def setup_logging(base_dir: str | Path = "logs", level: str | int = "INFO") -> Path:
    run_dir = Path(base_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_level = _log_level(level)

    # Third-party libraries only emit warnings/errors.
    root = logging.getLogger()
    _clear_handlers(root)
    root.setLevel(logging.WARNING)

    # Re-enable loggers that Hydra may have disabled.
    _enable_batchflow_loggers()

    # Parent BatchFlow logger handles console output.
    batchflow_logger = logging.getLogger("batchflow")
    batchflow_logger.disabled = False
    _clear_handlers(batchflow_logger)
    batchflow_logger.setLevel(log_level)
    batchflow_logger.propagate = False
    batchflow_logger.addHandler(_stream_handler(log_level))

    # Launcher logs -> batchflow.log
    launch_logger = logging.getLogger("batchflow.launch")
    launch_logger.disabled = False
    _clear_handlers(launch_logger)
    launch_logger.setLevel(log_level)
    launch_logger.propagate = True
    launch_logger.addHandler(_file_handler(run_dir / "batchflow.log", log_level))

    # Coordinator logs -> coordinator.log
    coordinator_logger = logging.getLogger("batchflow.coordinator")
    coordinator_logger.disabled = False
    _clear_handlers(coordinator_logger)
    coordinator_logger.setLevel(log_level)
    coordinator_logger.propagate = True
    coordinator_logger.addHandler(_file_handler(run_dir / "coordinator.log", log_level))

    # Some coordinator child loggers are created before setup_logging().
    _enable_batchflow_loggers()

    LOGGER.info(f"Logging initialized | dir={run_dir} | level={level}")
    return run_dir


def setup_worker_logging(
    worker_id: str,
    run_dir: str | Path,
    level: str | int = "INFO",
) -> None:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_level = _log_level(level)

    root = logging.getLogger()
    _clear_handlers(root)
    root.setLevel(logging.WARNING)

    _enable_batchflow_loggers()

    batchflow_logger = logging.getLogger("batchflow")
    batchflow_logger.disabled = False
    _clear_handlers(batchflow_logger)
    batchflow_logger.setLevel(log_level)
    batchflow_logger.propagate = False
    batchflow_logger.addHandler(_stream_handler(log_level))
    batchflow_logger.addHandler(
        _file_handler(run_dir / f"worker-{worker_id}.log", log_level)
    )

    _enable_batchflow_loggers()

    logging.getLogger("batchflow.worker").info(
        f"Logging initialized | worker={worker_id} | level={level}"
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
            LOGGER.warning(
                f"Worker did not stop cleanly, killing {self.name}"
            )
            self.process.kill()
            self.process.join(timeout=5.0)

        LOGGER.info(
            f"Worker stopped | name={self.name} | "
            f"exitcode={self.process.exitcode}"
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
    LOGGER.info(
        f"  bind address:     0.0.0.0:{coordinator_config.port}"
    )
    LOGGER.info(
        f"  grpc max workers: {coordinator_config.grpc_max_workers}"
    )
    LOGGER.info(
        f"  batch reuse:      "
        f"{'enabled' if coordinator_config.scheduler.reuse_enabled else 'disabled'}"
    )

    if redis_config.enabled:
        LOGGER.info(
            f"  redis:            "
            f"{redis_config.host}:{redis_config.port} "
            f"db={redis_config.db} ssl={redis_config.ssl}"
        )

    coordinator_service = CoordinatorService(
        config=coordinator_config,
        redis_config=redis_config,
    )

    dataset = build_registered_dataset_from_config(dataset_config)
    coordinator_service.register_dataset(dataset)

    LOGGER.info(
        f"Dataset ready | id={dataset.dataset_id} | "
        f"samples={dataset.sample_count} | "
        f"batch_size={dataset.batch_size} | "
        f"format={dataset.dataset_format} | "
        f"payload={dataset.payload_format.value}"
    )

    # The coordinator always listens on all local interfaces.
    # Other nodes connect using TopologyConfig.coordinator_address().
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

    logger = logging.getLogger("batchflow.worker")

    logger.info(f"Starting worker {worker_config.worker_id}")
    logger.info(
        f"  coordinator: {worker_config.coordinator_address}"
    )
    logger.info(f"  hostname:    {worker_config.hostname}")
    logger.info(
        f"  fetch addr:  "
        f"{worker_config.fetch_host}:{worker_config.fetch_port}"
    )
    logger.info(
        f"  redis:       "
        f"{'enabled' if worker_config.redis.enabled else 'disabled'}"
    )

    if worker_config.redis.enabled:
        logger.info(
            f"  redis addr:  "
            f"{worker_config.redis.host}:{worker_config.redis.port} "
            f"db={worker_config.redis.db} ssl={worker_config.redis.ssl}"
        )

    worker = DataWorker(
        worker_id=worker_config.worker_id,
        coordinator_address=worker_config.coordinator_address,
        hostname=worker_config.hostname,
        fetch_host=worker_config.fetch_host,
        fetch_port=worker_config.fetch_port,

        poll_interval_seconds=worker_config.poll_interval_seconds,
        heartbeat_interval_seconds=worker_config.heartbeat_interval_seconds,
        s3_fetch_threads=worker_config.s3_fetch_threads,
        decode_threads=worker_config.decode_threads,
        transient_ttl=worker_config.transient_ttl,
        redis_config=worker_config.redis,
    )

    try:
        worker.start()

        logger.info(
            f"Worker ready {worker_config.worker_id}"
        )

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info(
            f"Worker received KeyboardInterrupt "
            f"{worker_config.worker_id}"
        )

    finally:
        logger.info(
            f"Worker shutting down {worker_config.worker_id}"
        )

        worker.stop()

        logger.info(
            f"Worker shutdown complete {worker_config.worker_id}"
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
        f"Waiting for coordinator | address={address} | "
        f"timeout={timeout_seconds:.1f}s"
    )

    channel = grpc.insecure_channel(address)

    try:
        grpc.channel_ready_future(channel).result(
            timeout=timeout_seconds
        )

    except grpc.FutureTimeoutError as exc:
        raise RuntimeError(
            f"Coordinator is not reachable at {address} after "
            f"{timeout_seconds:.1f} seconds"
        ) from exc

    finally:
        channel.close()

    LOGGER.info(
        f"Coordinator reachable | address={address}"
    )


def launch_workers(
    *,
    node_id: str,
    node: NodeConfig,
    coordinator_address: str,
    redis_config: RedisConfig,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> list[RuntimeHandle]:
    if node.worker_count <= 0:
        return []

    LOGGER.info(
        f"Starting workers | node={node_id} | "
        f"host={node.host} | "
        f"workers={node.worker_count} | "
        f"coordinator={coordinator_address}"
    )

    handles: list[RuntimeHandle] = []

    for worker_index in range(node.worker_count):
        worker_config = make_worker_config(
            node_id=node_id,
            node=node,
            worker_index=worker_index,
            coordinator_address=coordinator_address,
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
    node_id: str,
) -> NodeConfig:
    if topology.coordinator.node not in topology.nodes:
        available = ", ".join(topology.nodes) or "<none>"

        raise ValueError(
            f"Coordinator node {topology.coordinator.node!r} "
            f"is not defined. Available nodes: {available}"
        )

    if node_id not in topology.nodes:
        available = ", ".join(topology.nodes) or "<none>"

        raise ValueError(
            f"Unknown node_id={node_id!r}. "
            f"Available nodes: {available}"
        )

    node = topology.nodes[node_id]

    if node.worker_count < 0:
        raise ValueError(
            f"worker_count must be >= 0 for node={node_id!r}, "
            f"got {node.worker_count}"
        )

    if not node.host.strip():
        raise ValueError(
            f"Node {node_id!r} must define a reachable host"
        )

    if not 1 <= topology.coordinator.port <= 65535:
        raise ValueError(
            f"Invalid coordinator port="
            f"{topology.coordinator.port}"
        )

    if node.worker_count > 0:
        last_worker_port = node.worker_port(
            node.worker_count - 1
        )

        if (
            not 1 <= node.worker_port_start <= 65535
            or last_worker_port > 65535
        ):
            raise ValueError(
                f"Invalid worker port range for node={node_id!r}: "
                f"{node.worker_port_start}-{last_worker_port}"
            )

    owns_coordinator = (
        topology.coordinator.node == node_id
    )

    if not owns_coordinator and node.worker_count == 0:
        raise ValueError(
            f"Node {node_id!r} has no BatchFlow services "
            f"configured: it is not the coordinator node "
            f"and worker_count=0"
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
        node_id=node_id,
    )

    coordinator_address = topology.coordinator_address()
    redis_config = runtime_redis_config(
        topology,
        scheduler,
    )

    owns_coordinator = (
        topology.coordinator.node == node_id
    )

    LOGGER.info(f"Launching node {node_id}")
    LOGGER.info(f"  host:        {node.host}")
    LOGGER.info(
        f"  coordinator: {coordinator_address}"
    )
    LOGGER.info(
        f"  owns coord:  {owns_coordinator}"
    )
    LOGGER.info(
        f"  workers:     {node.worker_count}"
    )

    handles: list[RuntimeHandle] = []

    if owns_coordinator:
        coordinator_config = coordinator_runtime_config(
            topology,
            scheduler,
        )

        handles.append(
            start_coordinator(
                coordinator_config=coordinator_config,
                dataset_config=dataset,
                redis_config=redis_config,
            )
        )

    # If this node starts workers, wait until the coordinator is ready.
    #
    # For the coordinator node this confirms that its local server is
    # accepting connections before workers start. For remote worker
    # nodes it verifies connectivity to the coordinator machine.
    if (
        node.worker_count > 0
        and (
            owns_coordinator
            or topology.verify_connectivity
        )
    ):
        _wait_for_grpc_endpoint(
            coordinator_address,
            timeout_seconds=(
                topology.startup_timeout_seconds
            ),
        )

    handles.extend(
        launch_workers(
            node_id=node_id,
            node=node,
            coordinator_address=coordinator_address,
            redis_config=redis_config,
            run_dir=run_dir,
            log_level=log_level,
        )
    )

    return handles


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def stop_all(
    handles: list[RuntimeHandle],
) -> None:
    if not handles:
        return

    LOGGER.info("Stopping BatchFlow")

    for handle in reversed(handles):
        try:
            handle.stop()

        except Exception:
            LOGGER.exception(
                f"Failed to stop runtime handle {handle}"
            )

    LOGGER.info("BatchFlow stopped")


def _install_signal_handlers(
    handles: list[RuntimeHandle],
) -> None:
    def _handle_signal(
        signum,
        frame,
    ) -> None:  # noqa: ARG001
        LOGGER.info(
            f"Received signal={signum}, stopping BatchFlow"
        )

        stop_all(handles)

        raise SystemExit(
            128 + int(signum)
        )

    signal.signal(
        signal.SIGINT,
        _handle_signal,
    )

    signal.signal(
        signal.SIGTERM,
        _handle_signal,
    )


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(
    cfg: DictConfig,
) -> None:
    run_dir = setup_logging(
        base_dir=cfg.logging.dir,
        level=cfg.logging.level,
    )

    node_id = str(cfg.node_id).strip()

    topology = parse_topology_config(cfg)
    scheduler = parse_scheduler_config(cfg)
    dataset = parse_dataset_config(cfg)

    LOGGER.info(
        "BatchFlow launch configuration"
    )
    LOGGER.info(
        f"  node:        {node_id}"
    )
    LOGGER.info(
        f"  coordinator: "
        f"{topology.coordinator_address()}"
    )
    LOGGER.info(
        f"  reuse:       "
        f"{'enabled' if scheduler.reuse_enabled else 'disabled'}"
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

        LOGGER.info(
            "BatchFlow running. Press Ctrl+C to stop."
        )

        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        LOGGER.info(
            "BatchFlow received KeyboardInterrupt"
        )

    except Exception:
        LOGGER.exception(
            "BatchFlow launch failed"
        )
        raise

    finally:
        stop_all(handles)


if __name__ == "__main__":
    mp.freeze_support()
    main()