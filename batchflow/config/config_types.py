from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf


T = TypeVar("T")


@dataclass
class RedisConfig:
    """Shared Redis/ElastiCache configuration."""

    # Runtime-only flag. The topology describes where Redis is;
    # SchedulerConfig.reuse_enabled decides whether it is used.
    enabled: bool = False

    host: str = ""
    port: int = 6379
    db: int = 0
    ssl: bool = False
    password: str = ""
    key_prefix: str = "batchflow"


@dataclass
class SchedulerConfig:
    """BatchFlow scheduling and reuse settings."""

    reuse_enabled: bool = True

    candidate_window_size: int = 32
    target_ready_batches: int = 16

    job_urgency_weight: float = 8.0
    batch_proximity_weight: float = 4.0
    reuse_weight: float = 3.0
    materialization_cost_weight: float = 1.0
    ready_cache_bonus: float = 100.0

    reuse_threshold: float = 1.0
    cache_cost_threshold: float = 8.0

    cache_capacity_bytes: int = 0
    batch_value_beta: float = 1.0
    preparation_time_ema_alpha: float = 0.2

    opportunistic_prefetch_enabled: bool = True
    extended_lookahead_multiplier: int = 2


@dataclass
class CoordinatorConfig:
    """Coordinator service placement and runtime settings."""

    # Topology node that owns the coordinator.
    node: str = "local"

    port: int = 50051
    grpc_max_workers: int = 8
    default_lookahead: int = 32

    # Runtime-only scheduler settings populated by the launcher.
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class DatasetConfig:
    """Dataset registration settings shared by BatchFlow and experiments."""

    dataset_id: str = "synthetic-torch"
    prefix_uri: str = "memory://synthetic-torch"
    split: str = "train"
    dataset_format: str = "synthetic_torch"
    transform_name: str | None = None
    text_transform_name: str | None = None

    annotations_uri: str | None = None
    class_descriptions_uri: str | None = None

    num_samples: int = 256
    batch_size: int = 32
    drop_last: bool = False
    shuffle: bool = False
    seed: int = 123

    input_shape: tuple[int, ...] = (3, 32, 32)
    num_classes: int = 10


@dataclass
class WorkerConfig:
    """Fully resolved configuration for one worker process."""

    worker_id: str = "worker-0"
    coordinator_address: str = "127.0.0.1:50051"

    hostname: str = "127.0.0.1"
    fetch_host: str = "127.0.0.1"
    fetch_port: int = 0

    poll_interval_seconds: float = 0.02
    heartbeat_interval_seconds: float = 2.0
    s3_fetch_threads: int = 4
    decode_threads: int = 4
    transient_ttl: int = 60

    redis: RedisConfig = field(default_factory=RedisConfig)


@dataclass
class NodeConfig:
    """One machine participating in the BatchFlow topology."""

    # Address other BatchFlow components use to reach this machine.
    host: str = "127.0.0.1"

    worker_count: int = 0
    worker_port_start: int = 60061

    # Common settings inherited by workers launched on this node.
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)

    def worker_port(self, worker_index: int) -> int:
        return self.worker_port_start + worker_index


@dataclass
class TopologyConfig:
    """Placement of BatchFlow coordinator, workers, and Redis."""

    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)

    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    verify_connectivity: bool = True

    def coordinator_address(self) -> str:
        """Address workers and trainers use to reach the coordinator."""

        if self.coordinator.node not in self.nodes:
            available = ", ".join(self.nodes) or "<none>"
            raise ValueError(
                f"Coordinator node {self.coordinator.node!r} is not defined. "
                f"Available nodes: {available}"
            )

        node = self.nodes[self.coordinator.node]
        return f"{node.host}:{self.coordinator.port}"


def merge_dataclass(cls: type[T], cfg: DictConfig | dict) -> T:
    return OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(cls), cfg)
    )


def parse_topology_config(cfg: DictConfig) -> TopologyConfig:
    """Parse where BatchFlow services run."""
    return merge_dataclass(TopologyConfig, cfg.topology)


def parse_scheduler_config(cfg: DictConfig) -> SchedulerConfig:
    """Parse BatchFlow scheduler settings."""
    return merge_dataclass(SchedulerConfig, cfg.scheduler)


def parse_dataset_config(cfg: DictConfig) -> DatasetConfig:
    """Parse dataset settings."""
    return merge_dataclass(DatasetConfig, cfg.dataset)


def runtime_redis_config(
    topology: TopologyConfig,
    scheduler: SchedulerConfig,
) -> RedisConfig:
    """Build the effective Redis configuration for this run."""

    return replace(
        topology.redis,
        enabled=scheduler.reuse_enabled,
    )


def coordinator_runtime_config(
    topology: TopologyConfig,
    scheduler: SchedulerConfig,
) -> CoordinatorConfig:
    """Attach scheduler settings to the coordinator runtime config."""

    return replace(
        topology.coordinator,
        scheduler=scheduler,
    )


def make_worker_config(
    *,
    node_id: str,
    node: NodeConfig,
    worker_index: int,
    coordinator_address: str,
    redis_config: RedisConfig,
) -> WorkerConfig:
    """Build the fully resolved config for one worker process."""

    if worker_index < 0 or worker_index >= node.worker_count:
        raise ValueError(
            f"Invalid worker_index={worker_index} for node={node_id!r} "
            f"with worker_count={node.worker_count}"
        )

    base = node.worker_config

    return WorkerConfig(
        worker_id=f"{node_id}-worker-{worker_index}",
        coordinator_address=coordinator_address,
        hostname=node.host,
        fetch_host=node.host,
        fetch_port=node.worker_port(worker_index),
        poll_interval_seconds=base.poll_interval_seconds,
        heartbeat_interval_seconds=base.heartbeat_interval_seconds,
        s3_fetch_threads=base.s3_fetch_threads,
        decode_threads=base.decode_threads,
        transient_ttl=base.transient_ttl,
        redis=redis_config,
    )