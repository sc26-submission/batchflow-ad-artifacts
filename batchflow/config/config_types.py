from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf


T = TypeVar("T")


@dataclass
class RedisConfig:
    """Shared Redis/ElastiCache configuration."""

    # Runtime-only flag. The topology describes where Redis is;
    # SchedulerConfig.cache_enabled decides whether it is used.
    enabled: bool = False

    host: str = ""
    port: int = 6379
    db: int = 0
    ssl: bool = False
    password: str = ""
    key_prefix: str = "batchflow"


@dataclass
class SchedulerConfig:
    """Coordinator scheduling and cache-policy defaults."""

    scheduling_strategy: str = "adaptive"

    # shared: every worker may serve any job.
    # static: workers on each node are partitioned evenly across jobs.
    worker_assignment_mode: str = "shared"
    static_job_count: int = 4

    # When enabled, jobs following the same epoch plan intentionally
    # refer to the same batch/cache identity.
    share_batches_across_jobs: bool = True

    # Controls both shared Redis caching and worker-local payload reuse.
    cache_enabled: bool = True

    candidate_window_size: int = 32

    job_urgency_weight: float = 8.0
    batch_proximity_weight: float = 4.0
    reuse_weight: float = 3.0
    materialization_cost_weight: float = 1.0
    ready_cache_bonus: float = 100.0

    reuse_threshold: float = 1.0
    cache_cost_threshold: float = 8.0
    trainer_bottleneck_weight: float = 0.5

    target_ready_batches: int = 16

    cache_capacity_bytes: int = 0
    batch_value_beta: float = 1.0
    preparation_time_ema_alpha: float = 0.2

    opportunistic_prefetch_enabled: bool = True
    extended_lookahead_multiplier: int = 2


@dataclass
class CoordinatorConfig:
    """Coordinator service placement and runtime settings."""

    # ID of the topology node that owns the coordinator.
    node: str = "local"

    port: int = 50051
    grpc_max_workers: int = 8
    default_lookahead: int = 32

    # Runtime-only policy populated by the launcher.
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

    # Kept for compatibility with the current worker implementation.
    # These can be renamed to advertised_host/bind_host in a later pass.
    hostname: str = "127.0.0.1"
    fetch_host: str = "127.0.0.1"
    fetch_port: int = 0

    # Used only by the static-allocation ablation.
    # None means the worker belongs to the shared pool.
    static_job_index: int | None = None

    poll_interval_seconds: float = 0.02
    heartbeat_interval_seconds: float = 2.0
    s3_fetch_threads: int = 4
    decode_threads: int = 4
    transient_ttl: int = 60

    # Populated by the launcher so each spawned worker receives
    # a self-contained Redis configuration.
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
        """Address workers/trainers use to reach the coordinator."""
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
    """Parse how BatchFlow schedules, reuses, and caches work."""
    policy_cfg = cfg.get("policy")

    if policy_cfg is None:
        return SchedulerConfig()

    return merge_dataclass(SchedulerConfig, policy_cfg)


def parse_dataset_config(cfg: DictConfig) -> DatasetConfig:
    return merge_dataclass(DatasetConfig, cfg.dataset)


def runtime_redis_config(
    topology: TopologyConfig,
    scheduler: SchedulerConfig,
) -> RedisConfig:
    """Create the effective Redis config for this run.

    The topology describes how Redis can be reached. The scheduling policy
    decides whether the current experiment actually uses it.
    """
    return replace(
        topology.redis,
        enabled=scheduler.cache_enabled,
    )


def coordinator_runtime_config(
    topology: TopologyConfig,
    scheduler: SchedulerConfig,
) -> CoordinatorConfig:
    """Attach the selected policy to the coordinator runtime config."""
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
    scheduler_config: SchedulerConfig,
    redis_config: RedisConfig,
) -> WorkerConfig:
    """Build the fully resolved config for one worker process."""

    if worker_index < 0 or worker_index >= node.worker_count:
        raise ValueError(
            f"Invalid worker_index={worker_index} for node={node_id!r} "
            f"with worker_count={node.worker_count}"
        )

    base = node.worker_config
    assignment_mode = scheduler_config.worker_assignment_mode.strip().lower()
    static_job_index: int | None = None

    if assignment_mode not in {"shared", "static"}:
        raise ValueError(
            f"Unknown worker_assignment_mode={assignment_mode!r}; "
            "expected 'shared' or 'static'"
        )

    if assignment_mode == "static":
        job_count = scheduler_config.static_job_count

        if job_count <= 0:
            raise ValueError("static_job_count must be > 0 in static mode")

        if node.worker_count % job_count != 0:
            raise ValueError(
                f"worker_count must be divisible by static_job_count for "
                f"node={node_id!r}: worker_count={node.worker_count}, "
                f"static_job_count={job_count}"
            )

        workers_per_job = node.worker_count // job_count
        static_job_index = worker_index // workers_per_job

    return WorkerConfig(
        worker_id=f"{node_id}-worker-{worker_index}",
        coordinator_address=coordinator_address,
        hostname=node.host,
        fetch_host=node.host,
        fetch_port=node.worker_port(worker_index),
        static_job_index=static_job_index,
        poll_interval_seconds=base.poll_interval_seconds,
        heartbeat_interval_seconds=base.heartbeat_interval_seconds,
        s3_fetch_threads=base.s3_fetch_threads,
        decode_threads=base.decode_threads,
        transient_ttl=base.transient_ttl,
        redis=redis_config,
    )