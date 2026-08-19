from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf


T = TypeVar("T")


@dataclass
class RedisConfig:
    """Shared Redis/ElastiCache configuration."""

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
    # static: each worker host is partitioned evenly across this many jobs.
    worker_assignment_mode: str = "shared"
    static_job_count: int = 4

    # The ablation policies disable cross-job prepared-batch sharing by
    # giving each job a private batch identity. When enabled, jobs following
    # the same epoch plan intentionally refer to the same batch/cache key.
    share_batches_across_jobs: bool = True

    # When false, every materialization is transient and reuse/cache scoring is
    # disabled. This is stronger than redis.enabled=false because BatchFlow can
    # otherwise reuse worker-local payloads even without Redis.
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

    # Fallback ready depth until online timing estimates are available.
    target_ready_batches: int = 16

    # Benefit-aware shared-cache management.
    cache_capacity_bytes: int = 0
    batch_value_beta: float = 1.0
    preparation_time_ema_alpha: float = 0.2

    # Spare workers may prepare farther-ahead reusable batches after the
    # immediate target lookahead of all active jobs is covered.
    opportunistic_prefetch_enabled: bool = True
    extended_lookahead_multiplier: int = 2


@dataclass
class CoordinatorConfig:
    # Local bind address for the coordinator gRPC server.
    host: str = "127.0.0.1"

    # Reachable address advertised to workers/trainers. When omitted, host is
    # used. This matters for disaggregated deployments where host=0.0.0.0.
    advertised_host: str | None = None

    port: int = 50051
    grpc_max_workers: int = 8
    default_lookahead: int = 32
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def bind_addr(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def addr(self) -> str:
        return f"{self.advertised_host or self.host}:{self.port}"


@dataclass
class DatasetConfig:
    """Dataset registration settings shared by BatchFlow and experiments."""

    dataset_id: str = "synthetic-torch"
    prefix_uri: str = "memory://synthetic-torch"
    split: str = "train"
    dataset_format: str = "synthetic_torch"
    transform_name: str | None = None
    text_transform_name: str | None = None

    # Optional metadata sources for non-ImageFolder datasets such as Open Images.
    annotations_uri: str | None = None
    class_descriptions_uri: str | None = None

    # Used by synthetic datasets. Real S3 datasets discover their sample count.
    num_samples: int = 256

    batch_size: int = 32
    drop_last: bool = False
    shuffle: bool = False
    seed: int = 123

    # Synthetic tensor shape and model metadata used by local examples/tests.
    input_shape: tuple[int, ...] = (3, 32, 32)
    num_classes: int = 10


@dataclass
class WorkerConfig:
    worker_id: str = "worker-0"
    coordinator_address: str = "127.0.0.1:50051"
    hostname: str = "127.0.0.1"
    fetch_host: str = "127.0.0.1"
    fetch_port: int = 0

    # Used only by the static-allocation ablation. None means the worker is
    # part of the shared pool.
    static_job_index: int | None = None

    poll_interval_seconds: float = 0.02
    heartbeat_interval_seconds: float = 2.0
    s3_fetch_threads: int = 4
    decode_threads: int = 4
    transient_ttl: int = 60

    # Populated from DeploymentConfig.redis by the launcher so each spawned
    # worker receives a self-contained configuration object.
    redis: RedisConfig = field(default_factory=RedisConfig)


@dataclass
class WorkerHostConfig:
    id: str = "local-node"

    # Address advertised to trainers for transient gRPC fetches.
    hostname: str = "127.0.0.1"

    worker_count: int = 1
    worker_port_start: int = 60061

    worker_config: WorkerConfig = field(default_factory=WorkerConfig)

    def worker_port(self, worker_index: int) -> int:
        return self.worker_port_start + worker_index


@dataclass
class DeploymentConfig:
    # co-located: coordinator and workers can be started together.
    # disaggregated: coordinator and worker roles are started independently.
    mode: str = "co-located"
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    worker_hosts: list[WorkerHostConfig] = field(default_factory=list)

    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    verify_connectivity: bool = True


def merge_dataclass(cls: type[T], cfg: DictConfig | dict) -> T:
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), cfg))


def parse_deployment_config(cfg: DictConfig) -> DeploymentConfig:
    """Merge topology and BatchFlow policy into one runtime configuration.

    Deployment files describe where the coordinator, workers, and optional
    Redis service live. Policy files describe how BatchFlow schedules work and
    whether cross-job reuse is enabled. Keeping these concerns separate avoids
    duplicating AWS addresses and worker topology across ablation stages.
    """

    merged = OmegaConf.merge(
        OmegaConf.structured(DeploymentConfig),
        cfg.deployment,
    )

    policy_cfg = cfg.get("policy")
    if policy_cfg is not None:
        merged.coordinator.scheduler = OmegaConf.merge(
            merged.coordinator.scheduler,
            policy_cfg,
        )

    # A deployment may contain a Redis endpoint because the topology supports
    # shared caching. Policies that disable caching should not connect to that
    # service at all.
    if not bool(merged.coordinator.scheduler.cache_enabled):
        merged.redis.enabled = False

    return OmegaConf.to_object(merged)


def parse_dataset_config(cfg: DictConfig) -> DatasetConfig:
    return merge_dataclass(DatasetConfig, cfg.dataset)


def make_worker_config(
    *,
    node: WorkerHostConfig,
    worker_index: int,
    coordinator_address: str,
    scheduler_config: SchedulerConfig,
    redis_config: RedisConfig | None = None,
) -> WorkerConfig:
    base = node.worker_config

    assignment_mode = str(scheduler_config.worker_assignment_mode).strip().lower()
    static_job_index: int | None = None

    if assignment_mode == "static":
        job_count = int(scheduler_config.static_job_count)
        if job_count <= 0:
            raise ValueError("static_job_count must be > 0 in static mode")
        if node.worker_count % job_count != 0:
            raise ValueError(
                f"worker_count must be divisible by static_job_count for "
                f"host={node.id!r}: worker_count={node.worker_count} "
                f"static_job_count={job_count}"
            )

        workers_per_job = node.worker_count // job_count
        static_job_index = worker_index // workers_per_job

    return WorkerConfig(
        worker_id=f"{node.id}-worker-{worker_index}",
        coordinator_address=coordinator_address,
        hostname=node.hostname,
        fetch_host=node.hostname,
        fetch_port=node.worker_port(worker_index),
        static_job_index=static_job_index,
        poll_interval_seconds=base.poll_interval_seconds,
        heartbeat_interval_seconds=base.heartbeat_interval_seconds,
        s3_fetch_threads=base.s3_fetch_threads,
        decode_threads=base.decode_threads,
        transient_ttl=base.transient_ttl,
        redis=redis_config or base.redis,
    )
