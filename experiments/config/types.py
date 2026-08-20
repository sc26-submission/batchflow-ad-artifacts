from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PytorchSystemConfig:
    """Settings specific to the regular PyTorch DataLoader baseline."""

    num_workers: int = 4
    persistent_workers: bool = True


@dataclass(frozen=True)
class TensorSocketSystemConfig:
    """Settings for the TensorSocket producer/consumer baseline."""

    num_workers: int = 8
    persistent_workers: bool = True
    host: str = "127.0.0.1"
    port: int = 5555
    control_port: int = 5556
    consumer_buffer_size: int = 8
    max_lag_batches: int = 64
    startup_timeout_seconds: float = 600.0
    receive_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class CoorDLRedisConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    ssl: bool = False
    password: str = ""


@dataclass(frozen=True)
class CoorDLSystemConfig:
    """Settings for the CoorDL coordinated-preprocessing baseline."""

    num_workers: int = 4
    persistent_workers: bool = True
    staging_window_per_owner: int = 1
    poll_interval_seconds: float = 0.01
    wait_timeout_seconds: float = 120.0
    startup_timeout_seconds: float = 120.0
    redis: CoorDLRedisConfig = field(default_factory=CoorDLRedisConfig)


@dataclass(frozen=True)
class JobConfig:
    """Training settings shared by every evaluated data-loading system."""

    name: str
    model_name: str
    num_batches: int
    warmup_batches: int
    learning_rate: float
    task: str = "classification"
    weight_decay: float = 0.0
    alpha: float = 0.4

    # Kept internal for reproducibility; normally omitted from workload YAMLs.
    seed: int = 123
    device: str = "auto"
    use_amp: bool | None = None
