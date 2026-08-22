from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from batchflow.common.payload_formats import PayloadFormat
from batchflow.common.enums import StrEnum


def now_ms() -> int:
    return int(time.time() * 1000)


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def stable_hash(parts: Iterable[Any]) -> str:
    normalized = json.dumps(list(parts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class StorageClass(str, Enum):
    REUSABLE = "reusable"
    TRANSIENT = "transient"


class PayloadStatus(str, Enum):
    PENDING = "pending"
    AVAILABLE = "available"
    FAILED = "failed"
    EVICTED = "evicted"


class MaterializeBatchTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"




class JobStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Worker:
    worker_id: str
    hostname: str
    registered_at_ms: int = field(default_factory=now_ms)
    last_heartbeat_at_ms: int = field(default_factory=now_ms)
    active_task_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_heartbeat_at_ms = now_ms()


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_uri: str
    label: int | None = None
    class_name: str | None = None
    transform_name: str | None = None


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    samples: list[Sample]
    batch_size: int
    payload_format: PayloadFormat
    dataset_format: str

    drop_last: bool = False
    shuffle: bool = True
    seed: int = 0

    transform_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def batches_per_epoch(self) -> int:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        if self.drop_last:
            return self.sample_count // self.batch_size

        return (self.sample_count + self.batch_size - 1) // self.batch_size
    

@dataclass(frozen=True)
class Batch:
    batch_id: str
    dataset_id: str
    epoch: int
    batch_index: int
    batch_size: int
    samples: list[Sample]

    planned_at_ms: int = 0
    transform_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return self.batch_id

    def sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

@dataclass(frozen=True)
class MaterializeBatchTask:
    task_id: str
    batch: Batch
    job_id: str
    priority: float
    storage_class: StorageClass
    payload_format: PayloadFormat
    dataset_format: str
    created_at_ms: int
    not_before_ms: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PayloadEntry:
    """Coordinator metadata for a materialized payload, transient or reusable."""

    cache_key: str
    status: PayloadStatus
    storage_class: StorageClass
    location: str | None = None
    size_bytes: int = 0
    worker_id: str | None = None
    created_at_ms: int = field(default_factory=now_ms)
    last_accessed_at_ms: int = field(default_factory=now_ms)
    expires_at_ms: int | None = None
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_accessed_at_ms = now_ms()
        self.access_count += 1

    def is_expired(self, ts_ms: int | None = None) -> bool:
        if self.expires_at_ms is None:
            return False

        ts_ms = now_ms() if ts_ms is None else ts_ms
        return ts_ms >= self.expires_at_ms


@dataclass(frozen=True)
class JobProgress:
    epoch: int = 0
    next_batch_index: int = 0
    last_batch_id: str = ""
    last_batch_index: int = -1
    updated_at_ms: int = field(default_factory=now_ms)

    def advance(self, batch_id: str, batch_index: int) -> JobProgress:
        return replace(
            self,
            next_batch_index=batch_index + 1,
            last_batch_id=batch_id,
            last_batch_index=batch_index,
            updated_at_ms=now_ms(),
        )

    def advance_to_next_epoch(self) -> JobProgress:
        return replace(
            self,
            epoch=self.epoch + 1,
            next_batch_index=0,
            last_batch_id="",
            last_batch_index=-1,
            updated_at_ms=now_ms(),
        )


@dataclass
class Job:
    job_id: str
    dataset_id: str
    lookahead_batches: int = 32
    progress: JobProgress = field(default_factory=JobProgress)
    status: JobStatus = JobStatus.ACTIVE
    closed_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        dataset_id: str,
        lookahead_batches: int = 32,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        return cls(
            job_id=job_id,
            dataset_id=dataset_id,
            lookahead_batches=lookahead_batches,
            progress=JobProgress(),
            status=JobStatus.ACTIVE,
            closed_reason="",
            metadata=metadata or {},
        )

class BatchHandleStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    
@dataclass(frozen=True)
class BatchHandle:
    batch_id: str
    cache_key: str
    status: BatchHandleStatus

    # Required when status == READY.
    fetch_host: str = ""
    fetch_port: int = 0
    fetch_key: str = ""
    payload_format: PayloadFormat | None = None
    dataset_format: str = ""

    location: str = ""
    expires_at_ms: int | None = None

    # Optional/debug/runtime details only.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status == BatchHandleStatus.READY

    @property
    def is_pending(self) -> bool:
        return self.status == BatchHandleStatus.PENDING

    @property
    def is_failed(self) -> bool:
        return self.status == BatchHandleStatus.FAILED

    def require_ready_fetch_info(self) -> None:
        if not self.is_ready:
            raise ValueError(f"batch handle is not ready: status={self.status}")

        if not self.fetch_key:
            raise ValueError(
                f"ready batch handle missing fetch_key: batch_id={self.batch_id}"
            )

        if self.payload_format is None:
            raise ValueError(
                f"ready batch handle missing payload_format: batch_id={self.batch_id}"
            )

        if not self.dataset_format:
            raise ValueError(
                f"ready batch handle missing dataset_format: batch_id={self.batch_id}"
            )

        if self.location.startswith(("redis://", "rediss://")):
            return

        if self.location.startswith("grpc://") or not self.location:
            if not self.fetch_host:
                raise ValueError(
                    "ready gRPC batch handle missing fetch_host: "
                    f"batch_id={self.batch_id}"
                )

            if self.fetch_port <= 0:
                raise ValueError(
                    "ready gRPC batch handle missing/invalid fetch_port: "
                    f"batch_id={self.batch_id}"
                )
            return

        raise ValueError(
            f"unsupported batch location={self.location!r} "
            f"batch_id={self.batch_id}"
        )
