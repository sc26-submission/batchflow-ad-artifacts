from __future__ import annotations

import logging
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from batchflow.config.config_types import DatasetConfig
from experiments.baselines.coordl.plan import CoorDLBatchPlanEntry
from experiments.baselines.coordl.staging import CoorDLStagingStore
from experiments.config.types import CoorDLSystemConfig
from experiments.runners.runner_core import configure_process_logging, set_seed
from experiments.torch_datasets.factory import build_torch_dataset


class _OwnedBatchSampler:
    def __init__(self, entries: tuple[CoorDLBatchPlanEntry, ...]) -> None:
        self.entries = entries

    def __iter__(self) -> Iterator[list[int]]:
        for entry in self.entries:
            yield list(entry.sample_indices)

    def __len__(self) -> int:
        return len(self.entries)


def build_coordl_dataset(dataset: DatasetConfig):
    if not dataset.prefix_uri.startswith("s3://"):
        raise ValueError(
            f"CoorDL experiment datasets must use an S3 URI, got {dataset.prefix_uri!r}"
        )

    return build_torch_dataset(dataset)


def dataset_sample_count(dataset: DatasetConfig) -> int:
    return len(build_coordl_dataset(dataset))


def _build_owner_dataloader(
    *,
    dataset: DatasetConfig,
    system: CoorDLSystemConfig,
    entries: tuple[CoorDLBatchPlanEntry, ...],
) -> DataLoader:
    torch_dataset = build_coordl_dataset(dataset)
    kwargs: dict[str, Any] = {
        "dataset": torch_dataset,
        "batch_sampler": _OwnedBatchSampler(entries),
        "num_workers": system.num_workers,
        "pin_memory": False,
    }

    if system.num_workers > 0:
        kwargs["persistent_workers"] = system.persistent_workers
        kwargs["prefetch_factor"] = 1

    return DataLoader(**kwargs)


def _sum_metric(batch: dict[str, Any], key: str) -> float:
    value = batch.get(key)
    if isinstance(value, torch.Tensor):
        return float(value.sum().item())
    if value is None:
        return 0.0
    return float(value)


def _serialize_batch(
    batch: dict[str, Any],
    *,
    entry: CoorDLBatchPlanEntry,
    prep_time_sec: float,
) -> bytes:
    if all(key in batch for key in ("image", "text", "text_atts", "idx")):
        data = {
            "image": batch["image"].contiguous(),
            "text": batch["text"].long().contiguous(),
            "text_atts": batch["text_atts"].long().contiguous(),
            "idx": batch["idx"].long().contiguous(),
        }
    elif all(key in batch for key in ("image", "label")):
        data = {
            "image": batch["image"].contiguous(),
            "label": batch["label"].long().contiguous(),
        }
    else:
        raise TypeError("CoorDL owner received an unsupported batch structure")

    payload = {
        **data,
        "batch_indices": batch.get("index").tolist(),
        "io_time_sec": _sum_metric(batch, "io_time_sec"),
        "decode_time_sec": _sum_metric(batch, "decode_time_sec"),
        "transform_time_sec": _sum_metric(batch, "transform_time_sec"),
        "prep_time_sec": prep_time_sec,
        "owner_job_index": entry.owner_index,
    }

    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _deserialize_batch(payload: bytes) -> dict[str, Any]:
    batch = torch.load(BytesIO(payload), map_location="cpu")
    if not isinstance(batch, dict):
        raise TypeError(f"invalid CoorDL payload type: {type(batch)!r}")
    return batch


def run_preparation_owner(
    *,
    owner_index: int,
    plan: tuple[CoorDLBatchPlanEntry, ...],
    dataset: DatasetConfig,
    system: CoorDLSystemConfig,
    namespace: str,
    stop_event: Any,
    status_queue: Any,
) -> None:
    configure_process_logging()
    logger = logging.getLogger(f"experiments.baselines.coordl.owner.{owner_index}")

    store = CoorDLStagingStore(system.redis, namespace)
    owned = tuple(entry for entry in plan if entry.owner_index == owner_index)

    try:
        set_seed(dataset.seed + owner_index)
        store.ping()

        dataloader = _build_owner_dataloader(
            dataset=dataset,
            system=system,
            entries=owned,
        )
        iterator = iter(dataloader)

        status_queue.put(("ready", owner_index, ""))
        logger.info(
            "CoorDL preparation owner ready owner=%d batches=%d workers=%d window=%d",
            owner_index,
            len(owned),
            system.num_workers,
            system.staging_window_per_owner,
        )

        pending: deque[str] = deque()

        for entry in owned:
            while len(pending) >= system.staging_window_per_owner:
                oldest = pending[0]
                if not store.exists(oldest):
                    pending.popleft()
                    continue

                if stop_event.is_set():
                    return

                time.sleep(system.poll_interval_seconds)

            if stop_event.is_set():
                return

            prep_start = time.perf_counter()
            batch = next(iterator)
            prep_time_sec = time.perf_counter() - prep_start

            payload = _serialize_batch(
                batch,
                entry=entry,
                prep_time_sec=prep_time_sec,
            )
            store.publish(entry.batch_id, payload)
            pending.append(entry.batch_id)

        while pending and not stop_event.is_set():
            if not store.exists(pending[0]):
                pending.popleft()
            else:
                time.sleep(system.poll_interval_seconds)

        logger.info("CoorDL preparation owner complete owner=%d", owner_index)

    except BaseException as exc:
        status_queue.put(("error", owner_index, repr(exc)))
        logger.exception("CoorDL preparation owner failed owner=%d", owner_index)
        raise

    finally:
        store.close()


class CoorDLBatchIterator:
    """Consume the shared, bounded CoorDL minibatch sequence.

    Every training job consumes the same minibatches in the same sequence.
    Preparation is partitioned across owners. The per-owner staging window
    bounds how far preparation can advance ahead of the slowest consumer.
    """

    def __init__(
        self,
        *,
        plan: tuple[CoorDLBatchPlanEntry, ...],
        system: CoorDLSystemConfig,
        namespace: str,
        job_index: int,
        job_id: str,
        num_jobs: int,
    ) -> None:
        self.plan = plan
        self.system = system
        self.job_index = job_index
        self.job_id = job_id
        self.num_jobs = num_jobs
        self.position = 0
        self.store = CoorDLStagingStore(system.redis, namespace)
        self.store.ping()

    def __iter__(self) -> CoorDLBatchIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self.position >= len(self.plan):
            raise StopIteration

        entry = self.plan[self.position]
        self.position += 1

        wait_start = time.perf_counter()
        payload = self.store.wait_and_consume(
            entry.batch_id,
            num_consumers=self.num_jobs,
            poll_interval_seconds=self.system.poll_interval_seconds,
            timeout_seconds=self.system.wait_timeout_seconds,
        )
        wait_time_sec = time.perf_counter() - wait_start

        deserialize_start = time.perf_counter()
        batch = _deserialize_batch(payload)
        deserialize_time_sec = time.perf_counter() - deserialize_start

        batch.update(
            {
                "batch_index": entry.batch_index,
                "batch_id": entry.batch_id,
                "job_id": self.job_id,
                "coordl_wait_time_sec": wait_time_sec,
                "coordl_deserialize_time_sec": deserialize_time_sec,
                "coordl_payload_bytes": len(payload),
                "coordl_is_owner": int(entry.owner_index == self.job_index),
            }
        )
        return batch

    def close(self) -> None:
        self.store.close()
