from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CoorDLBatchPlanEntry:
    batch_id: str
    epoch: int
    batch_index: int
    owner_index: int
    sample_indices: tuple[int, ...]


def build_coordl_batch_plan(
    *,
    num_samples: int,
    batch_size: int,
    total_batches: int,
    num_jobs: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> tuple[CoorDLBatchPlanEntry, ...]:
    """Build the shared minibatch sequence consumed by every CoorDL job.

    The dataset is randomized once per epoch. Minibatches from that shared
    sequence are assigned round-robin to preparation owners, while every job
    consumes every minibatch exactly once.
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if total_batches <= 0:
        raise ValueError(f"total_batches must be > 0, got {total_batches}")
    if num_jobs <= 0:
        raise ValueError(f"num_jobs must be > 0, got {num_jobs}")

    plan: list[CoorDLBatchPlanEntry] = []
    epoch = 0

    while len(plan) < total_batches:
        if shuffle:
            generator = torch.Generator().manual_seed(seed + epoch)
            indices = torch.randperm(num_samples, generator=generator).tolist()
        else:
            indices = list(range(num_samples))

        batch_index = 0
        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            if len(batch_indices) < batch_size and drop_last:
                continue

            plan.append(
                CoorDLBatchPlanEntry(
                    batch_id=f"e{epoch}_b{batch_index}",
                    epoch=epoch,
                    batch_index=batch_index,
                    owner_index=batch_index % num_jobs,
                    sample_indices=tuple(batch_indices),
                )
            )
            batch_index += 1

            if len(plan) >= total_batches:
                break

        if batch_index == 0:
            raise ValueError(
                "dataset does not contain a complete minibatch with the "
                "current batch_size/drop_last settings"
            )

        epoch += 1

    return tuple(plan)
