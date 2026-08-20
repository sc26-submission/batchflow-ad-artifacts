from __future__ import annotations

import random
import threading
from dataclasses import dataclass

from batchflow.common.core import Batch, Dataset, Job, Sample, now_ms


@dataclass(frozen=True)
class EpochPlan:
    plan_id: str
    dataset_id: str
    epoch: int
    sample_groups: list[list[Sample]]


class BatchCoordinator:
    def __init__(self, default_lookahead: int = 32) -> None:
        self.default_lookahead = default_lookahead
        self._epoch_plans_by_id: dict[str, EpochPlan] = {}
        self._lock = threading.RLock()

    def get_batches_for_job(
        self,
        dataset: Dataset,
        job: Job,
        *,
        reuse_enabled: bool = True,
    ) -> list[Batch]:
        epoch = job.progress.epoch
        start_index = max(0, job.progress.next_batch_index)
        lookahead = max(1, job.lookahead_batches or self.default_lookahead)

        epoch_plan = self._get_or_build_epoch_plan(dataset=dataset, epoch=epoch)
        stop_index = min(len(epoch_plan.sample_groups), start_index + lookahead)

        return [
            self._make_batch(
                dataset=dataset,
                epoch=epoch,
                batch_index=batch_index,
                samples=epoch_plan.sample_groups[batch_index],
                job_id=None if reuse_enabled else job.job_id,
            )
            for batch_index in range(start_index, stop_index)
        ]

    def _get_or_build_epoch_plan(self, *, dataset: Dataset, epoch: int) -> EpochPlan:
        plan_id = self._epoch_plan_id(dataset.dataset_id, epoch)

        with self._lock:
            existing = self._epoch_plans_by_id.get(plan_id)
            if existing is not None:
                return existing

        samples = list(dataset.samples)

        if dataset.shuffle:
            rng = random.Random(dataset.seed + epoch)
            rng.shuffle(samples)

        sample_groups = self._split_samples_into_batches(
            samples=samples,
            batch_size=dataset.batch_size,
            drop_last=dataset.drop_last,
        )

        plan = EpochPlan(
            plan_id=plan_id,
            dataset_id=dataset.dataset_id,
            epoch=epoch,
            sample_groups=sample_groups,
        )

        with self._lock:
            existing = self._epoch_plans_by_id.get(plan_id)
            if existing is not None:
                return existing

            self._epoch_plans_by_id[plan_id] = plan
            return plan

    def _split_samples_into_batches(
        self,
        *,
        samples: list[Sample],
        batch_size: int,
        drop_last: bool,
    ) -> list[list[Sample]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        groups: list[list[Sample]] = []

        for index in range(0, len(samples), batch_size):
            group = samples[index:index + batch_size]

            if len(group) < batch_size and drop_last:
                break

            if group:
                groups.append(group)

        return groups

    def _make_batch(
        self,
        *,
        dataset: Dataset,
        epoch: int,
        batch_index: int,
        samples: list[Sample],
        job_id: str | None = None,
    ) -> Batch:
        batch_id = self._batch_id(
            dataset_id=dataset.dataset_id,
            epoch=epoch,
            batch_index=batch_index,
            job_id=job_id,
        )

        return Batch(
            batch_id=batch_id,
            dataset_id=dataset.dataset_id,
            epoch=epoch,
            batch_index=batch_index,
            batch_size=dataset.batch_size,
            samples=samples,
            planned_at_ms=now_ms(),
            transform_name=dataset.transform_name,
            metadata=dict(dataset.metadata),
        )

    @staticmethod
    def _epoch_plan_id(dataset_id: str, epoch: int) -> str:
        return f"{dataset_id}:epoch-{epoch}"

    @staticmethod
    def _batch_id(
        *,
        dataset_id: str,
        epoch: int,
        batch_index: int,
        job_id: str | None = None,
    ) -> str:
        base = f"{dataset_id}:epoch-{epoch}:batch-{batch_index}"
        return f"{base}:job-{job_id}" if job_id else base