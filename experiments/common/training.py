from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from batchflow.config.config_types import DatasetConfig
from experiments.common.model_zoo import build_model, prepare_model_images
from experiments.config.types import JobConfig


@dataclass(frozen=True)
class batchResult:
    loss: float
    batch_size: int
    compute_time_sec: float
    h2d_time_sec: float
    forward_time_sec: float
    backward_time_sec: float
    optimizer_step_time_sec: float


class TrainingComponents:
    def __init__(
        self,
        *,
        job: JobConfig,
        dataset: DatasetConfig,
        device: torch.device,
    ) -> None:
        self.job = job
        self.dataset = dataset
        self.device = device
        self.task = job.task.strip().lower()

        if self.task == "classification":
            self.model = build_model(
                job.model_name,
                num_classes=dataset.num_classes,
            ).to(device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=job.learning_rate,
            )
            self.criterion: nn.Module | None = nn.CrossEntropyLoss()
        elif self.task == "retrieval":
            from experiments.models.albef import add_weight_decay, build_albef_retrieval_model

            vision_layers = _albef_vision_layers(job.model_name)
            self.model = build_albef_retrieval_model(
                vision_layers=vision_layers,
            ).to(device)
            self.optimizer = torch.optim.AdamW(
                add_weight_decay(self.model, job.weight_decay),
                lr=job.learning_rate,
            )
            self.criterion = None
        else:
            raise ValueError(f"unsupported training task {job.task!r}")

    def run_batch(
        self,
        batch: dict[str, Any],
        *,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
    ) -> batchResult:
        if self.task == "classification":
            return self._classification_batch(batch, scaler=scaler, amp_enabled=amp_enabled)
        return self._retrieval_batch(batch, scaler=scaler, amp_enabled=amp_enabled)

    def _classification_batch(
        self,
        batch: dict[str, Any],
        *,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
    ) -> batchResult:
        assert self.criterion is not None

        images = prepare_model_images(batch["image"])
        labels = batch["label"].long()
        images, labels, h2d_time = self._move_to_device(images, labels)

        return self._backprop_batch(
            forward=lambda: self.criterion(self.model(images), labels),
            batch_size=int(images.shape[0]),
            h2d_time=h2d_time,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

    def _retrieval_batch(
        self,
        batch: dict[str, Any],
        *,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
    ) -> batchResult:
        images = prepare_model_images(batch["image"])
        text = batch["text"].long()
        text_atts = batch["text_atts"].long()
        idx = batch["idx"].long()
        images, text, text_atts, idx, h2d_time = self._move_to_device(
            images, text, text_atts, idx
        )

        return self._backprop_batch(
            forward=lambda: self.model(
                images,
                text,
                text_atts,
                idx,
                self.job.alpha,
                is_train=True,
            ),
            batch_size=int(images.shape[0]),
            h2d_time=h2d_time,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

    def _move_to_device(self, *tensors: torch.Tensor) -> tuple[Any, ...]:
        start = time.perf_counter()
        non_blocking = self.device.type == "cuda"
        moved = tuple(
            tensor.to(self.device, non_blocking=non_blocking)
            for tensor in tensors
        )
        _cuda_sync(self.device)
        return (*moved, time.perf_counter() - start)

    def _backprop_batch(
        self,
        *,
        forward,
        batch_size: int,
        h2d_time: float,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
    ) -> batchResult:
        compute_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
            loss = forward()
        _cuda_sync(self.device)
        forward_time = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        _cuda_sync(self.device)
        backward_time = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        if amp_enabled:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        _cuda_sync(self.device)
        optimizer_time = time.perf_counter() - optimizer_start

        return batchResult(
            loss=float(loss.item()),
            batch_size=batch_size,
            compute_time_sec=time.perf_counter() - compute_start,
            h2d_time_sec=h2d_time,
            forward_time_sec=forward_time,
            backward_time_sec=backward_time,
            optimizer_step_time_sec=optimizer_time,
        )


def build_training_components(
    *,
    job: JobConfig,
    dataset: DatasetConfig,
    device: torch.device,
) -> TrainingComponents:
    return TrainingComponents(job=job, dataset=dataset, device=device)


def _albef_vision_layers(model_name: str) -> int:
    prefix = "albef_vision_"
    if not model_name.startswith(prefix):
        raise ValueError(
            f"retrieval model_name must look like {prefix}<layers>, got {model_name!r}"
        )

    try:
        return int(model_name.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"invalid ALBEF model_name {model_name!r}") from exc


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
