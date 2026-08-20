"""Minimal local PyTorch smoke test for BatchFlow.

This demo runs entirely on one machine and does not require AWS, S3, Redis,
or a GPU. Edit the settings below to make the test larger if needed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from batchflow.config.config_types import (
    CoordinatorConfig,
    DatasetConfig,
    DeploymentConfig,
    SchedulerConfig,
    WorkerConfig,
    WorkerHostConfig,
)
from batchflow.deployment.launch_batchflow import (
    launch_colocated_batchflow,
    setup_logging,
    stop_all,
)
from batchflow.integrations.pytorch.iterable_dataset import BatchFlowIterableDataset


LOGGER = logging.getLogger("batchflow.dev.pytorch_smoke_test")


HOST = "127.0.0.1"
COORDINATOR_PORT = 50051
WORKER_START_PORT = 60061

NUM_WORKERS = 2
NUM_SAMPLES = 256
BATCH_SIZE = 32
MAX_batches = 5
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10

LOOKAHEAD_BATCHES = 8
TARGET_READY_BATCHES = 4

RESULTS_DIR = Path("results/local_demo")
LOG_LEVEL = "INFO"


class TinyInputModel(nn.Module):
    """Small CPU-friendly model used only by the smoke test."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
        super().__init__()

        input_dim = 1
        for dim in input_shape:
            input_dim *= dim

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_deployment_config() -> DeploymentConfig:
    return DeploymentConfig(
        coordinator=CoordinatorConfig(
            host=HOST,
            port=COORDINATOR_PORT,
            grpc_max_workers=8,
            default_lookahead=LOOKAHEAD_BATCHES,
            scheduler=SchedulerConfig(
                target_ready_batches=TARGET_READY_BATCHES,
            ),
        ),
        worker_hosts=[
            WorkerHostConfig(
                id="local",
                hostname=HOST,
                worker_count=NUM_WORKERS,
                worker_port_start=WORKER_START_PORT,
                worker_config=WorkerConfig(),
            )
        ],
    )


def make_dataset_config() -> DatasetConfig:
    return DatasetConfig(
        dataset_id="synthetic-torch",
        prefix_uri="memory://synthetic-torch",
        split="train",
        transform_name=None,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        seed=123,
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
    )


def run_training(coordinator_address: str) -> None:
    dataset = BatchFlowIterableDataset(
        coordinator_address=coordinator_address,
        dataset_id="synthetic-torch",
        job_id="pytorch-smoke-test",
        max_batches=MAX_batches,
        lookahead_batches=LOOKAHEAD_BATCHES,
        request_poll_interval_seconds=0.05,
        fetch_timeout_seconds=30.0,
        max_ready_batches=TARGET_READY_BATCHES,
        parallel_fetch_workers=1,
        ready_queue_timeout_seconds=0.5,
        pin_memory=False,
        finish_job_on_close=True,
        log_interval_seconds=2.0,
        log_every_n_batches=0,
        log_trainer_waits=False,
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    model = TinyInputModel(INPUT_SHAPE, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loader_iter = iter(loader)

    LOGGER.info("Starting PyTorch smoke test")

    try:
        for batch in range(MAX_batches):
            batch_start = time.perf_counter()

            data_start = time.perf_counter()
            batch = next(loader_iter)
            data_time_sec = time.perf_counter() - data_start

            inputs = batch["x"]
            labels = batch["label"]

            compute_start = time.perf_counter()

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.batch()

            compute_time_sec = time.perf_counter() - compute_start
            batch_time_sec = time.perf_counter() - batch_start

            data_bottleneck_percent = (
                100.0 * data_time_sec / max(batch_time_sec, 1e-12)
            )

            update_metrics = getattr(loader_iter, "update_runtime_metrics", None)

            if callable(update_metrics):
                update_metrics(
                    data_bottleneck_percent=data_bottleneck_percent,
                    avg_data_time_sec=data_time_sec,
                    avg_compute_time_sec=compute_time_sec,
                    avg_coordinator_wait_total_time_sec=float(
                        batch.get("coordinator_wait_total_time_sec", 0.0)
                    ),
                    avg_coordinator_pending_polls=float(
                        batch.get("pending_polls_before_batch", 0)
                    ),
                )

            LOGGER.info(
                "batch=%d/%d loss=%.4f data_time=%.4fs compute_time=%.4fs "
                "fetch_location=%s",
                batch + 1,
                MAX_batches,
                float(loss.item()),
                data_time_sec,
                compute_time_sec,
                batch.get("fetch_location", ""),
            )

        LOGGER.info("PyTorch smoke test completed successfully")

    finally:
        close_fn = getattr(loader_iter, "close", None)

        if callable(close_fn):
            close_fn()


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        base_dir=run_dir,
        level=LOG_LEVEL,
    )

    deployment_config = make_deployment_config()
    dataset_config = make_dataset_config()

    handles = []

    LOGGER.info("Starting local BatchFlow PyTorch smoke test")
    LOGGER.info("results_dir=%s", run_dir.resolve())

    try:
        handles = launch_colocated_batchflow(
            deployment_config=deployment_config,
            dataset_config=dataset_config,
            run_dir=run_dir,
            log_level=LOG_LEVEL,
        )

        time.sleep(1.0)

        run_training(
            deployment_config.coordinator.addr
        )

    finally:
        stop_all(handles)

    LOGGER.info("Smoke test finished")


if __name__ == "__main__":
    main()