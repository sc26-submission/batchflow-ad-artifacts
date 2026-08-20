"""Run one or more PyTorch training jobs through BatchFlow using an S3 dataset.

The script launches a local BatchFlow coordinator and data workers, registers
an S3 image-folder dataset, runs independent PyTorch trainer processes, and
writes per-job timing metrics and a run summary.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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


LOGGER = logging.getLogger("batchflow.dev.pytorch_s3_training")


class SmoothedMeter:
    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.window_size:
            self.values.pop(0)

    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer: csv.DictWriter | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self.file is None:
            self.file = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()

        assert self.writer is not None
        self.writer.writerow(row)

    def flush(self) -> None:
        if self.file is not None:
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class TinyImageModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-job PyTorch training through BatchFlow over S3."
    )

    # BatchFlow runtime.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=50051)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-start-port", type=int, default=60061)

    # Dataset.
    parser.add_argument("--dataset-id", default="cifar10")
    parser.add_argument("--prefix-uri", default="s3://cifar10-bf")
    parser.add_argument("--split", default="train")
    parser.add_argument("--transform-name", default="cifar10_train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)

    # Training workload.
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--lookahead-batches", type=int, default=16)
    parser.add_argument("--max-ready-batches", type=int, default=4)
    parser.add_argument("--parallel-fetch-workers", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true", default=False)

    # Output.
    parser.add_argument("--results-dir", default="results/pytorch_s3_runs")
    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_job_logging(job_name: str, results_dir: Path, level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        results_dir / f"{job_name}.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def make_deployment_config(args: argparse.Namespace) -> DeploymentConfig:
    coordinator = CoordinatorConfig(
        host=args.host,
        port=args.coordinator_port,
        grpc_max_workers=max(8, args.num_jobs + args.num_workers + 2),
        default_lookahead=args.lookahead_batches,
        scheduler=SchedulerConfig(
            target_ready_batches=args.max_ready_batches,
        ),
    )

    worker_host = WorkerHostConfig(
        id="local",
        hostname=args.host,
        worker_count=args.num_workers,
        worker_port_start=args.worker_start_port,
        worker_config=WorkerConfig(),
    )

    return DeploymentConfig(
        coordinator=coordinator,
        worker_hosts=[worker_host],
    )


def make_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        dataset_id=args.dataset_id,
        prefix_uri=args.prefix_uri,
        split=args.split,
        transform_name=args.transform_name,
        num_samples=0,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        shuffle=args.shuffle,
        seed=123,
    )


def resolve_device(job_index: int, force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()

    if job_index >= device_count:
        raise RuntimeError(
            f"{job_index + 1} training jobs requested, but only "
            f"{device_count} CUDA device(s) are available."
        )

    return torch.device(f"cuda:{job_index}")


def make_batchflow_dataset(
    args: argparse.Namespace,
    coordinator_address: str,
    job_id: str,
    use_pin_memory: bool,
) -> BatchFlowIterableDataset:
    return BatchFlowIterableDataset(
        coordinator_address=coordinator_address,
        dataset_id=args.dataset_id,
        job_id=job_id,
        max_batches=args.warmup_batches + args.max_batches,
        lookahead_batches=args.lookahead_batches,
        request_poll_interval_seconds=0.05,
        fetch_timeout_seconds=30.0,
        max_ready_batches=args.max_ready_batches,
        parallel_fetch_workers=args.parallel_fetch_workers,
        ready_queue_timeout_seconds=0.5,
        pin_memory=use_pin_memory,
        finish_job_on_close=True,
        log_interval_seconds=2.0,
        log_every_n_batches=20,
        log_trainer_waits=False,
        log_pending_batch_every_n_polls=50,
    )


def batch_metric(batch: dict[str, Any], key: str, default: Any = 0.0) -> Any:
    return batch.get(key, default)


def run_training_job(
    job_index: int,
    args: argparse.Namespace,
    coordinator_address: str,
    run_id: str,
    results_dir_str: str,
) -> dict[str, Any]:
    results_dir = Path(results_dir_str)

    job_name = f"training-job-{job_index}"
    job_id = f"torch-job-{job_index}"

    setup_job_logging(job_name, results_dir, args.log_level)
    logger = logging.getLogger(f"batchflow.dev.pytorch_s3_training.{job_name}")

    device = resolve_device(job_index, args.force_cpu)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = make_batchflow_dataset(
        args=args,
        coordinator_address=coordinator_address,
        job_id=job_id,
        use_pin_memory=device.type == "cuda",
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    model = TinyImageModel(args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    data_meter = SmoothedMeter()
    compute_meter = SmoothedMeter()
    batch_meter = SmoothedMeter()
    throughput_meter = SmoothedMeter()

    metrics_path = results_dir / f"metrics_{job_name}.csv"
    csv_logger = CsvLogger(metrics_path)

    measured_batches = 0
    measured_samples = 0
    measured_time_sec = 0.0

    loader_iter = None
    start_time = time.perf_counter()

    logger.info(
        "Starting training job job=%s device=%s metrics=%s",
        job_name,
        device,
        metrics_path.resolve(),
    )

    try:
        model.train()
        loader_iter = iter(loader)

        total_batches = args.warmup_batches + args.max_batches

        for batch in range(total_batches):
            batch_start = time.perf_counter()

            data_start = time.perf_counter()
            batch = next(loader_iter)
            data_time_sec = time.perf_counter() - data_start

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            compute_start = time.perf_counter()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.batch()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            compute_time_sec = time.perf_counter() - compute_start
            batch_time_sec = time.perf_counter() - batch_start

            current_batch_size = int(images.shape[0])
            samples_per_sec = current_batch_size / max(batch_time_sec, 1e-12)
            is_warmup = batch < args.warmup_batches

            data_meter.update(data_time_sec)
            compute_meter.update(compute_time_sec)
            batch_meter.update(batch_time_sec)
            throughput_meter.update(samples_per_sec)

            data_bottleneck_percent = (
                100.0 * data_time_sec / max(batch_time_sec, 1e-12)
            )

            update_metrics = getattr(loader_iter, "update_runtime_metrics", None)

            if callable(update_metrics):
                update_metrics(
                    data_bottleneck_percent=data_bottleneck_percent,
                    avg_data_time_sec=data_meter.avg,
                    avg_compute_time_sec=compute_meter.avg,
                    avg_coordinator_wait_total_time_sec=float(
                        batch_metric(batch, "coordinator_wait_total_time_sec")
                    ),
                    avg_coordinator_pending_polls=float(
                        batch_metric(batch, "pending_polls_before_batch", 0)
                    ),
                )

            if not is_warmup:
                measured_batches += 1
                measured_samples += current_batch_size
                measured_time_sec += batch_time_sec

            logger.info(
                "batch=%d/%d warmup=%s loss=%.4f data=%.4fs compute=%.4fs "
                "batch_time=%.4fs throughput=%.2f samples/s cache=%s location=%s",
                batch + 1,
                total_batches,
                is_warmup,
                float(loss.item()),
                data_time_sec,
                compute_time_sec,
                batch_time_sec,
                samples_per_sec,
                batch_metric(batch, "client_cache_result", ""),
                batch_metric(batch, "fetch_location", ""),
            )

            csv_logger.log(
                {
                    "timestamp_unix": time.time(),
                    "run_id": run_id,
                    "job_index": job_index,
                    "job_name": job_name,
                    "job_id": job_id,
                    "device": str(device),
                    "batch": batch,
                    "warmup": int(is_warmup),
                    "batchflow_batch_id": batch_metric(batch, "batch_id", ""),
                    "batch_index": batch_metric(batch, "batch_index", -1),
                    "epoch": batch_metric(batch, "epoch", -1),
                    "batch_size": current_batch_size,
                    "loss": float(loss.item()),
                    "data_time_sec": data_time_sec,
                    "compute_time_sec": compute_time_sec,
                    "batch_time_sec": batch_time_sec,
                    "samples_per_sec": samples_per_sec,
                    "avg_data_time_sec": data_meter.avg,
                    "avg_compute_time_sec": compute_meter.avg,
                    "avg_batch_time_sec": batch_meter.avg,
                    "avg_samples_per_sec": throughput_meter.avg,
                    "data_bottleneck_percent": data_bottleneck_percent,
                    "handle_status": batch_metric(batch, "handle_status", ""),
                    "cache_result": batch_metric(batch, "cache_result", ""),
                    "client_cache_result": batch_metric(
                        batch, "client_cache_result", ""
                    ),
                    "fetch_location": batch_metric(batch, "fetch_location", ""),
                    "pending_polls_before_batch": batch_metric(
                        batch, "pending_polls_before_batch", 0
                    ),
                    "coordinator_wait_total_time_sec": batch_metric(
                        batch, "coordinator_wait_total_time_sec"
                    ),
                    "coordinator_rpc_time_sec": batch_metric(
                        batch, "coordinator_rpc_time_sec"
                    ),
                    "coordinator_sleep_time_sec": batch_metric(
                        batch, "coordinator_sleep_time_sec"
                    ),
                    "fetch_time_sec": batch_metric(batch, "fetch_time_sec"),
                    "trainer_decode_time_sec": batch_metric(
                        batch, "trainer_decode_time_sec"
                    ),
                    "trainer_pin_time_sec": batch_metric(
                        batch, "trainer_pin_time_sec"
                    ),
                    "payload_bytes": batch_metric(batch, "payload_bytes", 0),
                }
            )

            if batch > 0 and batch % 10 == 0:
                csv_logger.flush()

        measured_throughput = measured_samples / max(measured_time_sec, 1e-12)

        logger.info(
            "Training job finished job=%s measured_batches=%d "
            "throughput=%.2f samples/s",
            job_name,
            measured_batches,
            measured_throughput,
        )

        return {
            "timestamp_unix": time.time(),
            "run_id": run_id,
            "job_index": job_index,
            "job_name": job_name,
            "job_id": job_id,
            "device": str(device),
            "status": "ok",
            "duration_sec": time.perf_counter() - start_time,
            "measured_batches": measured_batches,
            "measured_samples": measured_samples,
            "measured_time_sec": measured_time_sec,
            "measured_samples_per_sec": measured_throughput,
            "error": "",
        }

    except Exception as exc:
        logger.exception("Training job failed job=%s", job_name)

        return {
            "timestamp_unix": time.time(),
            "run_id": run_id,
            "job_index": job_index,
            "job_name": job_name,
            "job_id": job_id,
            "device": str(device),
            "status": "failed",
            "duration_sec": time.perf_counter() - start_time,
            "measured_batches": measured_batches,
            "measured_samples": measured_samples,
            "measured_time_sec": measured_time_sec,
            "measured_samples_per_sec": 0.0,
            "error": repr(exc),
        }

    finally:
        if loader_iter is not None:
            close_fn = getattr(loader_iter, "close", None)
            if callable(close_fn):
                close_fn()

        csv_logger.flush()
        csv_logger.close()


def run_training_jobs(
    args: argparse.Namespace,
    coordinator_address: str,
    run_id: str,
    results_dir: Path,
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    summaries: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=args.num_jobs,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(
                run_training_job,
                job_index,
                args,
                coordinator_address,
                run_id,
                str(results_dir),
            )
            for job_index in range(args.num_jobs)
        ]

        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)

            LOGGER.info(
                "Training process completed job=%s status=%s",
                summary["job_name"],
                summary["status"],
            )

    return summaries


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")

    results_dir = Path(args.results_dir) / f"pytorch_s3_training_{run_id}"
    runtime_dir = results_dir / "runtime"

    results_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(base_dir=runtime_dir, level=args.log_level)

    LOGGER.info("results_dir=%s", results_dir.resolve())
    LOGGER.info("jobs=%d workers=%d", args.num_jobs, args.num_workers)

    deployment = make_deployment_config(args)
    dataset = make_dataset_config(args)

    handles = []

    try:
        handles = launch_colocated_batchflow(
            deployment_config=deployment,
            dataset_config=dataset,
            run_dir=runtime_dir,
            log_level=args.log_level,
        )

        time.sleep(1.0)

        summaries = run_training_jobs(
            args=args,
            coordinator_address=deployment.coordinator.addr,
            run_id=run_id,
            results_dir=results_dir,
        )

        summaries.sort(key=lambda row: row["job_index"])

        summary_path = results_dir / "summary.csv"
        write_csv(summary_path, summaries)

        failed = [row for row in summaries if row["status"] != "ok"]

        LOGGER.info(
            "PyTorch S3 run finished jobs=%d failed=%d",
            len(summaries),
            len(failed),
        )

        if failed:
            raise RuntimeError(f"{len(failed)} training job(s) failed")

    finally:
        stop_all(handles)
        LOGGER.info("Done: %s", results_dir.resolve())


if __name__ == "__main__":
    mp.freeze_support()
    main()