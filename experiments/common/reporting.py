from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PER_batch_COLUMNS = [
    "batch", 
    "warmup", 
    "batch_size", 
    "loss",
    "batch_time_sec", 
    "load_batch_time_sec", 
    "compute_time_sec",
    "h2d_time_sec", 
    "forward_time_sec", 
    "backward_time_sec",
    "optimizer_step_time_sec", 
    "samples_per_sec",
    "rolling_samples_per_sec", 
    "avg_batch_time_sec",
    "avg_data_time_sec", 
    "avg_compute_time_sec",
    "data_bottleneck_percent", 
    "avg_data_bottleneck_percent",
    "baseline_io_time_sec", 
    "baseline_decode_time_sec",
    "baseline_transform_time_sec",
    "batchflow_worker_io_time_sec",
    "batchflow_worker_decode_time_sec",
    "batchflow_worker_transform_time_sec",
    "batchflow_worker_stack_time_sec",
    "batchflow_worker_serialize_time_sec",
    "batchflow_fetch_time_sec",
    "batchflow_coordinator_rpc_time_sec",
    "batchflow_coordinator_sleep_time_sec",
    "batchflow_coordinator_wait_total_time_sec",
    "batchflow_coordinator_pending_polls",
    "trainer_decode_time_sec", "trainer_pin_time_sec",
    "prefetch_queue_size_before_put",
    "trainer_queue_size_after_get",
    "trainer_queue_empty_events",
    "tensorsocket_wait_time_sec", "tensorsocket_cache_hit",
    "coordl_wait_time_sec", "coordl_deserialize_time_sec",
    "coordl_io_time_sec", "coordl_decode_time_sec",
    "coordl_transform_time_sec", "coordl_prep_time_sec",
    "coordl_payload_bytes", "coordl_is_owner",
    "batch_index", "batch_id", "job_id",
    "elapsed_time_sec", "device", "use_amp",
]


JOB_AVERAGE_METRICS = {
    "avg_h2d_time_sec": "h2d_time_sec",
    "avg_forward_time_sec": "forward_time_sec",
    "avg_backward_time_sec": "backward_time_sec",
    "avg_optimizer_step_time_sec": "optimizer_step_time_sec",
    "avg_baseline_io_time_sec": "baseline_io_time_sec",
    "avg_baseline_decode_time_sec": "baseline_decode_time_sec",
    "avg_baseline_transform_time_sec": "baseline_transform_time_sec",
    "avg_batchflow_worker_io_time_sec": "batchflow_worker_io_time_sec",
    "avg_batchflow_worker_decode_time_sec": "batchflow_worker_decode_time_sec",
    "avg_batchflow_worker_transform_time_sec": "batchflow_worker_transform_time_sec",
    "avg_batchflow_worker_stack_time_sec": "batchflow_worker_stack_time_sec",
    "avg_batchflow_worker_serialize_time_sec": "batchflow_worker_serialize_time_sec",
    "avg_batchflow_fetch_time_sec": "batchflow_fetch_time_sec",
    "avg_batchflow_coordinator_rpc_time_sec": "batchflow_coordinator_rpc_time_sec",
    "avg_batchflow_coordinator_sleep_time_sec": "batchflow_coordinator_sleep_time_sec",
    "avg_batchflow_coordinator_wait_total_time_sec": "batchflow_coordinator_wait_total_time_sec",
    "avg_batchflow_coordinator_pending_polls": "batchflow_coordinator_pending_polls",
    "avg_trainer_decode_time_sec": "trainer_decode_time_sec",
    "avg_trainer_pin_time_sec": "trainer_pin_time_sec",
    "avg_tensorsocket_wait_time_sec": "tensorsocket_wait_time_sec",
    "tensorsocket_cache_hit_rate": "tensorsocket_cache_hit",
    "avg_coordl_wait_time_sec": "coordl_wait_time_sec",
    "avg_coordl_deserialize_time_sec": "coordl_deserialize_time_sec",
    "avg_coordl_io_time_sec": "coordl_io_time_sec",
    "avg_coordl_decode_time_sec": "coordl_decode_time_sec",
    "avg_coordl_transform_time_sec": "coordl_transform_time_sec",
    "avg_coordl_prep_time_sec": "coordl_prep_time_sec",
    "avg_coordl_payload_bytes": "coordl_payload_bytes",
    "coordl_owner_batch_fraction": "coordl_is_owner",
}


JOB_SUMMARY_COLUMNS = [
    "system_name", "workload_name", "job_name", "model_name",
    "run_id", "run_timestamp", "num_jobs",
    "dataset_id", "dataset_uri", "dataset_split", "transform_name",
    "batch_size", "mode",
    "total_batches", "completed_batches", "warmup_batches",
    "total_time_sec", "samples_per_sec", "batches_per_sec",
    "avg_batch_time_sec", "avg_data_time_sec", "avg_compute_time_sec",
    "data_to_compute_ratio", "data_bottleneck_percent",
    *JOB_AVERAGE_METRICS.keys(),
]


AGGREGATE_MEAN_METRICS = {
    "avg_job_samples_per_sec": "samples_per_sec",
    "avg_job_batches_per_sec": "batches_per_sec",
    "avg_job_data_time_sec": "avg_load_batch_time_sec",
    "avg_job_compute_time_sec": "avg_model_compute_time_sec",
    "avg_job_batch_time_sec": "avg_batch_time_sec",
}


AGGREGATE_COLUMNS = [
    "system_name", "workload_name", "run_id", "run_timestamp",
    "num_jobs", "dataset_id", "batch_size",
    "aggregate_samples_per_sec", "aggregate_batches_per_sec",
    *AGGREGATE_MEAN_METRICS.keys(),
    "pricing_name", "cost_resource_profile", "hourly_cost_usd",
    "cost_efficiency_batches_per_dollar",
    "ablation_stage", "ablation_label", "ablation_order",
    "ablation_deployment", "ablation_policy",
]


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return safe.strip("_") or "job"


def per_batch_metrics_path_for_job(output_dir: Path, job_name: str) -> Path:
    return output_dir / f"per_batch_metrics_{_safe_name(job_name)}.csv"


def _ordered_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen = list(dict.fromkeys(key for row in rows for key in row))
    return [key for key in preferred if key in seen] + [
        key for key in seen if key not in preferred
    ]


def write_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    preferred: list[str],
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _ordered_fields(rows, preferred)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class PerbatchMetricsWriter:
    """Streaming CSV writer used by one training process."""

    def __init__(self, path: Path, flush_every_batches: int = 10) -> None:
        self.path = path
        self.flush_every_batches = max(1, flush_every_batches)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._file = None
        self._writer: csv.DictWriter | None = None
        self._rows_since_flush = 0

    def write_batch(self, row: dict[str, Any]) -> None:
        if self._file is None:
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                # fieldnames=PER_batch_COLUMNS,
                fieldnames=row.keys(),
                extrasaction="ignore",
            )
            self._writer.writeheader()

        assert self._writer is not None
        self._writer.writerow(row)
        self._rows_since_flush += 1

        if self._rows_since_flush >= self.flush_every_batches:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._file is None:
            return

        self._file.flush()
        self._file.close()
        self._file = None
        self._writer = None
        self._rows_since_flush = 0


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_as_float(row, key) for row in rows) / len(rows)


def build_job_summary(
    path: Path,
    *,
    mode: str,
    warmup_batches: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rows = read_rows_csv(path)
    completed = [row for row in rows if not bool(int(_as_float(row, "warmup")))]

    total_time = sum(_as_float(row, "total_batch_time_sec") for row in completed)
    total_samples = sum(_as_float(row, "batch_size") for row in completed)
    total_batches = len(completed)
    avg_data = _average(completed, "total_load_batch_time_sec")
    avg_compute = _average(completed, "total_model_compute_time_sec")
    avg_batch = _average(completed, "total_batch_time_sec")

    averages = {
        output_name: _average(completed, source_name)
        for output_name, source_name in JOB_AVERAGE_METRICS.items()
    }

    return {
        **metadata,
        "total_batches": len(rows),
        "warmup_batches": warmup_batches,
        "measured_batches": total_batches,
        "total_time_sec": total_time,
        "samples_per_sec": total_samples / max(total_time, 1e-12),
        "batches_per_sec": total_batches / max(total_time, 1e-12),
        "avg_batch_time_sec": avg_batch,
        "avg_load_batch_time_sec": avg_data,
        "avg_model_compute_time_sec": avg_compute,
        # "data_to_compute_ratio": avg_data / max(avg_compute, 1e-12),
        # "data_bottleneck_percent": (
        #     100.0 * avg_data / max(avg_data + avg_compute, 1e-12)
        # ),
        # **averages,
    }


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping, got {type(value).__name__}")

    return value


def _load_cost_metadata(
    cfg: Any,
    aggregate_batches_per_sec: float,
) -> dict[str, Any]:
    system_cfg = cfg.get("system")
    cost_cfg = _plain_dict(system_cfg.get("cost") if system_cfg else None)

    pricing_name = str(cost_cfg.get("pricing", "")).strip()
    resource_profile = str(cost_cfg.get("resource_profile", "")).strip()

    ablation_cfg = _plain_dict(cfg.get("ablation"))
    ablation_enabled = bool(ablation_cfg.get("enabled", False))

    if ablation_enabled:
        resource_profile = (
            str(ablation_cfg.get("cost_resource_profile", "")).strip()
            or resource_profile
        )

    hourly_cost_usd = 0.0

    if pricing_name or resource_profile:
        if not pricing_name or not resource_profile:
            raise ValueError(
                "Cost reporting requires both system.cost.pricing "
                "and system.cost.resource_profile"
            )

        pricing_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "pricing"
            / f"{pricing_name}.yaml"
        )

        if not pricing_path.exists():
            raise FileNotFoundError(
                f"Pricing config {pricing_name!r} not found at {pricing_path}"
            )

        pricing = OmegaConf.load(pricing_path)
        currency = str(pricing.get("currency", "USD")).upper()

        if currency != "USD":
            raise ValueError(
                f"Pricing config {pricing_path} uses currency={currency!r}; "
                "expected USD"
            )

        hourly_prices = {
            str(name): float(value)
            for name, value in _plain_dict(pricing.get("hourly_usd")).items()
        }
        profiles = _plain_dict(pricing.get("resource_profiles"))

        if resource_profile not in profiles:
            available = ", ".join(sorted(profiles)) or "<none>"
            raise KeyError(
                f"Unknown cost resource profile {resource_profile!r}. "
                f"Available profiles: {available}"
            )

        for resource, raw_count in _plain_dict(profiles[resource_profile]).items():
            count = float(raw_count)

            if count < 0:
                raise ValueError(
                    f"Resource count for {resource!r} must be >= 0, got {count}"
                )
            if count == 0:
                continue
            if resource not in hourly_prices:
                raise KeyError(
                    f"Resource {resource!r} has no hourly price in {pricing_path}"
                )

            hourly_cost_usd += count * hourly_prices[resource]

        if hourly_cost_usd <= 0:
            raise ValueError(
                f"Cost profile {resource_profile!r} has zero hourly cost"
            )

    batches_per_dollar = (
        aggregate_batches_per_sec * 3600.0 / hourly_cost_usd
        if hourly_cost_usd > 0
        else 0.0
    )

    return {
        # "pricing_name": pricing_name,
        # "cost_resource_profile": resource_profile,
        "hourly_cost_usd": hourly_cost_usd,
        "cost_efficiency_batches_per_dollar": batches_per_dollar,
        # "ablation_stage": (
        #     str(ablation_cfg.get("stage", "")) if ablation_enabled else ""
        # ),
        # "ablation_label": (
        #     str(ablation_cfg.get("label", "")) if ablation_enabled else ""
        # ),
        # "ablation_order": (
        #     ablation_cfg.get("order", "") if ablation_enabled else ""
        # ),
        # "ablation_deployment": (
        #     str(ablation_cfg.get("deployment", "")) if ablation_enabled else ""
        # ),
        # "ablation_policy": (
        #     str(ablation_cfg.get("policy", "")) if ablation_enabled else ""
        # ),
    }


class ExperimentReporter:
    def __init__(
        self,
        *,
        results_dir: Path,
        workload_name: str,
        system_name: str,
        dataset_config: Any,
        cfg: Any,
        run_id: str,
        run_timestamp: str,
    ) -> None:
        self.workload_name = workload_name
        self.system_name = system_name
        self.dataset = dataset_config
        self.cfg = cfg
        self.run_id = run_id
        self.run_timestamp = run_timestamp
        self.run_dir = results_dir / workload_name / system_name / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_resolved_config(self) -> None:
        OmegaConf.save(
            config=self.cfg,
            f=str(self.run_dir / "config.yaml"),
            resolve=True,
        )

    def _job_metadata(self, job: Any, num_jobs: int) -> dict[str, Any]:
        return {
            "workload_name": self.workload_name,
            "job_name": job.name,
            "model_name": job.model_name,
            # "run_id": self.run_id,
            # "run_timestamp": self.run_timestamp,
            # "num_jobs": num_jobs,
            # "dataset_id": self.dataset.dataset_id,
            # "dataset_uri": self.dataset.prefix_uri,
            # "dataset_split": self.dataset.split,
            "transform_name": self.dataset.transform_name or "",
            "batch_size": self.dataset.batch_size,
        }

    def write_summaries(self, *, jobs: Any, mode: str) -> None:
        jobs = list(jobs)
        if not jobs:
            raise ValueError("Cannot write summaries without any jobs")

        num_jobs = len(jobs)

        job_rows = [
            build_job_summary(
                per_batch_metrics_path_for_job(self.run_dir, job.name),
                mode=mode,
                warmup_batches=job.warmup_batches,
                metadata=self._job_metadata(job, num_jobs),
            )
            for job in jobs
        ]

        write_rows_csv(
            self.run_dir / "per_job_summary.csv",
            job_rows,
            job_rows[0].keys() #if job_rows else JOB_SUMMARY_COLUMNS,
            # JOB_SUMMARY_COLUMNS,
        )

        aggregate_samples = sum(_as_float(row, "samples_per_sec") for row in job_rows)
        aggregate_batches = sum(_as_float(row, "batches_per_sec") for row in job_rows)

        aggregate = {
            "system_name": self.system_name,
            "workload_name": self.workload_name,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "num_jobs": num_jobs,
            "dataset_id": self.dataset.dataset_id,
            "batch_size": self.dataset.batch_size,
            "total_batches": sum(_as_float(row, "total_batches") for row in job_rows),
            "aggregate_samples_per_sec": aggregate_samples,
            "aggregate_batches_per_sec": aggregate_batches,
            # **{
            #     output_name: _average(job_rows, source_name)
            #     for output_name, source_name in AGGREGATE_MEAN_METRICS.items()
            # },
        }

        aggregate.update(
            _load_cost_metadata(
                self.cfg,
                aggregate_batches_per_sec=aggregate_batches,
            )
        )

        write_rows_csv(
            self.run_dir / "aggregate_summary.csv",
            [aggregate],
            AGGREGATE_COLUMNS,
        )