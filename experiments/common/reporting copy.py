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
    "data_time_sec",
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
    "trainer_decode_time_sec",
    "trainer_pin_time_sec",
    "prefetch_queue_size_before_put",
    "trainer_queue_size_after_get",
    "trainer_queue_empty_events",
    "tensorsocket_wait_time_sec",
    "tensorsocket_cache_hit",
    "coordl_wait_time_sec",
    "coordl_deserialize_time_sec",
    "coordl_io_time_sec",
    "coordl_decode_time_sec",
    "coordl_transform_time_sec",
    "coordl_prep_time_sec",
    "coordl_payload_bytes",
    "coordl_is_owner",
    "batch_index",
    "batch_id",
    "job_id",
    "elapsed_time_sec",
    "device",
    "use_amp",
]


JOB_SUMMARY_COLUMNS = [
    "system_name",
    "workload_name",
    "job_name",
    "model_name",
    "run_id",
    "run_timestamp",
    "num_jobs",
    "dataset_id",
    "dataset_uri",
    "dataset_split",
    "transform_name",
    "batch_size",
    "mode",
    "batches_completed",
    "measured_batches",
    "warmup_batches",
    "total_time_sec",
    "samples_per_sec",
    "batches_per_sec",
    "avg_batch_time_sec",
    "avg_data_time_sec",
    "avg_compute_time_sec",
    "data_to_compute_ratio",
    "data_bottleneck_percent",
    "avg_h2d_time_sec",
    "avg_forward_time_sec",
    "avg_backward_time_sec",
    "avg_optimizer_step_time_sec",
    "avg_baseline_io_time_sec",
    "avg_baseline_decode_time_sec",
    "avg_baseline_transform_time_sec",
    "avg_batchflow_worker_io_time_sec",
    "avg_batchflow_worker_decode_time_sec",
    "avg_batchflow_worker_transform_time_sec",
    "avg_batchflow_worker_stack_time_sec",
    "avg_batchflow_worker_serialize_time_sec",
    "avg_batchflow_fetch_time_sec",
    "avg_batchflow_coordinator_rpc_time_sec",
    "avg_batchflow_coordinator_sleep_time_sec",
    "avg_batchflow_coordinator_wait_total_time_sec",
    "avg_batchflow_coordinator_pending_polls",
    "avg_trainer_decode_time_sec",
    "avg_trainer_pin_time_sec",
    "avg_tensorsocket_wait_time_sec",
    "tensorsocket_cache_hit_rate",
    "avg_coordl_wait_time_sec",
    "avg_coordl_deserialize_time_sec",
    "avg_coordl_io_time_sec",
    "avg_coordl_decode_time_sec",
    "avg_coordl_transform_time_sec",
    "avg_coordl_prep_time_sec",
    "avg_coordl_payload_bytes",
    "coordl_owner_batch_fraction",
]


AGGREGATE_COLUMNS = [
    "system_name",
    "workload_name",
    "run_id",
    "run_timestamp",
    "num_jobs",
    "dataset_id",
    "batch_size",
    "aggregate_samples_per_sec",
    "aggregate_batches_per_sec",
    "mean_job_samples_per_sec",
    "mean_job_batches_per_sec",
    "mean_job_data_time_sec",
    "mean_job_compute_time_sec",
    "mean_job_batch_time_sec",
    "pricing_name",
    "cost_resource_profile",
    "hourly_cost_usd",
    "cost_efficiency_batches_per_dollar",
    "ablation_stage",
    "ablation_label",
    "ablation_order",
    "ablation_deployment",
    "ablation_policy",
]


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return safe.strip("_") or "job"


def per_batch_metrics_path_for_job(output_dir: Path, job_name: str) -> Path:
    return output_dir / f"per_batch_metrics_{_safe_name(job_name)}.csv"


def _ordered_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)

    return [key for key in preferred if key in seen] + [
        key for key in seen if key not in preferred
    ]


def write_rows_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
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
            fields = _ordered_fields([row], PER_batch_COLUMNS)
            self._writer = csv.DictWriter(self._file, fieldnames=fields)
            self._writer.writeheader()

        assert self._writer is not None
        self._writer.writerow(row)
        self._rows_since_flush += 1

        if self._rows_since_flush >= self.flush_every_batches:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_float(row, key) for row in rows) / len(rows)


def build_job_summary(
    path: Path,
    *,
    mode: str,
    warmup_batches: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rows = read_rows_csv(path)
    measured = [row for row in rows if int(float(row.get("warmup", 0) or 0)) == 0]

    total_time_sec = sum(_float(row, "batch_time_sec") for row in measured)
    total_samples = sum(_float(row, "batch_size") for row in measured)
    avg_data = _average(measured, "data_time_sec")
    avg_compute = _average(measured, "compute_time_sec")

    return {
        **metadata,
        "mode": mode,
        "batches_completed": len(rows),
        "measured_batches": len(measured),
        "warmup_batches": warmup_batches,
        "total_time_sec": total_time_sec,
        "samples_per_sec": total_samples / max(total_time_sec, 1e-12),
        "batches_per_sec": len(measured) / max(total_time_sec, 1e-12),
        "avg_batch_time_sec": _average(measured, "batch_time_sec"),
        "avg_data_time_sec": avg_data,
        "avg_compute_time_sec": avg_compute,
        "data_to_compute_ratio": avg_data / max(avg_compute, 1e-12),
        "data_bottleneck_percent": avg_data / max(avg_data + avg_compute, 1e-12),
        "avg_h2d_time_sec": _average(measured, "h2d_time_sec"),
        "avg_forward_time_sec": _average(measured, "forward_time_sec"),
        "avg_backward_time_sec": _average(measured, "backward_time_sec"),
        "avg_optimizer_step_time_sec": _average(measured, "optimizer_step_time_sec"),
        "avg_baseline_io_time_sec": _average(measured, "baseline_io_time_sec"),
        "avg_baseline_decode_time_sec": _average(measured, "baseline_decode_time_sec"),
        "avg_baseline_transform_time_sec": _average(measured, "baseline_transform_time_sec"),
        "avg_batchflow_worker_io_time_sec": _average(measured, "batchflow_worker_io_time_sec"),
        "avg_batchflow_worker_decode_time_sec": _average(measured, "batchflow_worker_decode_time_sec"),
        "avg_batchflow_worker_transform_time_sec": _average(measured, "batchflow_worker_transform_time_sec"),
        "avg_batchflow_worker_stack_time_sec": _average(measured, "batchflow_worker_stack_time_sec"),
        "avg_batchflow_worker_serialize_time_sec": _average(measured, "batchflow_worker_serialize_time_sec"),
        "avg_batchflow_fetch_time_sec": _average(measured, "batchflow_fetch_time_sec"),
        "avg_batchflow_coordinator_rpc_time_sec": _average(measured, "batchflow_coordinator_rpc_time_sec"),
        "avg_batchflow_coordinator_sleep_time_sec": _average(measured, "batchflow_coordinator_sleep_time_sec"),
        "avg_batchflow_coordinator_wait_total_time_sec": _average(measured, "batchflow_coordinator_wait_total_time_sec"),
        "avg_batchflow_coordinator_pending_polls": _average(measured, "batchflow_coordinator_pending_polls"),
        "avg_trainer_decode_time_sec": _average(measured, "trainer_decode_time_sec"),
        "avg_trainer_pin_time_sec": _average(measured, "trainer_pin_time_sec"),
        "avg_tensorsocket_wait_time_sec": _average(measured, "tensorsocket_wait_time_sec"),
        "tensorsocket_cache_hit_rate": _average(measured, "tensorsocket_cache_hit"),
        "avg_coordl_wait_time_sec": _average(measured, "coordl_wait_time_sec"),
        "avg_coordl_deserialize_time_sec": _average(measured, "coordl_deserialize_time_sec"),
        "avg_coordl_io_time_sec": _average(measured, "coordl_io_time_sec"),
        "avg_coordl_decode_time_sec": _average(measured, "coordl_decode_time_sec"),
        "avg_coordl_transform_time_sec": _average(measured, "coordl_transform_time_sec"),
        "avg_coordl_prep_time_sec": _average(measured, "coordl_prep_time_sec"),
        "avg_coordl_payload_bytes": _average(measured, "coordl_payload_bytes"),
        "coordl_owner_batch_fraction": _average(measured, "coordl_is_owner"),
    }


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return dict(value)


def _load_cost_metadata(cfg, aggregate_batches_per_sec: float) -> dict[str, Any]:
    """Resolve fixed artifact pricing and compute run-level cost efficiency.

    Cost is intentionally reported only at the aggregate/system level because
    the infrastructure is shared by all concurrent jobs.
    """
    system_cfg = cfg.get("system")
    cost_cfg = _plain_dict(system_cfg.get("cost") if system_cfg else None)

    pricing_name = str(cost_cfg.get("pricing", "")).strip()
    resource_profile = str(cost_cfg.get("resource_profile", "")).strip()

    ablation_cfg = _plain_dict(cfg.get("ablation"))
    ablation_enabled = bool(ablation_cfg.get("enabled", False))
    if ablation_enabled:
        override = str(ablation_cfg.get("cost_resource_profile", "")).strip()
        if override:
            resource_profile = override

    hourly_cost_usd = 0.0
    batches_per_dollar = 0.0

    if pricing_name or resource_profile:
        if not pricing_name or not resource_profile:
            raise ValueError(
                "cost reporting requires both system.cost.pricing and "
                "system.cost.resource_profile"
            )

        pricing_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "pricing"
            / f"{pricing_name}.yaml"
        )
        if not pricing_path.exists():
            raise FileNotFoundError(
                f"pricing config {pricing_name!r} not found at {pricing_path}"
            )

        pricing = OmegaConf.load(pricing_path)
        currency = str(pricing.get("currency", "USD")).upper()
        if currency != "USD":
            raise ValueError(
                f"pricing config {pricing_path} uses currency={currency!r}; "
                "aggregate_summary.csv currently reports USD costs"
            )

        hourly_prices = {
            str(name): float(value)
            for name, value in _plain_dict(pricing.get("hourly_usd")).items()
        }
        profiles = _plain_dict(pricing.get("resource_profiles"))
        if resource_profile not in profiles:
            available = ", ".join(sorted(profiles)) or "<none>"
            raise KeyError(
                f"unknown cost resource profile {resource_profile!r} in "
                f"{pricing_path}. Available profiles: {available}"
            )

        resources = _plain_dict(profiles[resource_profile])
        for resource, raw_count in resources.items():
            count = float(raw_count)
            if count < 0:
                raise ValueError(
                    f"resource count must be >= 0 for {resource!r}, got {count}"
                )
            if count == 0:
                continue
            if resource not in hourly_prices:
                raise KeyError(
                    f"resource {resource!r} has no hourly price in {pricing_path}"
                )
            hourly_cost_usd += count * hourly_prices[resource]

        if hourly_cost_usd <= 0:
            raise ValueError(
                f"cost profile {resource_profile!r} has zero hourly cost"
            )

        batches_per_dollar = (
            float(aggregate_batches_per_sec) * 3600.0 / hourly_cost_usd
        )

    return {
        "pricing_name": pricing_name,
        "cost_resource_profile": resource_profile,
        "hourly_cost_usd": hourly_cost_usd,
        "cost_efficiency_batches_per_dollar": batches_per_dollar,
        "ablation_stage": (
            str(ablation_cfg.get("stage", "")) if ablation_enabled else ""
        ),
        "ablation_label": (
            str(ablation_cfg.get("label", "")) if ablation_enabled else ""
        ),
        "ablation_order": (
            ablation_cfg.get("order", "") if ablation_enabled else ""
        ),
        "ablation_deployment": (
            str(ablation_cfg.get("deployment", "")) if ablation_enabled else ""
        ),
        "ablation_policy": (
            str(ablation_cfg.get("policy", "")) if ablation_enabled else ""
        ),
    }


class ExperimentReporter:
    def __init__(
        self,
        *,
        results_dir: Path,
        workload_name: str,
        system_name: str,
        dataset_config,
        cfg,
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

    def _job_metadata(self, job, num_jobs: int) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "workload_name": self.workload_name,
            "job_name": job.name,
            "model_name": job.model_name,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "num_jobs": num_jobs,
            "dataset_id": self.dataset.dataset_id,
            "dataset_uri": self.dataset.prefix_uri,
            "dataset_split": self.dataset.split,
            "transform_name": self.dataset.transform_name or "",
            "batch_size": self.dataset.batch_size,
        }

    def write_summaries(self, *, jobs, mode: str) -> None:
        num_jobs = len(jobs)
        job_rows: list[dict[str, Any]] = []

        for job in jobs:
            path = per_batch_metrics_path_for_job(self.run_dir, job.name)
            job_rows.append(
                build_job_summary(
                    path,
                    mode=mode,
                    warmup_batches=job.warmup_batches,
                    metadata=self._job_metadata(job, num_jobs),
                )
            )

        write_rows_csv(self.run_dir / "job_summary.csv", job_rows, JOB_SUMMARY_COLUMNS)

        aggregate_samples = sum(float(row["samples_per_sec"]) for row in job_rows)
        aggregate_batches = sum(float(row["batches_per_sec"]) for row in job_rows)

        aggregate = {
            "system_name": self.system_name,
            "workload_name": self.workload_name,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "num_jobs": num_jobs,
            "dataset_id": self.dataset.dataset_id,
            "batch_size": self.dataset.batch_size,
            "aggregate_samples_per_sec": aggregate_samples,
            "aggregate_batches_per_sec": aggregate_batches,
            "mean_job_samples_per_sec": aggregate_samples / num_jobs,
            "mean_job_batches_per_sec": aggregate_batches / num_jobs,
            "mean_job_data_time_sec": sum(
                float(row["avg_data_time_sec"]) for row in job_rows
            ) / num_jobs,
            "mean_job_compute_time_sec": sum(
                float(row["avg_compute_time_sec"]) for row in job_rows
            ) / num_jobs,
            "mean_job_batch_time_sec": sum(
                float(row["avg_batch_time_sec"]) for row in job_rows
            ) / num_jobs,
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
