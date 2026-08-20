from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_COLUMNS = [
    "workload_name",
    "system_name",
    "run_id",
    "run_timestamp",
    "num_jobs",
    "dataset_id",
    "batch_size",
    "aggregate_batches_per_sec",
    "aggregate_samples_per_sec",
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
    "run_dir",
]

GROUP_COLUMNS = [
    "workload_name",
    "system_name",
    "repetitions",
    "mean_aggregate_batches_per_sec",
    "std_aggregate_batches_per_sec",
    "mean_aggregate_samples_per_sec",
    "std_aggregate_samples_per_sec",
    "mean_cost_efficiency_batches_per_dollar",
    "std_cost_efficiency_batches_per_dollar",
    "mean_job_data_time_sec",
    "mean_job_compute_time_sec",
    "mean_job_batch_time_sec",
    "hourly_cost_usd",
    "pricing_name",
    "cost_resource_profile",
    "ablation_stage",
    "ablation_label",
    "ablation_order",
    "ablation_deployment",
    "ablation_policy",
]


def _read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_run_row(aggregate_path: Path) -> dict[str, Any]:
    aggregate = _read_single_csv_row(aggregate_path)
    return {
        **aggregate,
        "run_dir": str(aggregate_path.parent),
    }


def aggregate_run_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average repetitions while keeping ablation stages separate."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in run_rows:
        key = (
            str(row.get("workload_name", "")),
            str(row.get("system_name", "")),
            str(row.get("ablation_stage", "")),
        )
        groups[key].append(row)

    output: list[dict[str, Any]] = []

    for (workload_name, system_name, _stage), rows in sorted(groups.items()):
        throughputs = [_float(row, "aggregate_batches_per_sec") for row in rows]
        sample_rates = [_float(row, "aggregate_samples_per_sec") for row in rows]
        efficiencies = [
            _float(row, "cost_efficiency_batches_per_dollar") for row in rows
        ]
        data_times = [_float(row, "mean_job_data_time_sec") for row in rows]
        compute_times = [_float(row, "mean_job_compute_time_sec") for row in rows]
        batch_times = [_float(row, "mean_job_batch_time_sec") for row in rows]

        first = rows[0]
        hourly_costs = {_float(row, "hourly_cost_usd") for row in rows}
        if len(hourly_costs) > 1:
            raise ValueError(
                "hourly cost changed across repetitions for "
                f"{workload_name}/{system_name}/{first.get('ablation_stage', '')}: "
                f"{sorted(hourly_costs)}"
            )

        output.append(
            {
                "workload_name": workload_name,
                "system_name": system_name,
                "repetitions": len(rows),
                "mean_aggregate_batches_per_sec": _mean(throughputs),
                "std_aggregate_batches_per_sec": _sample_std(throughputs),
                "mean_aggregate_samples_per_sec": _mean(sample_rates),
                "std_aggregate_samples_per_sec": _sample_std(sample_rates),
                "mean_cost_efficiency_batches_per_dollar": _mean(efficiencies),
                "std_cost_efficiency_batches_per_dollar": _sample_std(efficiencies),
                "mean_job_data_time_sec": _mean(data_times),
                "mean_job_compute_time_sec": _mean(compute_times),
                "mean_job_batch_time_sec": _mean(batch_times),
                "hourly_cost_usd": next(iter(hourly_costs), 0.0),
                "pricing_name": first.get("pricing_name", ""),
                "cost_resource_profile": first.get("cost_resource_profile", ""),
                "ablation_stage": first.get("ablation_stage", ""),
                "ablation_label": first.get("ablation_label", ""),
                "ablation_order": first.get("ablation_order", ""),
                "ablation_deployment": first.get("ablation_deployment", ""),
                "ablation_policy": first.get("ablation_policy", ""),
            }
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect completed aggregate_summary.csv files and average "
            "repeated runs. Cost efficiency is already computed by the "
            "experiment reporter."
        )
    )
    parser.add_argument("--results-dir", default="exp_results")
    parser.add_argument("--output-dir", default="analysis_results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    aggregate_paths = sorted(results_dir.glob("*/*/run_*/aggregate_summary.csv"))
    if not aggregate_paths:
        raise FileNotFoundError(
            f"no completed aggregate_summary.csv files found under {results_dir}"
        )

    run_rows = [build_run_row(path) for path in aggregate_paths]
    group_rows = aggregate_run_rows(run_rows)

    run_output = output_dir / "run_summary.csv"
    group_output = output_dir / "group_summary.csv"

    _write_csv(run_output, run_rows, RUN_COLUMNS)
    _write_csv(group_output, group_rows, GROUP_COLUMNS)

    print(f"Wrote {run_output}")
    print(f"Wrote {group_output}")


if __name__ == "__main__":
    main()
