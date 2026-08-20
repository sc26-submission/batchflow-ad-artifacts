from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from batchflow.common.utils import CsvLogger


def _load_existing_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]
class ExperimentCsvWriters:
    """
    Minimal experiment CSV writer.

    Writes:
    - batch_rows.csv                  (overwritten for the current run)
    - all_summary_rows.csv           (per-job summaries, appended across runs)
    - all_aggregate_summary_rows.csv (aggregate summaries, appended across runs)

    Notes:
    - batch_rows.csv contains only rows from the current invocation
    - summary CSVs preserve prior rows if requested
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        append_existing_all_summary_rows: bool = True,
        append_existing_all_aggregate_summary_rows: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._batch_rows_logger = CsvLogger(self.out_dir / "batch_rows.csv")
        self._all_summary_logger = CsvLogger(self.out_dir / "all_summary_rows.csv")
        self._all_aggregate_summary_logger = CsvLogger(
            self.out_dir / "all_aggregate_summary_rows.csv"
        )

        if append_existing_all_summary_rows:
            for row in _load_existing_csv_rows(self.out_dir / "all_summary_rows.csv"):
                self._all_summary_logger.log(row)

        if append_existing_all_aggregate_summary_rows:
            for row in _load_existing_csv_rows(
                self.out_dir / "all_aggregate_summary_rows.csv"
            ):
                self._all_aggregate_summary_logger.log(row)

    def log_batch_rows(self, batch_rows: list[dict[str, Any]]) -> None:
        for row in batch_rows:
            self._batch_rows_logger.log(row)
        self._batch_rows_logger.flush()

    def log_summary_row(self, summary_row: dict[str, Any]) -> None:
        self._all_summary_logger.log(summary_row)
        self._all_summary_logger.flush()

    def log_aggregate_summary_row(self, summary_row: dict[str, Any]) -> None:
        self._all_aggregate_summary_logger.log(summary_row)
        self._all_aggregate_summary_logger.flush()

    def flush(self) -> None:
        self._batch_rows_logger.flush()
        self._all_summary_logger.flush()
        self._all_aggregate_summary_logger.flush()

    def close(self) -> None:
        self._batch_rows_logger.close()
        self._all_summary_logger.close()
        self._all_aggregate_summary_logger.close()