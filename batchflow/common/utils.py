from __future__ import annotations

from pathlib import Path
import csv
from typing import Any, Optional
import threading
import time
import os

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None
    _NVML_AVAILABLE = False


class ResourceMonitor:
    def __init__(self, *, sample_interval_seconds: float = 0.25, logger: Any = None) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.logger = logger

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # sample counts
        self._samples = 0
        self._gpu_samples = 0

        # main process CPU / memory
        self._process_cpu_percent_raw_sum = 0.0
        self._process_cpu_percent_normalized_sum = 0.0
        self._process_cpu_percent_affinity_normalized_sum = 0.0
        self._rss_mb_sum = 0.0

        # process tree CPU / memory
        self._process_tree_cpu_percent_raw_sum = 0.0
        self._process_tree_cpu_percent_normalized_sum = 0.0
        self._process_tree_cpu_percent_affinity_normalized_sum = 0.0
        self._process_tree_rss_mb_sum = 0.0
        self._process_tree_process_count_sum = 0.0

        # system CPU / memory
        self._system_cpu_percent_sum = 0.0
        self._system_ram_percent_sum = 0.0

        # GPU
        self._gpu_util_percent_sum = 0.0
        self._gpu_mem_used_mb_sum = 0.0
        self._gpu_mem_percent_sum = 0.0

        self._process = psutil.Process() if psutil is not None else None
        self._logical_cpu_count = psutil.cpu_count(logical=True) if psutil is not None else 1

        try:
            self._affinity_cpu_count = len(os.sched_getaffinity(0))
        except Exception:
            self._affinity_cpu_count = self._logical_cpu_count

        self._nvml_initialized = False
        self._gpu_handle = None

    def start(self) -> None:
        if self._process is not None:
            # prime main process CPU counters
            try:
                self._process.cpu_percent(interval=None)
            except Exception:
                pass

            # prime child process CPU counters
            try:
                for child in self._process.children(recursive=True):
                    try:
                        child.cpu_percent(interval=None)
                    except Exception:
                        pass
            except Exception:
                pass

        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                if pynvml.nvmlDeviceGetCount() > 0:
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._gpu_handle = None
                self._nvml_initialized = False

        self._thread = threading.Thread(
            target=self._run,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        with self._lock:
            return {
                # main process
                # "avg_process_cpu_percent_raw": self._safe_avg(
                #     self._process_cpu_percent_raw_sum, self._samples
                # ),
                # "avg_process_cpu_percent_normalized": self._safe_avg(
                #     self._process_cpu_percent_normalized_sum, self._samples
                # ),
                # "avg_process_cpu_percent_affinity_normalized": self._safe_avg(
                #     self._process_cpu_percent_affinity_normalized_sum, self._samples
                # ),
                # "avg_rss_mb": self._safe_avg(self._rss_mb_sum, self._samples),

                # process tree
                    #Total CPU usage of your process tree across ALL cores (i.e. Percent of one core × number of cores used)
                "avg_process_tree_cpu_percent_raw": self._safe_avg(
                    self._process_tree_cpu_percent_raw_sum, self._samples
                ),
                #Fraction of the entire machine’s CPU capacity used by your process tree (i.e. Percent of one core × number of cores used / total number of cores)
                "avg_process_tree_cpu_percent_normalized": self._safe_avg(
                    self._process_tree_cpu_percent_normalized_sum, self._samples
                ),

                #How much of your allowed CPU budget you are using (i.e. Percent of one core × number of cores used / number of cores in your CPU affinity)
                "avg_process_tree_cpu_percent_affinity_normalized": self._safe_avg(
                    self._process_tree_cpu_percent_affinity_normalized_sum, self._samples
                ),

                
                "avg_process_tree_rss_mb": self._safe_avg(
                    self._process_tree_rss_mb_sum, self._samples
                ),
                # "avg_process_tree_process_count": self._safe_avg(
                #     self._process_tree_process_count_sum, self._samples
                # ),

                # affinity/system
                "cpu_affinity_core_count": float(self._affinity_cpu_count),
                "logical_cpu_count": float(self._logical_cpu_count),
                "avg_system_cpu_percent": self._safe_avg(
                    self._system_cpu_percent_sum, self._samples
                ),
                "avg_system_ram_percent": self._safe_avg(
                    self._system_ram_percent_sum, self._samples
                ),

                # gpu
                "avg_gpu_util_percent": self._safe_avg(
                    self._gpu_util_percent_sum, self._gpu_samples
                ),
                "avg_gpu_mem_used_mb": self._safe_avg(
                    self._gpu_mem_used_mb_sum, self._gpu_samples
                ),
                "avg_gpu_mem_percent": self._safe_avg(
                    self._gpu_mem_percent_sum, self._gpu_samples
                ),

                # "resource_samples": float(self._samples),
                # "gpu_resource_samples": float(self._gpu_samples),
            }

    def _safe_avg(self, total: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return float(total / count)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                if self.logger is not None:
                    self.logger.warning("Resource monitor sampling failed", exc_info=True)
                else:
                    print("resource monitor sampling failed")
            time.sleep(self.sample_interval_seconds)

    def _sample_once(self) -> None:
        process_cpu_percent_raw = 0.0
        process_cpu_percent_normalized = 0.0
        process_cpu_percent_affinity_normalized = 0.0
        rss_mb = 0.0

        process_tree_cpu_percent_raw = 0.0
        process_tree_cpu_percent_normalized = 0.0
        process_tree_cpu_percent_affinity_normalized = 0.0
        process_tree_rss_mb = 0.0
        process_tree_process_count = 0.0

        system_cpu_percent = 0.0
        system_ram_percent = 0.0

        if self._process is not None and psutil is not None:
            # ----- main process -----
            try:
                process_cpu_percent_raw = float(self._process.cpu_percent(interval=None))
            except Exception:
                process_cpu_percent_raw = 0.0

            denom_total = float(max(1, self._logical_cpu_count))
            process_cpu_percent_normalized = process_cpu_percent_raw / denom_total

            denom_affinity = float(max(1, self._affinity_cpu_count))
            process_cpu_percent_affinity_normalized = process_cpu_percent_raw / denom_affinity

            try:
                rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                rss_mb = 0.0

            # ----- process tree -----
            procs: list[Any] = []
            try:
                procs = [self._process] + self._process.children(recursive=True)
            except Exception:
                procs = [self._process]

            tree_cpu_raw = 0.0
            tree_rss_mb = 0.0
            live_count = 0

            for proc in procs:
                try:
                    tree_cpu_raw += float(proc.cpu_percent(interval=None))
                    tree_rss_mb += float(proc.memory_info().rss) / (1024.0 * 1024.0)
                    live_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception:
                    continue

            process_tree_cpu_percent_raw = tree_cpu_raw
            process_tree_cpu_percent_normalized = tree_cpu_raw / denom_total
            process_tree_cpu_percent_affinity_normalized = tree_cpu_raw / denom_affinity
            process_tree_rss_mb = tree_rss_mb
            process_tree_process_count = float(live_count)

            # ----- system -----
            try:
                system_cpu_percent = float(psutil.cpu_percent(interval=None))
            except Exception:
                system_cpu_percent = 0.0

            try:
                system_ram_percent = float(psutil.virtual_memory().percent)
            except Exception:
                system_ram_percent = 0.0

        gpu_util_percent = 0.0
        gpu_mem_used_mb = 0.0
        gpu_mem_percent = 0.0
        have_gpu = False

        if self._gpu_handle is not None and _NVML_AVAILABLE:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)

                gpu_util_percent = float(util.gpu)
                gpu_mem_used_mb = float(mem.used) / (1024.0 * 1024.0)
                gpu_mem_percent = (
                    float(mem.used) / float(mem.total) * 100.0 if mem.total > 0 else 0.0
                )
                have_gpu = True
            except Exception:
                have_gpu = False

        with self._lock:
            self._samples += 1

            self._process_cpu_percent_raw_sum += process_cpu_percent_raw
            self._process_cpu_percent_normalized_sum += process_cpu_percent_normalized
            self._process_cpu_percent_affinity_normalized_sum += process_cpu_percent_affinity_normalized
            self._rss_mb_sum += rss_mb

            self._process_tree_cpu_percent_raw_sum += process_tree_cpu_percent_raw
            self._process_tree_cpu_percent_normalized_sum += process_tree_cpu_percent_normalized
            self._process_tree_cpu_percent_affinity_normalized_sum += process_tree_cpu_percent_affinity_normalized
            self._process_tree_rss_mb_sum += process_tree_rss_mb
            self._process_tree_process_count_sum += process_tree_process_count

            self._system_cpu_percent_sum += system_cpu_percent
            self._system_ram_percent_sum += system_ram_percent

            if have_gpu:
                self._gpu_samples += 1
                self._gpu_util_percent_sum += gpu_util_percent
                self._gpu_mem_used_mb_sum += gpu_mem_used_mb
                self._gpu_mem_percent_sum += gpu_mem_percent


class CsvLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        flush_every_n: int = 10,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []

        self._flush_every_n = max(1, int(flush_every_n))

    def log(self, row: dict[str, Any]) -> None:
        new_fields = [k for k in row.keys() if k not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(new_fields)

        self._rows.append(row)

        if len(self._rows) % self._flush_every_n == 0:
            self.flush()

    def _write_all_rows(self) -> None:
        if not self._rows:
            return

        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def flush(self) -> None:
        self._write_all_rows()

    def close(self) -> None:
        self._write_all_rows()


class SmoothedMeter:
    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.window_size:
            self.values.pop(0)

    @property
    def avg(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
class CsvLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        flush_every_n: int = 10,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []

        self._flush_every_n = max(1, int(flush_every_n))

    def log(self, row: dict[str, Any]) -> None:
        # update schema
        new_fields = [k for k in row.keys() if k not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(new_fields)

        self._rows.append(row)

        # auto flush
        if len(self._rows) % self._flush_every_n == 0:
            self.flush()

    def _write_all_rows(self) -> None:
        if not self._rows:
            return

        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def flush(self) -> None:
        self._write_all_rows()

    def close(self) -> None:
        self._write_all_rows()

class SmoothedMeter:
    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.window_size:
            self.values.pop(0)

    @property
    def avg(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
