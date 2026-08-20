"""Lightweight, best-effort benchmarking for ``run_analysis.py``.

This module deliberately depends only on the Python standard library.  The
producer supplies its existing run metadata and timing, while temporary
executor observers record the task metrics that Coffea already reports and
time its local final accumulator merge.  Driver memory snapshots are single
reads of Linux ``/proc/self/status`` at workflow boundaries.
"""

from __future__ import annotations

import atexit
import csv
import math
import os
import statistics
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


LEGACY_CSV_COLUMNS = [
    "timestamp",
    "campaign",
    "experiment",
    "repeat",
    "backend",
    "input_json",
    "executor",
    "hist_list",
    "n_root_files",
    "n_events",
    "n_samples",
    "preprocessing_time_s",
    "submission_time_s",
    "processing_time_s",
    "accumulation_time_s",
    "runner_runtime_s",
    "serialization_time_s",
    "events_per_second",
    "measured_memory_min_mb",
    "measured_memory_q25_mb",
    "measured_memory_median_mb",
    "measured_memory_mean_mb",
    "measured_memory_q75_mb",
    "measured_memory_max_mb",
    "measured_memory_std_mb",
    "n_memory_samples",
    "output_pkl_size_mb",
    "n_histograms",
    "n_sumw2_histograms",
]

WORKER_MEMORY_COLUMNS = []
for _task_type in ("preprocessing", "processing"):
    WORKER_MEMORY_COLUMNS.extend(
        f"worker_{_task_type}_memory_{stat}_mb"
        for stat in ("min", "q25", "median", "mean", "q75", "max", "std")
    )
    WORKER_MEMORY_COLUMNS.append(f"worker_{_task_type}_n_samples")

DRIVER_MEMORY_COLUMNS = [
    f"driver_{snapshot}_{metric}_mb"
    for snapshot in (
        "start",
        "after_preprocessing",
        "after_runner",
        "before_serialization",
        "after_serialization",
        "after_final_merge",
    )
    for metric in ("rss", "peak_rss")
] + ["driver_final_peak_rss_mb"]

WORKER_RUNTIME_COLUMNS = []
for _task_type in ("preprocessing", "processing"):
    WORKER_RUNTIME_COLUMNS.extend(
        f"worker_{_task_type}_runtime_{stat}_s"
        for stat in ("min", "q25", "median", "mean", "q75", "max", "std")
    )
    WORKER_RUNTIME_COLUMNS.append(f"worker_{_task_type}_runtime_n_samples")

TASK_COUNT_COLUMNS = ["n_preprocessing_tasks", "n_processing_tasks"]

CSV_COLUMNS = (
    LEGACY_CSV_COLUMNS
    + WORKER_MEMORY_COLUMNS
    + DRIVER_MEMORY_COLUMNS
    + ["final_merge_time_s"]
    + WORKER_RUNTIME_COLUMNS
    + TASK_COUNT_COLUMNS
)


LEGACY_RESULTS_PATH = Path(__file__).with_name("benchmark_results.csv")
DEFAULT_RESULTS_PATH = Path(__file__).with_name("benchmark_results_extended.csv")


def parse_proc_status_memory(status_text: str) -> dict[str, float | None]:
    """Parse current and peak RSS from Linux ``/proc/*/status`` text."""

    values: dict[str, float | None] = {"rss_mb": None, "peak_rss_mb": None}
    field_names = {"VmRSS": "rss_mb", "VmHWM": "peak_rss_mb"}
    unit_scales = {
        "b": 1.0 / (1024.0 * 1024.0),
        "kb": 1.0 / 1024.0,
        "mb": 1.0,
    }

    for line in status_text.splitlines():
        name, separator, raw_value = line.partition(":")
        output_name = field_names.get(name)
        if not separator or output_name is None:
            continue
        fields = raw_value.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except (TypeError, ValueError):
            continue
        unit = fields[1].lower() if len(fields) > 1 else "b"
        scale = unit_scales.get(unit)
        if scale is not None and math.isfinite(value) and value >= 0:
            values[output_name] = value * scale

    return values


def read_proc_status_memory(
    status_path: str | os.PathLike[str] = "/proc/self/status",
) -> dict[str, float | None]:
    """Read driver RSS values, returning missing values on any procfs error."""

    try:
        status_text = Path(status_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {"rss_mb": None, "peak_rss_mb": None}
    try:
        return parse_proc_status_memory(status_text)
    except Exception:
        return {"rss_mb": None, "peak_rss_mb": None}


@contextmanager
def capture_executor_metrics_best_effort(
    recorder: Any, executor_name: str
) -> Iterator[None]:
    """Run analysis inside an observer whose own failures are suppressed."""

    if recorder is None:
        yield
        return
    try:
        observer = recorder.capture_executor_metrics(executor_name)
        observer.__enter__()
    except Exception:
        yield
        return

    try:
        yield
    except BaseException as analysis_error:
        try:
            suppress = observer.__exit__(
                type(analysis_error), analysis_error, analysis_error.__traceback__
            )
        except Exception:
            suppress = False
        if not suppress:
            raise
    else:
        try:
            observer.__exit__(None, None, None)
        except Exception:
            pass


def _quantile(values: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated quantile without requiring NumPy."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _memory_statistics(
    samples: list[float],
    *,
    prefix: str,
    count_column: str,
) -> dict[str, float | int | None]:
    """Compute descriptive statistics with caller-selected column names."""

    columns = {
        statistic: f"{prefix}_{statistic}_mb"
        for statistic in ("min", "q25", "median", "mean", "q75", "max", "std")
    }
    empty = {column: None for column in columns.values()}
    empty[count_column] = 0

    finite_samples = []
    for sample in samples:
        try:
            value = float(sample)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            finite_samples.append(value)
    if not finite_samples:
        return empty

    return {
        columns["min"]: min(finite_samples),
        columns["q25"]: _quantile(finite_samples, 0.25),
        columns["median"]: statistics.median(finite_samples),
        columns["mean"]: statistics.fmean(finite_samples),
        columns["q75"]: _quantile(finite_samples, 0.75),
        columns["max"]: max(finite_samples),
        columns["std"]: statistics.pstdev(finite_samples),
        count_column: len(finite_samples),
    }


def memory_statistics(samples: list[float]) -> dict[str, float | int | None]:
    """Compute the requested descriptive statistics for measured memory."""

    return _memory_statistics(
        samples,
        prefix="measured_memory",
        count_column="n_memory_samples",
    )


def _runtime_statistics(
    samples: list[float],
    *,
    task_type: str,
) -> dict[str, float | int | None]:
    """Compute worker-runtime statistics using the memory conventions."""

    prefix = f"worker_{task_type}_runtime"
    columns = {
        statistic: f"{prefix}_{statistic}_s"
        for statistic in ("min", "q25", "median", "mean", "q75", "max", "std")
    }
    count_column = f"{prefix}_n_samples"
    empty = {column: None for column in columns.values()}
    empty[count_column] = 0

    finite_samples = []
    for sample in samples:
        try:
            value = float(sample)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            finite_samples.append(value)
    if not finite_samples:
        return empty

    return {
        columns["min"]: min(finite_samples),
        columns["q25"]: _quantile(finite_samples, 0.25),
        columns["median"]: statistics.median(finite_samples),
        columns["mean"]: statistics.fmean(finite_samples),
        columns["q75"]: _quantile(finite_samples, 0.75),
        columns["max"]: max(finite_samples),
        columns["std"]: statistics.pstdev(finite_samples),
        count_column: len(finite_samples),
    }


def _is_histogram(value: Any) -> bool:
    """Recognize the histogram APIs serialized by the producer."""

    class_names = {cls.__name__ for cls in type(value).__mro__}
    if class_names.intersection({"Hist", "HistEFT", "SparseHist"}):
        return True
    return hasattr(value, "axes") and (
        callable(getattr(value, "fill", None))
        or callable(getattr(value, "values", None))
    )


def count_serialized_histograms(payload: Any) -> tuple[int, int]:
    """Count unique histogram objects and legacy ``_sumw2`` companions."""

    histogram_ids: set[int] = set()
    sumw2_ids: set[int] = set()
    visited_containers: set[int] = set()

    def visit(value: Any, key: Any = None) -> None:
        if _is_histogram(value):
            object_id = id(value)
            histogram_ids.add(object_id)
            if isinstance(key, str) and key.endswith("_sumw2"):
                sumw2_ids.add(object_id)
            return

        if isinstance(value, Mapping):
            object_id = id(value)
            if object_id in visited_containers:
                return
            visited_containers.add(object_id)
            for child_key, child_value in value.items():
                visit(child_value, child_key)
        elif isinstance(value, (list, tuple)):
            object_id = id(value)
            if object_id in visited_containers:
                return
            visited_containers.add(object_id)
            for child_value in value:
                visit(child_value)

    visit(payload)
    return len(histogram_ids), len(sumw2_ids)


class RunAnalysisBenchmark:
    """Collect and persist one benchmark record for a producer execution.

    All recorder operations are best effort: a benchmark failure must never
    affect the analysis result or execution flow.
    """

    def __init__(
        self,
        results_path: str | os.PathLike[str] | None = None,
        *,
        auto_finalize: bool = True,
    ) -> None:
        configured_path = os.environ.get("TOPEFT_BENCHMARK_RESULTS")
        selected_path = Path(results_path or configured_path or DEFAULT_RESULTS_PATH)
        # Never append richer rows to the reproducibility artifact, even if an
        # old environment override still points at it.
        if selected_path.resolve() == LEGACY_RESULTS_PATH.resolve():
            selected_path = DEFAULT_RESULTS_PATH
        self.results_path = selected_path
        self._wall_started = time.perf_counter()
        self._finalized = False
        self._metadata: dict[str, Any] = {}
        self._timings: dict[str, float] = {
            "preprocessing_time_s": 0.0,
            "submission_time_s": 0.0,
            "processing_time_s": 0.0,
            "accumulation_time_s": 0.0,
        }
        self._memory_samples: list[float] = []
        self._worker_memory_samples: dict[str, list[float]] = {
            "preprocessing": [],
            "processing": [],
        }
        self._worker_runtime_samples: dict[str, list[float]] = {
            "preprocessing": [],
            "processing": [],
        }
        self._task_counts: dict[str, int] = {
            "preprocessing": 0,
            "processing": 0,
        }
        self._driver_memory: dict[str, float | None] = {
            column: None for column in DRIVER_MEMORY_COLUMNS
        }
        self._output_metrics: dict[str, Any] = {
            "output_pkl_size_mb": None,
            "n_histograms": None,
            "n_sumw2_histograms": None,
        }
        self._n_events: int | None = None
        self._runner_runtime: float | None = None
        self._serialization_time: float | None = None
        self._final_merge_time: float | None = None
        self._processing_started: float | None = None
        self._submitted_at: float | None = None
        self._processed_at: float | None = None
        self._executor_stages_observed = False

        if auto_finalize:
            atexit.register(self.finalize)
        self.record_driver_memory("start")

    def configure(self, **metadata: Any) -> None:
        try:
            self._metadata.update(metadata)
            if "n_events" in metadata and metadata["n_events"] is not None:
                self._n_events = int(metadata["n_events"])
        except Exception:
            pass

    def record_memory_sample(
        self, measured_memory_mb: Any, task_type: str | None = None
    ) -> None:
        try:
            value = float(measured_memory_mb)
        except (TypeError, ValueError):
            return
        if math.isfinite(value) and value >= 0:
            self._memory_samples.append(value)
            if task_type in self._worker_memory_samples:
                self._worker_memory_samples[task_type].append(value)

    def record_worker_runtime_sample(
        self, runtime_seconds: Any, task_type: str
    ) -> None:
        """Record one valid Work Queue task runtime in seconds."""

        if task_type not in self._worker_runtime_samples:
            return
        try:
            value = float(runtime_seconds)
        except (OverflowError, TypeError, ValueError):
            return
        if math.isfinite(value) and value >= 0:
            self._worker_runtime_samples[task_type].append(value)

    def record_driver_memory(self, snapshot: str) -> None:
        """Take one best-effort current/peak RSS snapshot for a named boundary."""

        rss_column = f"driver_{snapshot}_rss_mb"
        peak_column = f"driver_{snapshot}_peak_rss_mb"
        if rss_column not in self._driver_memory or peak_column not in self._driver_memory:
            return
        try:
            memory = read_proc_status_memory()
            self._driver_memory[rss_column] = memory.get("rss_mb")
            self._driver_memory[peak_column] = memory.get("peak_rss_mb")
        except Exception:
            pass

    def record_runner_runtime(self, seconds: float, n_events: int | None = None) -> None:
        try:
            self._runner_runtime = max(0.0, float(seconds))
            if n_events is not None:
                self._n_events = int(n_events)
                self._metadata["n_events"] = self._n_events
            if not self._executor_stages_observed:
                self._timings["processing_time_s"] = self._runner_runtime
        except Exception:
            pass

    def record_serialization_time(self, seconds: float) -> None:
        try:
            self._serialization_time = max(0.0, float(seconds))
        except Exception:
            pass

    def record_output(self, output_path: str | os.PathLike[str], payload: Any) -> None:
        try:
            self._output_metrics["output_pkl_size_mb"] = (
                os.path.getsize(output_path) / (1024.0 * 1024.0)
            )
        except OSError:
            self._output_metrics["output_pkl_size_mb"] = None

        try:
            n_histograms, n_sumw2 = count_serialized_histograms(payload)
            self._output_metrics["n_histograms"] = n_histograms
            self._output_metrics["n_sumw2_histograms"] = n_sumw2
        except Exception:
            # Histogram introspection is diagnostic only.
            pass

    def _observe_work_queue_task(self, task: Any, queue: Any) -> None:
        try:
            successful = getattr(task, "successful", None)
            if callable(successful) and not successful():
                return

            category = str(getattr(task, "category", "")).lower()
            task_type = None
            if category == "preprocessing":
                task_type = "preprocessing"
            elif category.startswith("processing"):
                task_type = "processing"

            if task_type is not None:
                self._task_counts[task_type] += 1
                try:
                    runtime_seconds = float(task.cmd_execution_time) / 1e6
                except (AttributeError, OverflowError, TypeError, ValueError):
                    runtime_seconds = None
                self.record_worker_runtime_sample(runtime_seconds, task_type)

            monitor_mode = getattr(queue.executor, "resource_monitor", None)
            if monitor_mode and monitor_mode != "off":
                measured = getattr(task, "resources_measured", None)
                self.record_memory_sample(
                    getattr(measured, "memory", None), task_type=task_type
                )
        except Exception:
            pass

    def _update_work_queue_milestones(self, queue: Any) -> None:
        if self._processing_started is None:
            return
        try:
            stats = queue.stats_coffea
            events_total = stats["events_total"]
            now = time.perf_counter()
            if self._submitted_at is None and stats["events_submitted"] >= events_total:
                self._submitted_at = now
            if self._processed_at is None and stats["events_processed"] >= events_total:
                self._processed_at = now
        except Exception:
            pass

    @contextmanager
    def capture_executor_metrics(self, executor_name: str) -> Iterator[None]:
        """Observe existing Coffea stage and memory reporting during execution."""

        if executor_name not in {"work_queue", "taskvine"}:
            yield
            return

        patches: list[tuple[type, str, Any]] = []
        try:
            from coffea.processor import work_queue_tools

            queue_class = work_queue_tools.CoffeaWQ
            task_class = work_queue_tools.CoffeaWQTask
            original_report = task_class.report
            original_preprocessing = queue_class._preprocessing
            original_processing = queue_class._processing
            original_submit_processing_tasks = queue_class._submit_processing_tasks
            original_update_bars = queue_class._update_bars
            original_final_accumulation = getattr(
                queue_class, "_final_accumulation", None
            )
            recorder = self

            def report(task: Any, queue: Any) -> Any:
                recorder._observe_work_queue_task(task, queue)
                return original_report(task, queue)

            def preprocessing(queue: Any, *args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original_preprocessing(queue, *args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - started
                    previous = recorder._timings["preprocessing_time_s"] or 0.0
                    recorder._timings["preprocessing_time_s"] = previous + elapsed
                    recorder.record_driver_memory("after_preprocessing")

            def final_accumulation(queue: Any, *args: Any, **kwargs: Any) -> Any:
                """Time only Coffea's direct local ``accumulate_result_files`` call."""

                original_accumulate = work_queue_tools.accumulate_result_files

                def timed_accumulate(*merge_args: Any, **merge_kwargs: Any) -> Any:
                    started = time.perf_counter()
                    try:
                        return original_accumulate(*merge_args, **merge_kwargs)
                    finally:
                        elapsed = time.perf_counter() - started
                        previous = recorder._final_merge_time or 0.0
                        recorder._final_merge_time = previous + elapsed
                        recorder.record_driver_memory("after_final_merge")

                work_queue_tools.accumulate_result_files = timed_accumulate
                try:
                    return original_final_accumulation(queue, *args, **kwargs)
                finally:
                    work_queue_tools.accumulate_result_files = original_accumulate

            def processing(queue: Any, *args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                recorder._executor_stages_observed = True
                recorder._processing_started = started
                recorder._submitted_at = None
                recorder._processed_at = None
                try:
                    return original_processing(queue, *args, **kwargs)
                finally:
                    ended = time.perf_counter()
                    try:
                        recorder._n_events = int(
                            queue.stats_coffea["events_processed"]
                        )
                        recorder._metadata["n_events"] = recorder._n_events
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                    submitted = recorder._submitted_at or started
                    processed = recorder._processed_at or ended
                    submitted = min(max(submitted, started), ended)
                    processed = min(max(processed, submitted), ended)
                    recorder._timings["submission_time_s"] = submitted - started
                    recorder._timings["processing_time_s"] = processed - submitted
                    recorder._timings["accumulation_time_s"] = ended - processed

            def update_bars(queue: Any, *args: Any, **kwargs: Any) -> Any:
                recorder._update_work_queue_milestones(queue)
                return original_update_bars(queue, *args, **kwargs)

            def submit_processing_tasks(queue: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_submit_processing_tasks(queue, *args, **kwargs)
                recorder._update_work_queue_milestones(queue)
                return result

            replacements = [
                (task_class, "report", report),
                (queue_class, "_preprocessing", preprocessing),
                (queue_class, "_processing", processing),
                (queue_class, "_submit_processing_tasks", submit_processing_tasks),
                (queue_class, "_update_bars", update_bars),
            ]
            if callable(original_final_accumulation) and hasattr(
                work_queue_tools, "accumulate_result_files"
            ):
                replacements.append(
                    (queue_class, "_final_accumulation", final_accumulation)
                )

            for owner, name, replacement in replacements:
                patches.append((owner, name, getattr(owner, name)))
                setattr(owner, name, replacement)
        except Exception:
            for owner, name, original in reversed(patches):
                try:
                    setattr(owner, name, original)
                except Exception:
                    pass
            patches = []

        try:
            yield
        finally:
            for owner, name, original in reversed(patches):
                try:
                    setattr(owner, name, original)
                except Exception:
                    pass

    def _build_row(self) -> dict[str, Any]:
        total_runtime = self._runner_runtime
        serialization_time = self._serialization_time
        if total_runtime is None:
            total_runtime = max(0.0, time.perf_counter() - self._wall_started)
        events_per_second = None
        if self._n_events is not None and total_runtime > 0:
            events_per_second = self._n_events / total_runtime

        row = {column: None for column in CSV_COLUMNS}
        row.update(self._metadata)
        row.update(self._timings)
        row.update(memory_statistics(self._memory_samples))
        for task_type, samples in self._worker_memory_samples.items():
            row.update(
                _memory_statistics(
                    samples,
                    prefix=f"worker_{task_type}_memory",
                    count_column=f"worker_{task_type}_n_samples",
                )
            )
        for task_type, samples in self._worker_runtime_samples.items():
            row.update(_runtime_statistics(samples, task_type=task_type))
            row[f"n_{task_type}_tasks"] = self._task_counts[task_type]
        row.update(self._driver_memory)
        row.update(self._output_metrics)
        final_memory = read_proc_status_memory()
        row["driver_final_peak_rss_mb"] = final_memory.get("peak_rss_mb")
        row.update(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "runner_runtime_s": total_runtime,
                "serialization_time_s": serialization_time,
                "events_per_second": events_per_second,
                "final_merge_time_s": self._final_merge_time,
            }
        )
        return row

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return value

    def _append_row(self, row: dict[str, Any]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a+", newline="", encoding="utf-8") as stream:
            try:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            stream.seek(0, os.SEEK_END)
            write_header = stream.tell() == 0
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {column: self._csv_value(row.get(column)) for column in CSV_COLUMNS}
            )
            stream.flush()

    @staticmethod
    def _summary_value(value: Any, *, scientific: bool = False) -> str:
        if value is None:
            return "n/a"
        if scientific:
            return f"{float(value):.2e}"
        return f"{float(value):.1f}"

    def _print_summary(self, row: dict[str, Any]) -> None:
        if self.results_path.resolve() == DEFAULT_RESULTS_PATH.resolve():
            display_path = os.path.join("benchmark", self.results_path.name)
        else:
            display_path = str(self.results_path)
        print("\n========================================================")
        print("Benchmark summary")
        print("========================================================\n")
        print(f"Backend              : {row.get('backend') or 'n/a'}")
        print(
            "Runner runtime       : {} s".format(
                self._summary_value(row.get("runner_runtime_s"))
            )
        )
        print(
            "Serialization        : {} s".format(
                self._summary_value(row.get("serialization_time_s"))
            )
        )
        print(
            "Events/sec           : {}".format(
                self._summary_value(row.get("events_per_second"), scientific=True)
            )
        )
        print("\nMeasured memory (MB)\n")
        print(f"Min                  : {self._summary_value(row.get('measured_memory_min_mb'))}")
        print(f"Median               : {self._summary_value(row.get('measured_memory_median_mb'))}")
        print(f"Mean                 : {self._summary_value(row.get('measured_memory_mean_mb'))}")
        print(f"Max                  : {self._summary_value(row.get('measured_memory_max_mb'))}")
        print("\nWorker processing memory (MB)\n")
        print(
            "Median               : {}".format(
                self._summary_value(row.get("worker_processing_memory_median_mb"))
            )
        )
        print(
            "Max                  : {}".format(
                self._summary_value(row.get("worker_processing_memory_max_mb"))
            )
        )
        print("\nWorker processing runtime (s)\n")
        print(
            "Median               : {}".format(
                self._summary_value(row.get("worker_processing_runtime_median_s"))
            )
        )
        print(
            "Max                  : {}".format(
                self._summary_value(row.get("worker_processing_runtime_max_s"))
            )
        )
        print(f"Tasks                : {row.get('n_processing_tasks', 0)}")
        print("\nDriver memory (MB)\n")
        print(
            "After runner          : {}".format(
                self._summary_value(row.get("driver_after_runner_rss_mb"))
            )
        )
        if row.get("driver_after_final_merge_rss_mb") is not None:
            print(
                "After final merge     : {}".format(
                    self._summary_value(row.get("driver_after_final_merge_rss_mb"))
                )
            )
        print(
            "Final peak            : {}".format(
                self._summary_value(row.get("driver_final_peak_rss_mb"))
            )
        )
        if row.get("final_merge_time_s") is not None:
            print(
                "\nFinal merge          : {} s".format(
                    self._summary_value(row.get("final_merge_time_s"))
                )
            )
        print(
            "\nOutput PKL           : {} MB".format(
                self._summary_value(row.get("output_pkl_size_mb"))
            )
        )
        print(f"\nBenchmark row written to:\n{display_path}")

    def finalize(self) -> dict[str, Any] | None:
        """Append the result once and print the sole benchmark log block."""

        if self._finalized:
            return None
        self._finalized = True
        try:
            atexit.unregister(self.finalize)
        except Exception:
            pass

        try:
            row = self._build_row()
            self._append_row(row)
            self._print_summary(row)
            return row
        except Exception:
            # Recording must never make a physics run fail or change its result.
            return None
