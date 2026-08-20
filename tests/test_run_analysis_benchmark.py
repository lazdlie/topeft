import csv
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.topeft_run2.benchmark.benchmark_run_analysis import (
    CSV_COLUMNS,
    DEFAULT_RESULTS_PATH,
    DRIVER_MEMORY_COLUMNS,
    LEGACY_CSV_COLUMNS,
    LEGACY_RESULTS_PATH,
    RunAnalysisBenchmark,
    TASK_COUNT_COLUMNS,
    WORKER_MEMORY_COLUMNS,
    WORKER_RUNTIME_COLUMNS,
    capture_executor_metrics_best_effort,
    count_serialized_histograms,
    memory_statistics,
    parse_proc_status_memory,
    read_proc_status_memory,
)

_PREVIOUS_EXTENDED_COLUMNS = (
    LEGACY_CSV_COLUMNS
    + WORKER_MEMORY_COLUMNS
    + DRIVER_MEMORY_COLUMNS
    + ["final_merge_time_s"]
)


class _FakeHistogram:
    axes = ()

    def fill(self):
        return None


def test_memory_statistics_uses_all_measured_samples():
    stats = memory_statistics([100.0, 200.0, 300.0, 400.0])

    assert stats["measured_memory_min_mb"] == 100.0
    assert stats["measured_memory_q25_mb"] == 175.0
    assert stats["measured_memory_median_mb"] == 250.0
    assert stats["measured_memory_mean_mb"] == 250.0
    assert stats["measured_memory_q75_mb"] == 325.0
    assert stats["measured_memory_max_mb"] == 400.0
    assert stats["measured_memory_std_mb"] == pytest.approx(111.80339887498948)
    assert stats["n_memory_samples"] == 4


def test_memory_statistics_ignores_invalid_samples():
    stats = memory_statistics([None, "bad", -1, float("nan"), float("inf"), 64])

    assert stats["measured_memory_min_mb"] == 64
    assert stats["measured_memory_max_mb"] == 64
    assert stats["n_memory_samples"] == 1


def test_parse_proc_status_memory_converts_kb_to_mb():
    status = """Name:\tpython\nVmHWM:\t4096 kB\nVmRSS:\t3072 kB\nThreads:\t1\n"""

    assert parse_proc_status_memory(status) == {
        "rss_mb": 3.0,
        "peak_rss_mb": 4.0,
    }


def test_proc_status_memory_handles_missing_file_and_values(tmp_path):
    incomplete_status = tmp_path / "status"
    incomplete_status.write_text("Name:\tpython\nVmRSS:\tnot-a-number kB\n")

    assert read_proc_status_memory(incomplete_status) == {
        "rss_mb": None,
        "peak_rss_mb": None,
    }
    assert read_proc_status_memory(tmp_path / "missing") == {
        "rss_mb": None,
        "peak_rss_mb": None,
    }


def test_count_serialized_histograms_includes_sumw2_companions():
    output = {
        "njets": _FakeHistogram(),
        "njets_sumw2": _FakeHistogram(),
        "metadata": {"value": 1},
    }

    assert count_serialized_histograms(output) == (2, 1)


def test_finalize_creates_header_and_appends_one_extended_row(
    tmp_path, capsys, monkeypatch
):
    results_path = tmp_path / "benchmark_results.csv"
    output_path = tmp_path / "output.pkl.gz"
    output_path.write_bytes(b"12345678")
    monkeypatch.setattr(
        "analysis.topeft_run2.benchmark.benchmark_run_analysis.read_proc_status_memory",
        lambda: {"rss_mb": 80.0, "peak_rss_mb": 120.0},
    )
    benchmark = RunAnalysisBenchmark(results_path, auto_finalize=False)
    benchmark.configure(
        campaign="UL17",
        experiment="smoke",
        repeat=1,
        backend="MultiCell",
        input_json="sample.json",
        executor="futures",
        hist_list="njets",
        n_root_files=1,
        n_events=999,
        n_samples=1,
    )
    benchmark.record_memory_sample(128)
    benchmark.record_memory_sample(256)
    benchmark.record_memory_sample(100, task_type="preprocessing")
    benchmark.record_memory_sample(200, task_type="processing")
    benchmark.record_runner_runtime(2.0, 100)
    benchmark.record_driver_memory("after_runner")
    benchmark.record_driver_memory("before_serialization")
    benchmark.record_driver_memory("after_serialization")
    assert benchmark._serialization_time is None
    benchmark.record_serialization_time(0.25)
    assert benchmark._serialization_time == 0.25
    benchmark.record_output(output_path, {"njets": _FakeHistogram()})

    row = benchmark.finalize()
    assert row is not None
    assert benchmark.finalize() is None
    assert row["preprocessing_time_s"] == 0.0
    assert row["submission_time_s"] == 0.0
    assert row["processing_time_s"] == 2.0
    assert row["accumulation_time_s"] == 0.0
    assert row["runner_runtime_s"] == 2.0
    assert row["serialization_time_s"] == 0.25
    assert row["n_events"] == 100
    assert row["events_per_second"] == 50.0
    assert row["worker_preprocessing_memory_median_mb"] == 100.0
    assert row["worker_preprocessing_n_samples"] == 1
    assert row["worker_processing_memory_median_mb"] == 200.0
    assert row["worker_processing_n_samples"] == 1
    assert row["worker_preprocessing_runtime_n_samples"] == 0
    assert row["worker_processing_runtime_n_samples"] == 0
    assert row["n_preprocessing_tasks"] == 0
    assert row["n_processing_tasks"] == 0
    assert row["driver_start_rss_mb"] == 80.0
    assert row["driver_after_runner_peak_rss_mb"] == 120.0
    assert row["driver_before_serialization_rss_mb"] == 80.0
    assert row["driver_after_serialization_peak_rss_mb"] == 120.0
    assert row["driver_final_peak_rss_mb"] == 120.0

    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0].keys() == dict.fromkeys(CSV_COLUMNS).keys()
    assert len(rows) == 1
    assert rows[0]["backend"] == "MultiCell"
    assert rows[0]["runner_runtime_s"] == "2.000000"
    assert rows[0]["serialization_time_s"] == "0.250000"
    assert rows[0]["n_events"] == "100"
    assert rows[0]["n_memory_samples"] == "4"
    assert rows[0]["n_histograms"] == "1"
    assert rows[0]["n_sumw2_histograms"] == "0"
    assert float(rows[0]["output_pkl_size_mb"]) == pytest.approx(
        output_path.stat().st_size / (1024 * 1024), abs=1e-6
    )

    summary = capsys.readouterr().out
    assert "Benchmark summary" in summary
    assert "Backend              : MultiCell" in summary
    assert "Runner runtime       : 2.0 s" in summary
    assert "Serialization        : 0.2 s" in summary
    assert "Worker processing runtime (s)" in summary
    assert "Tasks                : 0" in summary


def test_work_queue_observer_classifies_successful_task_memory_and_restores_hooks(
    monkeypatch,
):
    class FakeTask:
        def __init__(
            self,
            category="processing",
            memory=321.0,
            succeeded=True,
            runtime_us=2_000_000,
        ):
            self.category = category
            self.resources_measured = SimpleNamespace(memory=memory)
            self._succeeded = succeeded
            self.cmd_execution_time = runtime_us

        def successful(self):
            return self._succeeded

        def report(self, queue):
            return "reported"

    class FakeQueue:
        def __init__(self):
            self.executor = SimpleNamespace(resource_monitor="measure")
            self.stats_coffea = {
                "events_total": 10,
                "events_submitted": 0,
                "events_processed": 0,
            }

        def _preprocessing(self):
            return "preprocessed"

        def _submit_processing_tasks(self):
            self.stats_coffea["events_submitted"] = 10

        def _update_bars(self):
            return None

        def _processing(self):
            self._submit_processing_tasks()
            self.stats_coffea["events_processed"] = 10
            self._update_bars()
            return "processed"

        def _final_accumulation(self, accumulator):
            return fake_tools.accumulate_result_files([1], accumulator)

    def fake_accumulate_result_files(result_files, accumulator):
        return accumulator + result_files

    fake_tools = types.ModuleType("coffea.processor.work_queue_tools")
    fake_tools.CoffeaWQ = FakeQueue
    fake_tools.CoffeaWQTask = FakeTask
    fake_tools.accumulate_result_files = fake_accumulate_result_files
    fake_processor = types.ModuleType("coffea.processor")
    fake_processor.work_queue_tools = fake_tools
    fake_coffea = types.ModuleType("coffea")
    fake_coffea.processor = fake_processor
    monkeypatch.setitem(sys.modules, "coffea", fake_coffea)
    monkeypatch.setitem(sys.modules, "coffea.processor", fake_processor)
    monkeypatch.setitem(sys.modules, "coffea.processor.work_queue_tools", fake_tools)

    original_report = FakeTask.report
    original_processing = FakeQueue._processing
    benchmark = RunAnalysisBenchmark(auto_finalize=False)
    benchmark.configure(n_events=999)
    queue = FakeQueue()

    with benchmark.capture_executor_metrics("work_queue"):
        assert FakeTask("preprocessing", 100, runtime_us=1_000_000).report(queue) == "reported"
        assert FakeTask("processing", 200, runtime_us=2_000_000).report(queue) == "reported"
        assert FakeTask("accumulating", 300).report(queue) == "reported"
        assert FakeTask(
            "processing", 999, succeeded=False, runtime_us=999_000_000
        ).report(queue) == "reported"
        assert FakeTask("processing", None, runtime_us=None).report(queue) == "reported"
        assert queue._preprocessing() == "preprocessed"
        assert queue._processing() == "processed"
        assert queue._final_accumulation([0]) == [0, 1]

    assert FakeTask.report is original_report
    assert FakeQueue._processing is original_processing
    assert benchmark._memory_samples == [100.0, 200.0, 300.0]
    assert benchmark._worker_memory_samples["preprocessing"] == [100.0]
    assert benchmark._worker_memory_samples["processing"] == [200.0]
    assert benchmark._worker_runtime_samples["preprocessing"] == [1.0]
    assert benchmark._worker_runtime_samples["processing"] == [2.0]
    assert benchmark._task_counts == {"preprocessing": 1, "processing": 2}
    assert benchmark._n_events == 10
    assert benchmark._metadata["n_events"] == 10
    assert benchmark._executor_stages_observed is True
    assert all(value >= 0 for value in benchmark._timings.values())
    assert benchmark._final_merge_time is not None
    assert benchmark._driver_memory["driver_after_preprocessing_rss_mb"] is not None
    assert benchmark._driver_memory["driver_after_final_merge_rss_mb"] is not None


def test_worker_runtime_statistics_are_independent_and_ignore_invalid_values():
    benchmark = RunAnalysisBenchmark(auto_finalize=False)
    for runtime in (1, 2, 3, 4, None, "bad", -1, float("nan"), float("inf")):
        benchmark.record_worker_runtime_sample(runtime, "preprocessing")
    for runtime in (10, 20, 30):
        benchmark.record_worker_runtime_sample(runtime, "processing")

    row = benchmark._build_row()

    assert row["worker_preprocessing_runtime_min_s"] == 1
    assert row["worker_preprocessing_runtime_q25_s"] == 1.75
    assert row["worker_preprocessing_runtime_median_s"] == 2.5
    assert row["worker_preprocessing_runtime_mean_s"] == 2.5
    assert row["worker_preprocessing_runtime_q75_s"] == 3.25
    assert row["worker_preprocessing_runtime_max_s"] == 4
    assert row["worker_preprocessing_runtime_std_s"] == pytest.approx(
        1.118033988749895
    )
    assert row["worker_preprocessing_runtime_n_samples"] == 4
    assert row["worker_processing_runtime_min_s"] == 10
    assert row["worker_processing_runtime_median_s"] == 20
    assert row["worker_processing_runtime_mean_s"] == 20
    assert row["worker_processing_runtime_max_s"] == 30
    assert row["worker_processing_runtime_std_s"] == pytest.approx(
        8.16496580927726
    )
    assert row["worker_processing_runtime_n_samples"] == 3


def test_failed_retries_do_not_increment_successful_task_counts():
    class FakeTask:
        def __init__(self, category, succeeded, runtime_us, memory):
            self.category = category
            self._succeeded = succeeded
            self.cmd_execution_time = runtime_us
            self.resources_measured = SimpleNamespace(memory=memory)

        def successful(self):
            return self._succeeded

    queue = SimpleNamespace(executor=SimpleNamespace(resource_monitor="measure"))
    benchmark = RunAnalysisBenchmark(auto_finalize=False)

    # Two failed attempts followed by one successful retry count as one task.
    benchmark._observe_work_queue_task(
        FakeTask("processing", False, 90_000_000, 900), queue
    )
    benchmark._observe_work_queue_task(
        FakeTask("processing", False, 80_000_000, 800), queue
    )
    benchmark._observe_work_queue_task(
        FakeTask("processing", True, 5_000_000, 200), queue
    )
    benchmark._observe_work_queue_task(
        FakeTask("preprocessing", False, 70_000_000, 700), queue
    )
    benchmark._observe_work_queue_task(
        FakeTask("preprocessing", True, 1_000_000, 100), queue
    )

    row = benchmark._build_row()
    assert row["n_preprocessing_tasks"] == 1
    assert row["n_processing_tasks"] == 1
    assert row["worker_preprocessing_runtime_n_samples"] == 1
    assert row["worker_processing_runtime_n_samples"] == 1
    assert row["worker_preprocessing_runtime_median_s"] == 1
    assert row["worker_processing_runtime_median_s"] == 5
    assert benchmark._memory_samples == [200.0, 100.0]


def test_task_runtime_observability_does_not_change_previous_fields(monkeypatch):
    class FakeTask:
        category = "processing"
        cmd_execution_time = 2_000_000

        def successful(self):
            return True

    monkeypatch.setattr(
        "analysis.topeft_run2.benchmark.benchmark_run_analysis.read_proc_status_memory",
        lambda: {"rss_mb": 80.0, "peak_rss_mb": 120.0},
    )
    queue = SimpleNamespace(executor=SimpleNamespace(resource_monitor="off"))
    benchmark = RunAnalysisBenchmark(auto_finalize=False)
    benchmark.configure(n_events=100, backend="MultiCell")
    benchmark.record_runner_runtime(2.0, n_events=100)
    before = benchmark._build_row()

    benchmark._observe_work_queue_task(FakeTask(), queue)
    after = benchmark._build_row()

    assert {column: before[column] for column in _PREVIOUS_EXTENDED_COLUMNS} == {
        column: after[column] for column in _PREVIOUS_EXTENDED_COLUMNS
    }
    assert after["worker_processing_runtime_median_s"] == 2
    assert after["n_processing_tasks"] == 1


def test_final_merge_hook_times_only_real_merge_call(monkeypatch):
    calls = []

    class FakeTask:
        def report(self, queue):
            return True

    class FakeQueue:
        def _preprocessing(self):
            return None

        def _processing(self):
            return None

        def _submit_processing_tasks(self):
            return None

        def _update_bars(self):
            return None

        def _final_accumulation(self, accumulator):
            calls.append("before")
            result = fake_tools.accumulate_result_files([2], accumulator)
            calls.append("after")
            return result

    def fake_accumulate_result_files(result_files, accumulator):
        calls.append("merge")
        return accumulator + result_files

    fake_tools = types.ModuleType("coffea.processor.work_queue_tools")
    fake_tools.CoffeaWQ = FakeQueue
    fake_tools.CoffeaWQTask = FakeTask
    fake_tools.accumulate_result_files = fake_accumulate_result_files
    fake_processor = types.ModuleType("coffea.processor")
    fake_processor.work_queue_tools = fake_tools
    fake_coffea = types.ModuleType("coffea")
    fake_coffea.processor = fake_processor
    monkeypatch.setitem(sys.modules, "coffea", fake_coffea)
    monkeypatch.setitem(sys.modules, "coffea.processor", fake_processor)
    monkeypatch.setitem(sys.modules, "coffea.processor.work_queue_tools", fake_tools)

    benchmark = RunAnalysisBenchmark(auto_finalize=False)
    original_accumulate = fake_tools.accumulate_result_files
    with benchmark.capture_executor_metrics("work_queue"):
        assert FakeQueue()._final_accumulation([1]) == [1, 2]

    assert calls == ["before", "merge", "after"]
    assert benchmark._final_merge_time is not None
    assert benchmark._final_merge_time >= 0
    assert fake_tools.accumulate_result_files is original_accumulate


def test_benchmark_failures_are_best_effort(tmp_path, monkeypatch):
    benchmark = RunAnalysisBenchmark(tmp_path / "results.csv", auto_finalize=False)
    monkeypatch.setattr(benchmark, "_build_row", lambda: (_ for _ in ()).throw(OSError()))

    assert benchmark.finalize() is None


def test_broken_executor_observer_does_not_affect_analysis():
    class BrokenBenchmark:
        def capture_executor_metrics(self, executor_name):
            raise RuntimeError("benchmark observer failed")

    analysis_completed = False
    with capture_executor_metrics_best_effort(BrokenBenchmark(), "work_queue"):
        analysis_completed = True

    assert analysis_completed is True


def test_extended_results_file_contains_only_the_declared_header():
    header = DEFAULT_RESULTS_PATH.read_text(encoding="utf-8").strip().splitlines()[0]
    assert next(csv.reader([header])) == CSV_COLUMNS


def test_new_columns_are_appended_after_the_previous_extended_schema():
    expected_new_columns = [
        "worker_preprocessing_runtime_min_s",
        "worker_preprocessing_runtime_q25_s",
        "worker_preprocessing_runtime_median_s",
        "worker_preprocessing_runtime_mean_s",
        "worker_preprocessing_runtime_q75_s",
        "worker_preprocessing_runtime_max_s",
        "worker_preprocessing_runtime_std_s",
        "worker_preprocessing_runtime_n_samples",
        "worker_processing_runtime_min_s",
        "worker_processing_runtime_q25_s",
        "worker_processing_runtime_median_s",
        "worker_processing_runtime_mean_s",
        "worker_processing_runtime_q75_s",
        "worker_processing_runtime_max_s",
        "worker_processing_runtime_std_s",
        "worker_processing_runtime_n_samples",
        "n_preprocessing_tasks",
        "n_processing_tasks",
    ]

    assert CSV_COLUMNS[: len(_PREVIOUS_EXTENDED_COLUMNS)] == _PREVIOUS_EXTENDED_COLUMNS
    assert WORKER_RUNTIME_COLUMNS + TASK_COUNT_COLUMNS == expected_new_columns
    assert CSV_COLUMNS[len(_PREVIOUS_EXTENDED_COLUMNS) :] == expected_new_columns


def test_legacy_results_schema_is_unchanged_and_never_selected():
    header = LEGACY_RESULTS_PATH.read_text(encoding="utf-8").splitlines()[0]
    benchmark = RunAnalysisBenchmark(LEGACY_RESULTS_PATH, auto_finalize=False)

    assert next(csv.reader([header])) == LEGACY_CSV_COLUMNS
    assert DEFAULT_RESULTS_PATH.name == "benchmark_results_extended.csv"
    assert DEFAULT_RESULTS_PATH != LEGACY_RESULTS_PATH
    assert benchmark.results_path == DEFAULT_RESULTS_PATH
