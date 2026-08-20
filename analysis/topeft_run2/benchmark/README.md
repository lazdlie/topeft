# Extended run-analysis benchmark

New benchmark runs write one row per completed run to
`benchmark_results_extended.csv`. The historical `benchmark_results.csv` is a
reproducibility artifact and is never appended to by the extended recorder.
`TOPEFT_BENCHMARK_RESULTS` may still select a different new output path, except
that selecting the historical file is redirected to the extended default.

The extended CSV starts with every historical column in its original order.
It then adds separate worker task distributions, driver RSS snapshots, the
direct local final-merge time, and successful task counts. Missing or
unavailable measurements are empty CSV cells; sample counts are zero. No
arrays or samples are stored in cells.

## Worker memory

Work Queue's `resources_measured.memory` value is recorded only when resource
monitoring is enabled, the value is finite and nonnegative, and Coffea reports
the task as successful. The aggregate `measured_memory_*` distribution retains
all such successful Work Queue task categories for backward compatibility,
including remote accumulation tasks. `worker_preprocessing_memory_*` contains
only tasks whose Coffea category is `preprocessing`.
`worker_processing_memory_*` contains only categories beginning with
`processing`; it does not contain preprocessing or remote accumulation tasks.
Each distribution reports min, linearly interpolated q25, median, arithmetic
mean, linearly interpolated q75, max, population standard deviation, and its
sample count.

## Worker runtime and task counts

Worker runtime comes directly from each completed Work Queue task's
`cmd_execution_time`, the same microsecond value Coffea reports in seconds by
dividing by `1e6`. The preprocessing and processing classification and
successful-task requirement are identical to worker memory. Negative,
nonnumeric, NaN, and infinite runtimes are excluded. Runtime distributions use
the same quantiles and population standard deviation as memory.

`n_preprocessing_tasks` and `n_processing_tasks` count successful observed
tasks in their respective categories, regardless of whether that task has a
valid runtime sample. Failed attempts are not counted; when a failed task is
retried, only a later successful completion contributes to the count. The
runtime sample count can therefore be smaller than the task count when runtime
metadata is missing or invalid.

## Driver memory

Each snapshot is one read of `/proc/self/status`. `rss_mb` is Linux `VmRSS`
(current resident set), and `peak_rss_mb` is `VmHWM` (the process peak so far),
converted from KiB to MiB. There is no sampling loop or background thread.

- `driver_start_*`: immediately after CLI parsing when the recorder is created.
- `driver_after_preprocessing_*`: after Coffea Work Queue `_preprocessing`
  returns. It is empty for executors without that clean boundary.
- `driver_after_runner_*`: after the Coffea runner returns successfully and its
  existing runtime is recorded.
- `driver_before_serialization_*`: immediately before starting the timed
  `gzip.open`/`cloudpickle.dump` block.
- `driver_after_serialization_*`: immediately after that timed block ends.
- `driver_after_final_merge_*`: immediately after Coffea's local
  `accumulate_result_files` call returns, before its result-file cleanup. It is
  empty when Coffea or the executor does not expose this boundary.
- `driver_final_peak_rss_mb`: `VmHWM` read while the completed CSV row is being
  built, after output metrics have been inspected and before the row is written.

## Timing semantics

`runner_runtime_s` remains the existing `run_analysis.py` wall clock around the
Coffea runner workflow and is the primary comparison metric.
`serialization_time_s` remains wall time around only `gzip.open(...)` and
`cloudpickle.dump(...)`. `preprocessing_time_s`, `submission_time_s`,
`processing_time_s`, and `accumulation_time_s` are Work Queue diagnostic
stage/milestone timings. They can represent overlapping activity and must not
be summed into an analysis runtime.

`final_merge_time_s` times the direct `accumulate_result_files(...)` call inside
Coffea 0.7.21's `_final_accumulation`, the block announced by “Merging with
local final accumulator...”. It excludes subsequent result-file cleanup and
status checks. The hook is omitted automatically if those Coffea internals are
not available. There is no separate environment-excluded wall-time metric:
Coffea does not expose a stable, clean boundary for it in this workflow, and
the recorder does not estimate one.

All hooks are diagnostic and best-effort. Recorder failures are suppressed so
they cannot fail a physics run, and neither accumulator contents nor processor
behavior are modified.
