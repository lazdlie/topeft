#!/usr/bin/env python3

"""
Performance benchmark for Standard and MultiCell HistEFT storage.

The benchmark compares:

- histogram construction time,
- fill time,
- EFT evaluation time,
- pickle serialization time,
- serialized size,
- gzip-compressed serialized size,
- maximum resident memory,
- scaling with event count and sparse-category count.

Each benchmark configuration runs in a fresh subprocess. This prevents
the memory usage of one storage implementation from contaminating the
measurement of the other implementation.

Example
-------
Run the default benchmark:

    python benchmark_histeft_multicell.py

Run selected event counts:

    python benchmark_histeft_multicell.py \
        --events 10000 50000 100000

Use fewer sparse categories:

    python benchmark_histeft_multicell.py \
        --processes 2 \
        --channels 2 \
        --systematics 3

Save results under a custom prefix:

    python benchmark_histeft_multicell.py \
        --output benchmark_results/run1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path
from typing import Any

import hist
import numpy as np

from topcoffea.modules.histEFT import HistEFT


# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------

WC_NAMES = ["ctG", "cQu8"]

# Two Wilson coefficients imply six quadratic polynomial terms:
#
# 0: SM
# 1: SM * ctG
# 2: ctG^2
# 3: SM * cQu8
# 4: ctG * cQu8
# 5: cQu8^2
N_QUAD_TERMS = 6

AVAILABLE_PROCESSES = [
    "ttH",
    "ttW",
    "ttZ",
    "tZq",
    "tHq",
    "tttt",
]

AVAILABLE_CHANNELS = [
    "2lss",
    "3l",
    "4l",
    "2los",
]

AVAILABLE_SYSTEMATICS = [
    "nominal",
    "JESUp",
    "JESDown",
    "btagUp",
    "btagDown",
]

N_BINS = 5
BIN_START = 0.0
BIN_STOP = 5.0

DEFAULT_EVENT_COUNTS = [
    10_000,
    50_000,
    100_000,
]

DEFAULT_WC_POINTS = [
    {},
    {"ctG": 0.0, "cQu8": 0.0},
    {"ctG": 1.0, "cQu8": 0.0},
    {"ctG": 0.0, "cQu8": 1.0},
    {"ctG": 1.0, "cQu8": 1.0},
    {"ctG": -0.5, "cQu8": 0.75},
]


# ----------------------------------------------------------------------
# Timing utilities
# ----------------------------------------------------------------------

def measure_call(function, *args, **kwargs):
    """
    Measure the wall-clock execution time of one function call.

    Returns
    -------
    tuple
        function result, elapsed seconds
    """
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return result, elapsed


def summarize_timings(values):
    """
    Return summary statistics for a sequence of timing measurements.
    """
    values = list(values)

    if not values:
        raise ValueError("Cannot summarize an empty timing sequence")

    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "stdev": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


# ----------------------------------------------------------------------
# Memory utilities
# ----------------------------------------------------------------------

def maximum_resident_memory_mb():
    """
    Return the maximum resident-set size of the current process.

    On Linux, ru_maxrss is reported in KiB.
    On macOS, it is reported in bytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_rss = usage.ru_maxrss

    if platform.system() == "Darwin":
        return maximum_rss / (1024.0**2)

    return maximum_rss / 1024.0


# ----------------------------------------------------------------------
# Toy-data generation
# ----------------------------------------------------------------------

def selected_categories(
    n_processes,
    n_channels,
    n_systematics,
):
    """
    Return the selected category labels and all their combinations.
    """
    if not 1 <= n_processes <= len(AVAILABLE_PROCESSES):
        raise ValueError(
            "n_processes must be between 1 and "
            f"{len(AVAILABLE_PROCESSES)}"
        )

    if not 1 <= n_channels <= len(AVAILABLE_CHANNELS):
        raise ValueError(
            "n_channels must be between 1 and "
            f"{len(AVAILABLE_CHANNELS)}"
        )

    if not 1 <= n_systematics <= len(AVAILABLE_SYSTEMATICS):
        raise ValueError(
            "n_systematics must be between 1 and "
            f"{len(AVAILABLE_SYSTEMATICS)}"
        )

    processes = AVAILABLE_PROCESSES[:n_processes]
    channels = AVAILABLE_CHANNELS[:n_channels]
    systematics = AVAILABLE_SYSTEMATICS[:n_systematics]

    combinations = list(
        product(
            processes,
            channels,
            systematics,
        )
    )

    return processes, channels, systematics, combinations


def generate_test_data(
    n_events,
    seed,
    processes,
    channels,
    systematics,
):
    """
    Generate a deterministic, balanced multi-category toy dataset.

    The category assignment guarantees that every possible sparse
    category appears at least once, provided that:

        n_events >= number of sparse categories
    """
    rng = np.random.default_rng(seed)

    categories = list(
        product(
            processes,
            channels,
            systematics,
        )
    )

    n_categories = len(categories)

    if n_events < n_categories:
        raise ValueError(
            "n_events must be at least as large as the number of "
            f"sparse categories ({n_categories})"
        )

    # Repeat the complete category list until enough event assignments
    # are available. This guarantees that every category is represented.
    repeated_categories = np.asarray(
        categories
        * (
            (n_events + n_categories - 1)
            // n_categories
        ),
        dtype=object,
    )[:n_events]

    rng.shuffle(repeated_categories)

    process_values = repeated_categories[:, 0].astype(str)
    channel_values = repeated_categories[:, 1].astype(str)
    systematic_values = repeated_categories[:, 2].astype(str)

    # Include underflow and overflow deliberately.
    njets = rng.uniform(
        -1.0,
        6.0,
        size=n_events,
    )

    weights = rng.uniform(
        0.5,
        1.5,
        size=n_events,
    )

    eft_coeff = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_events, N_QUAD_TERMS),
    )

    # Keep the SM coefficient positive and nonzero.
    eft_coeff[:, 0] = rng.uniform(
        0.5,
        2.0,
        size=n_events,
    )

    return {
        "process": process_values,
        "channel": channel_values,
        "systematic": systematic_values,
        "njets": njets,
        "weight": weights,
        "eft_coeff": eft_coeff,
    }


def category_combinations(data):
    """
    Return sorted sparse-category combinations present in the dataset.
    """
    return sorted(
        set(
            zip(
                data["process"],
                data["channel"],
                data["systematic"],
            )
        )
    )


def category_mask(data, category):
    """
    Construct the event mask for one sparse category.
    """
    process_name, channel_name, systematic_name = category

    return (
        (data["process"] == process_name)
        & (data["channel"] == channel_name)
        & (data["systematic"] == systematic_name)
    )


# ----------------------------------------------------------------------
# Histogram construction and filling
# ----------------------------------------------------------------------

def build_histogram(
    use_multicell,
    processes,
    channels,
    systematics,
):
    """
    Construct either Standard or MultiCell HistEFT.

    Fixed categories are used instead of growth axes so that both
    implementations receive the same known categorical structure.
    """
    return HistEFT(
        hist.axis.StrCategory(
            processes,
            name="process",
        ),
        hist.axis.StrCategory(
            channels,
            name="channel",
        ),
        hist.axis.StrCategory(
            systematics,
            name="systematic",
        ),
        hist.axis.Regular(
            N_BINS,
            BIN_START,
            BIN_STOP,
            name="njets",
        ),
        wc_names=WC_NAMES,
        use_multicell=use_multicell,
    )


def fill_histogram(histogram, data):
    """
    Fill one sparse category at a time.

    This avoids relying on vectorized categorical-string filling and
    mirrors the internal SparseHist organization.
    """
    for category in category_combinations(data):
        mask = category_mask(data, category)

        process_name, channel_name, systematic_name = category

        histogram.fill(
            process=process_name,
            channel=channel_name,
            systematic=systematic_name,
            njets=data["njets"][mask],
            weight=data["weight"][mask],
            eft_coeff=data["eft_coeff"][mask],
        )


# ----------------------------------------------------------------------
# Evaluation and serialization
# ----------------------------------------------------------------------

def evaluate_histogram(histogram, wc_points):
    """
    Evaluate the histogram at all requested Wilson-coefficient points.

    A small numerical checksum is returned so that Python cannot simply
    discard the results as unused.
    """
    checksum = 0.0

    for wc_point in wc_points:
        evaluated = histogram.eval(wc_point)

        for values in evaluated.values():
            checksum += float(
                np.sum(
                    np.asarray(values),
                    dtype=np.float64,
                )
            )

    return checksum


def serialize_histogram(histogram):
    """
    Serialize the complete HistEFT object with the highest pickle
    protocol.

    Returns
    -------
    bytes
        Serialized histogram payload.
    """
    return pickle.dumps(
        histogram,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def compress_payload(payload):
    """
    Compress an existing pickle payload with gzip.
    """
    return gzip.compress(
        payload,
        compresslevel=6,
    )


def raw_storage_bytes(histogram):
    """
    Estimate the number of bytes exposed by raw histogram views.

    This is not the complete Python-object memory footprint. It is the
    sum of the raw NumPy view sizes and provides a useful storage-level
    comparison.
    """
    raw = histogram.view(
        flow=True,
        as_dict=True,
    )

    return sum(
        np.asarray(values).nbytes
        for values in raw.values()
    )


# ----------------------------------------------------------------------
# Single fresh-process benchmark
# ----------------------------------------------------------------------

def run_worker_benchmark(args):
    """
    Run one benchmark configuration inside a fresh process.
    """
    use_multicell = args.mode == "multicell"

    (
        processes,
        channels,
        systematics,
        expected_categories,
    ) = selected_categories(
        n_processes=args.processes,
        n_channels=args.channels,
        n_systematics=args.systematics,
    )

    data, generation_time = measure_call(
        generate_test_data,
        n_events=args.worker_events,
        seed=args.seed,
        processes=processes,
        channels=channels,
        systematics=systematics,
    )

    histogram, build_time = measure_call(
        build_histogram,
        use_multicell=use_multicell,
        processes=processes,
        channels=channels,
        systematics=systematics,
    )

    _, fill_time = measure_call(
        fill_histogram,
        histogram,
        data,
    )

    actual_categories = len(
        histogram.view(
            flow=True,
            as_dict=True,
        )
    )

    if actual_categories != len(expected_categories):
        raise RuntimeError(
            "Unexpected sparse-category count after filling.\n"
            f"Expected: {len(expected_categories)}\n"
            f"Received: {actual_categories}"
        )

    # Warm-up evaluation before recording repeated measurements.
    warmup_checksum = evaluate_histogram(
        histogram,
        DEFAULT_WC_POINTS,
    )

    eval_timings = []
    eval_checksum = 0.0

    for _ in range(args.eval_repetitions):
        checksum, elapsed = measure_call(
            evaluate_histogram,
            histogram,
            DEFAULT_WC_POINTS,
        )

        eval_timings.append(elapsed)
        eval_checksum += checksum

    eval_summary = summarize_timings(
        eval_timings
    )

    pickle_payload, pickle_time = measure_call(
        serialize_histogram,
        histogram,
    )

    compressed_payload, gzip_time = measure_call(
        compress_payload,
        pickle_payload,
    )

    raw_bytes = raw_storage_bytes(
        histogram
    )

    result = {
        "mode": args.mode,
        "use_multicell": use_multicell,
        "events": args.worker_events,
        "processes": len(processes),
        "channels": len(channels),
        "systematics": len(systematics),
        "categories": actual_categories,
        "dense_regular_bins": N_BINS,
        "dense_bins_with_flow": N_BINS + 2,
        "eft_coefficients": N_QUAD_TERMS,
        "seed": args.seed,
        "generation_time_s": generation_time,
        "build_time_s": build_time,
        "fill_time_s": fill_time,
        "fill_rate_events_per_s": (
            args.worker_events / fill_time
        ),
        "eval_repetitions": args.eval_repetitions,
        "eval_time_mean_s": eval_summary["mean"],
        "eval_time_median_s": eval_summary["median"],
        "eval_time_min_s": eval_summary["minimum"],
        "eval_time_max_s": eval_summary["maximum"],
        "eval_time_stdev_s": eval_summary["stdev"],
        "pickle_time_s": pickle_time,
        "gzip_time_s": gzip_time,
        "raw_storage_bytes": raw_bytes,
        "raw_storage_mb": raw_bytes / (1024.0**2),
        "pickle_size_bytes": len(pickle_payload),
        "pickle_size_mb": len(pickle_payload) / (1024.0**2),
        "gzip_size_bytes": len(compressed_payload),
        "gzip_size_mb": len(compressed_payload) / (1024.0**2),
        "peak_rss_mb": maximum_resident_memory_mb(),
        "warmup_checksum": warmup_checksum,
        "evaluation_checksum": eval_checksum,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    print(
        json.dumps(
            result,
            sort_keys=True,
        )
    )


# ----------------------------------------------------------------------
# Parent benchmark controller
# ----------------------------------------------------------------------

def run_worker_subprocess(
    script_path,
    mode,
    n_events,
    args,
):
    """
    Run one benchmark worker and parse its JSON result.
    """
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        "--mode",
        mode,
        "--worker-events",
        str(n_events),
        "--processes",
        str(args.processes),
        "--channels",
        str(args.channels),
        "--systematics",
        str(args.systematics),
        "--seed",
        str(args.seed),
        "--eval-repetitions",
        str(args.eval_repetitions),
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark worker failed for mode={mode}, "
            f"events={n_events}.\n\n"
            f"STDOUT:\n{completed.stdout}\n\n"
            f"STDERR:\n{completed.stderr}"
        )

    output_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError(
            "Benchmark worker returned no output"
        )

    try:
        return json.loads(
            output_lines[-1]
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse benchmark worker output as JSON.\n\n"
            f"STDOUT:\n{completed.stdout}"
        ) from error


def calculate_relative_metrics(results):
    """
    Add Standard-versus-MultiCell ratios for matching configurations.

    Ratio convention:

        standard value / multicell value

    A value greater than one means MultiCell uses less time, storage,
    or memory for that metric.
    """
    grouped = {}

    for result in results:
        key = (
            result["events"],
            result["processes"],
            result["channels"],
            result["systematics"],
            result["categories"],
        )

        grouped.setdefault(
            key,
            {},
        )[result["mode"]] = result

    ratio_fields = [
        "build_time_s",
        "fill_time_s",
        "eval_time_median_s",
        "pickle_time_s",
        "raw_storage_bytes",
        "pickle_size_bytes",
        "gzip_size_bytes",
        "peak_rss_mb",
    ]

    comparisons = []

    for key, mode_results in sorted(grouped.items()):
        if not {
            "standard",
            "multicell",
        }.issubset(mode_results):
            continue

        standard = mode_results["standard"]
        multicell = mode_results["multicell"]

        comparison = {
            "events": key[0],
            "processes": key[1],
            "channels": key[2],
            "systematics": key[3],
            "categories": key[4],
        }

        for field in ratio_fields:
            standard_value = standard[field]
            multicell_value = multicell[field]

            ratio_name = (
                field.removesuffix("_s")
                .removesuffix("_bytes")
                .removesuffix("_mb")
                + "_standard_over_multicell"
            )

            comparison[ratio_name] = (
                standard_value / multicell_value
                if multicell_value != 0
                else float("inf")
            )

        comparisons.append(comparison)

    return comparisons


# ----------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------

def format_seconds(value):
    """
    Format timing values for terminal output.
    """
    if value < 1e-3:
        return f"{value * 1e6:9.2f} us"

    if value < 1.0:
        return f"{value * 1e3:9.2f} ms"

    return f"{value:9.3f} s "


def print_results_table(results):
    """
    Print the main benchmark results.
    """
    print("\nBenchmark results")
    print("=" * 128)

    header = (
        f"{'Mode':<11}"
        f"{'Events':>11}"
        f"{'Cats':>7}"
        f"{'Build':>14}"
        f"{'Fill':>14}"
        f"{'Eval median':>14}"
        f"{'Peak RSS':>13}"
        f"{'Raw MB':>11}"
        f"{'Pickle MB':>12}"
        f"{'Gzip MB':>11}"
    )

    print(header)
    print("-" * 128)

    for result in results:
        print(
            f"{result['mode']:<11}"
            f"{result['events']:>11,d}"
            f"{result['categories']:>7}"
            f"{format_seconds(result['build_time_s']):>14}"
            f"{format_seconds(result['fill_time_s']):>14}"
            f"{format_seconds(result['eval_time_median_s']):>14}"
            f"{result['peak_rss_mb']:>11.2f} MB"
            f"{result['raw_storage_mb']:>11.4f}"
            f"{result['pickle_size_mb']:>12.4f}"
            f"{result['gzip_size_mb']:>11.4f}"
        )

    print("=" * 128)


def print_comparison_table(comparisons):
    """
    Print Standard/MultiCell ratios.
    """
    print("\nStandard / MultiCell ratios")
    print("=" * 112)

    print(
        f"{'Events':>11}"
        f"{'Cats':>7}"
        f"{'Fill':>11}"
        f"{'Eval':>11}"
        f"{'Peak RSS':>13}"
        f"{'Raw size':>12}"
        f"{'Pickle':>11}"
        f"{'Gzip':>11}"
    )

    print("-" * 112)

    for result in comparisons:
        print(
            f"{result['events']:>11,d}"
            f"{result['categories']:>7}"
            f"{result['fill_time_standard_over_multicell']:>11.3f}"
            f"{result['eval_time_median_standard_over_multicell']:>11.3f}"
            f"{result['peak_rss_standard_over_multicell']:>13.3f}"
            f"{result['raw_storage_standard_over_multicell']:>12.3f}"
            f"{result['pickle_size_standard_over_multicell']:>11.3f}"
            f"{result['gzip_size_standard_over_multicell']:>11.3f}"
        )

    print("=" * 112)
    print(
        "Ratio > 1 means the MultiCell value is smaller or faster "
        "for that metric."
    )


def write_csv(path, rows):
    """
    Write a list of dictionaries to CSV.
    """
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """
    Write benchmark data to a formatted JSON file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            sort_keys=True,
        )


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def parse_arguments():
    """
    Parse parent-controller and internal worker arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Standard and MultiCell HistEFT performance."
        )
    )

    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=DEFAULT_EVENT_COUNTS,
        help=(
            "Event counts used by the parent benchmark. "
            "Default: 10000 50000 100000"
        ),
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=3,
        help="Number of process categories. Default: 3.",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of channel categories. Default: 3.",
    )

    parser.add_argument(
        "--systematics",
        type=int,
        default=3,
        help="Number of systematic categories. Default: 3.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--eval-repetitions",
        type=int,
        default=10,
        help=(
            "Number of timed repetitions for the full set of WC "
            "evaluations. Default: 10."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("histeft_multicell_benchmark"),
        help=(
            "Output prefix. The program creates .csv, .json, "
            "_ratios.csv, and _ratios.json files."
        ),
    )

    # Internal worker options.
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "standard",
            "multicell",
        ],
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--worker-events",
        type=int,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def validate_parent_arguments(args):
    """
    Validate arguments used by the parent controller.
    """
    if any(
        n_events <= 0
        for n_events in args.events
    ):
        raise ValueError(
            "All event counts must be positive"
        )

    if args.eval_repetitions <= 0:
        raise ValueError(
            "eval_repetitions must be positive"
        )

    _, _, _, categories = selected_categories(
        n_processes=args.processes,
        n_channels=args.channels,
        n_systematics=args.systematics,
    )

    minimum_events = len(categories)

    too_small = [
        n_events
        for n_events in args.events
        if n_events < minimum_events
    ]

    if too_small:
        raise ValueError(
            "Each event count must be at least the number of sparse "
            f"categories ({minimum_events}). Invalid values: {too_small}"
        )


def main():
    """
    Entry point for parent and worker modes.
    """
    args = parse_arguments()

    if args.worker:
        if args.mode is None:
            raise ValueError(
                "--mode is required in worker mode"
            )

        if args.worker_events is None:
            raise ValueError(
                "--worker-events is required in worker mode"
            )

        run_worker_benchmark(args)
        return

    validate_parent_arguments(args)

    script_path = Path(__file__).resolve()

    print("=" * 78)
    print("HistEFT Standard vs MultiCell benchmark")
    print("=" * 78)
    print(f"Python executable:     {sys.executable}")
    print(f"Python version:        {platform.python_version()}")
    print(f"Processes:             {args.processes}")
    print(f"Channels:              {args.channels}")
    print(f"Systematics:           {args.systematics}")

    category_count = (
        args.processes
        * args.channels
        * args.systematics
    )

    print(f"Sparse categories:     {category_count}")
    print(f"Event configurations:  {args.events}")
    print(f"Eval repetitions:      {args.eval_repetitions}")
    print("=" * 78)

    results = []

    # Alternate order for each event count so one mode is not always
    # systematically executed first.
    for configuration_index, n_events in enumerate(args.events):
        if configuration_index % 2 == 0:
            modes = [
                "standard",
                "multicell",
            ]
        else:
            modes = [
                "multicell",
                "standard",
            ]

        for mode in modes:
            print(
                f"Running mode={mode:<9} "
                f"events={n_events:>10,d} ...",
                end="",
                flush=True,
            )

            result = run_worker_subprocess(
                script_path=script_path,
                mode=mode,
                n_events=n_events,
                args=args,
            )

            results.append(result)

            print(
                f" done "
                f"(fill={result['fill_time_s']:.4f} s, "
                f"peak RSS={result['peak_rss_mb']:.2f} MB)"
            )

    results.sort(
        key=lambda row: (
            row["events"],
            row["mode"],
        )
    )

    comparisons = calculate_relative_metrics(
        results
    )

    print_results_table(results)
    print_comparison_table(comparisons)

    output_prefix = args.output

    results_csv_path = output_prefix.with_suffix(
        ".csv"
    )

    results_json_path = output_prefix.with_suffix(
        ".json"
    )

    ratios_csv_path = output_prefix.parent / (
        output_prefix.name
        + "_ratios.csv"
    )

    ratios_json_path = output_prefix.parent / (
        output_prefix.name
        + "_ratios.json"
    )

    write_csv(
        results_csv_path,
        results,
    )

    write_json(
        results_json_path,
        results,
    )

    write_csv(
        ratios_csv_path,
        comparisons,
    )

    write_json(
        ratios_json_path,
        comparisons,
    )

    print("\nOutput files")
    print("------------")
    print(results_csv_path)
    print(results_json_path)
    print(ratios_csv_path)
    print(ratios_json_path)


if __name__ == "__main__":
    main()