#!/usr/bin/env python3

import argparse
import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


'''''
example usage:
    only one histogram:
/users/ldiazlie/.conda/envs/coffea-multicell-env/bin/python \
    compare_histeft_pkl.py \
    /users/ldiazlie/topeft/histos/plotsTopEFT.pkl.gz \
    /users/ldiazlie/topeft/histos/plotsTopEFT_multicell.pkl.gz \
    --histogram njets_sumw2 \
    --show-content

    all pkl file: 
    /users/ldiazlie/.conda/envs/coffea-multicell-env/bin/python \
    compare_histeft_pkl.py \
    /users/ldiazlie/topeft/histos/plotsTopEFT.pkl.gz \
    /users/ldiazlie/topeft/histos/plotsTopEFT_multicell.pkl.gz \
    --all \ 
    --show-content
'''''

@dataclass
class ContentSummary:
    total_values: int
    nonzero_values: int
    nonzero_fraction: float
    categories_with_content: int
    total_categories: int
    minimum: float
    maximum: float
    total_sum: float
    absolute_sum: float
    all_zeros: bool
    all_finite: bool


@dataclass
class ComparisonResult:
    histogram: str
    passed: bool
    compared_categories: int
    shape_mismatches: int
    value_mismatches: int
    missing_in_standard: int
    missing_in_multicell: int
    global_max_abs_diff: float
    global_max_rel_diff: float
    standard_summary: ContentSummary | None = None
    multicell_summary: ContentSummary | None = None
    message: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare HistEFT raw coefficients directly between "
            "Standard and MultiCell PKL files."
        )
    )

    parser.add_argument(
        "standard_pkl",
        type=Path,
        help="Standard HistEFT .pkl.gz file",
    )

    parser.add_argument(
        "multicell_pkl",
        type=Path,
        help="MultiCell HistEFT .pkl.gz file",
    )

    selection = parser.add_mutually_exclusive_group(required=True)

    selection.add_argument(
        "--histogram",
        "-H",
        help="Compare one histogram, for example njets or njets_sumw2",
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help="Compare all compatible histogram keys common to both PKLs",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance (default: 1e-10)",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance (default: 1e-12)",
    )

    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Print detailed content statistics for every histogram",
    )

    parser.add_argument(
        "--show-mismatches",
        action="store_true",
        help="Print individual category mismatches",
    )

    parser.add_argument(
        "--allow-all-zero",
        action="store_true",
        help=(
            "Do not fail when a histogram contains only zeros. "
            "By default, all-zero histograms are treated as failures."
        ),
    )

    return parser.parse_args()


def load_pkl(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with gzip.open(path, "rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, got {type(data).__name__}"
        )

    return data


def is_supported_histogram(obj: Any) -> bool:
    """
    Return True for histogram-like objects supported by this comparator.

    Required interface:
      - .view(flow=True, as_dict=True)
      - .axes
      - _use_multicell attribute or a quadratic_term axis
    """
    if not hasattr(obj, "view"):
        return False

    if not hasattr(obj, "axes"):
        return False

    try:
        raw = obj.view(flow=True, as_dict=True)
    except (TypeError, AttributeError, ValueError):
        return False

    return isinstance(raw, dict)


def get_axis_names(histogram: Any) -> list[str]:
    return [
        axis.name
        for axis in histogram.axes
    ]


def get_storage_type(histogram: Any) -> Any:
    """
    Try to obtain the dense histogram storage type without making it
    a requirement for the comparison.
    """
    dense_hists = getattr(histogram, "_dense_hists", None)

    if not dense_hists:
        return None

    try:
        first_dense = next(iter(dense_hists.values()))
    except StopIteration:
        return None

    return getattr(first_dense, "storage_type", None)


def normalize_sparse_key(key: Any) -> Any:
    """
    Convert sparse keys into a stable, hashable representation.
    """
    if isinstance(key, tuple):
        return key

    try:
        return tuple(key)
    except TypeError:
        return key


def normalize_raw_array(histogram: Any, array: Any) -> np.ndarray:
    """
    Normalize Standard and MultiCell storage to a common layout:

        (... dense bins including flow, EFT coefficients)

    Standard HistEFT:
        (... dense bins, quadratic_term including flow)
        -> remove the quadratic_term underflow and overflow

    MultiCell HistEFT:
        (EFT coefficients, ... dense bins)
        -> move the coefficient axis to the final position
    """
    values = np.asarray(array)

    use_multicell = bool(
        getattr(histogram, "_use_multicell", False)
    )

    if use_multicell:
        if values.ndim == 0:
            raise ValueError(
                "MultiCell raw array has no coefficient axis."
            )

        return np.moveaxis(values, 0, -1)

    axis_names = get_axis_names(histogram)

    if "quadratic_term" not in axis_names:
        raise ValueError(
            "Standard histogram does not contain a quadratic_term axis."
        )

    if values.ndim == 0 or values.shape[-1] < 3:
        raise ValueError(
            "Standard raw array does not contain quadratic_term flow bins."
        )

    return values[..., 1:-1]


def extract_raw_coefficients(
    histogram: Any,
) -> dict[Any, np.ndarray]:
    raw = histogram.view(
        flow=True,
        as_dict=True,
    )

    output = {}

    for sparse_key, raw_array in raw.items():
        key = normalize_sparse_key(sparse_key)

        if key in output:
            raise RuntimeError(
                f"Duplicate sparse key after normalization: {key}"
            )

        output[key] = normalize_raw_array(
            histogram,
            raw_array,
        )

    return output


def summarize_content(
    raw_coefficients: dict[Any, np.ndarray],
) -> ContentSummary:
    arrays = [
        np.asarray(array)
        for array in raw_coefficients.values()
    ]

    if not arrays:
        return ContentSummary(
            total_values=0,
            nonzero_values=0,
            nonzero_fraction=0.0,
            categories_with_content=0,
            total_categories=0,
            minimum=float("nan"),
            maximum=float("nan"),
            total_sum=0.0,
            absolute_sum=0.0,
            all_zeros=True,
            all_finite=True,
        )

    flattened = np.concatenate(
        [array.ravel() for array in arrays]
    )

    total_values = int(flattened.size)
    nonzero_values = int(np.count_nonzero(flattened))

    categories_with_content = sum(
        bool(np.any(array != 0))
        for array in arrays
    )

    return ContentSummary(
        total_values=total_values,
        nonzero_values=nonzero_values,
        nonzero_fraction=(
            nonzero_values / total_values
            if total_values
            else 0.0
        ),
        categories_with_content=categories_with_content,
        total_categories=len(arrays),
        minimum=(
            float(np.min(flattened))
            if flattened.size
            else float("nan")
        ),
        maximum=(
            float(np.max(flattened))
            if flattened.size
            else float("nan")
        ),
        total_sum=float(np.sum(flattened)),
        absolute_sum=float(np.sum(np.abs(flattened))),
        all_zeros=bool(np.all(flattened == 0)),
        all_finite=bool(np.all(np.isfinite(flattened))),
    )


def maximum_differences(
    array_a: np.ndarray,
    array_b: np.ndarray,
) -> tuple[float, float]:
    difference = array_a - array_b

    max_abs = (
        float(np.max(np.abs(difference)))
        if difference.size
        else 0.0
    )

    denominator = np.maximum(
        np.abs(array_a),
        np.abs(array_b),
    )

    nonzero = denominator > 0

    max_rel = (
        float(
            np.max(
                np.abs(difference[nonzero])
                / denominator[nonzero]
            )
        )
        if np.any(nonzero)
        else 0.0
    )

    return max_abs, max_rel


def print_content_summary(
    label: str,
    summary: ContentSummary,
):
    print(f"\n{label} content summary")
    print("-" * 58)
    print("Total values:             ", summary.total_values)
    print("Nonzero values:           ", summary.nonzero_values)
    print("Nonzero fraction:         ", summary.nonzero_fraction)
    print(
        "Categories with content:  ",
        summary.categories_with_content,
    )
    print("Total categories:         ", summary.total_categories)
    print("Minimum value:            ", summary.minimum)
    print("Maximum value:            ", summary.maximum)
    print("Sum of values:            ", summary.total_sum)
    print("Sum of absolute values:   ", summary.absolute_sum)
    print("All zeros:                ", summary.all_zeros)
    print("All finite:               ", summary.all_finite)


def compare_histogram(
    histogram_name: str,
    standard_histogram: Any,
    multicell_histogram: Any,
    rtol: float,
    atol: float,
    allow_all_zero: bool,
    show_content: bool,
    show_mismatches: bool,
) -> ComparisonResult:
    try:
        standard_raw = extract_raw_coefficients(
            standard_histogram
        )
        multicell_raw = extract_raw_coefficients(
            multicell_histogram
        )
    except Exception as error:
        return ComparisonResult(
            histogram=histogram_name,
            passed=False,
            compared_categories=0,
            shape_mismatches=0,
            value_mismatches=0,
            missing_in_standard=0,
            missing_in_multicell=0,
            global_max_abs_diff=0.0,
            global_max_rel_diff=0.0,
            message=f"Extraction failed: {error}",
        )

    standard_summary = summarize_content(standard_raw)
    multicell_summary = summarize_content(multicell_raw)

    standard_keys = set(standard_raw)
    multicell_keys = set(multicell_raw)

    common_keys = sorted(
        standard_keys & multicell_keys,
        key=str,
    )

    only_standard = sorted(
        standard_keys - multicell_keys,
        key=str,
    )

    only_multicell = sorted(
        multicell_keys - standard_keys,
        key=str,
    )

    shape_mismatches = 0
    value_mismatches = 0
    compared_categories = 0

    global_max_abs = 0.0
    global_max_rel = 0.0

    passed = True
    messages = []

    if only_standard:
        passed = False
        messages.append(
            f"{len(only_standard)} categories exist only in Standard"
        )

    if only_multicell:
        passed = False
        messages.append(
            f"{len(only_multicell)} categories exist only in MultiCell"
        )

    for key in common_keys:
        standard_array = standard_raw[key]
        multicell_array = multicell_raw[key]

        if standard_array.shape != multicell_array.shape:
            shape_mismatches += 1
            passed = False

            if show_mismatches:
                print(
                    f"\n[{histogram_name}] "
                    f"Shape mismatch for category {key}"
                )
                print("  Standard: ", standard_array.shape)
                print("  MultiCell:", multicell_array.shape)

            continue

        compared_categories += 1

        max_abs, max_rel = maximum_differences(
            standard_array,
            multicell_array,
        )

        global_max_abs = max(
            global_max_abs,
            max_abs,
        )

        global_max_rel = max(
            global_max_rel,
            max_rel,
        )

        category_close = np.allclose(
            standard_array,
            multicell_array,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )

        if not category_close:
            value_mismatches += 1
            passed = False

            if show_mismatches:
                print(
                    f"\n[{histogram_name}] "
                    f"Value mismatch for category {key}"
                )
                print("  Shape:       ", standard_array.shape)
                print("  Max abs diff:", max_abs)
                print("  Max rel diff:", max_rel)

    if not standard_summary.all_finite:
        passed = False
        messages.append("Standard contains non-finite values")

    if not multicell_summary.all_finite:
        passed = False
        messages.append("MultiCell contains non-finite values")

    # ------------------------------------------------------------------
# Handle histograms with no content
# ------------------------------------------------------------------

    both_all_zero = (
        standard_summary.all_zeros
        and multicell_summary.all_zeros
    )

    only_standard_zero = (
        standard_summary.all_zeros
        and not multicell_summary.all_zeros
    )

    only_multicell_zero = (
        multicell_summary.all_zeros
        and not standard_summary.all_zeros
    )

    if not allow_all_zero:

        if both_all_zero:
            # Both implementations produced an empty histogram.
            # This is considered a successful comparison.
            messages.append(
                "Standard and MultiCell contain only zeros"
            )

        elif only_standard_zero:
            passed = False
            messages.append(
                "Standard contains only zeros while MultiCell contains nonzero values"
            )

        elif only_multicell_zero:
            passed = False
            messages.append(
                "MultiCell contains only zeros while Standard contains nonzero values"
            )
    if shape_mismatches:
        messages.append(
            f"{shape_mismatches} shape mismatches"
        )

    if value_mismatches:
        messages.append(
            f"{value_mismatches} value mismatches"
        )

    if show_content:
        print_content_summary(
            f"{histogram_name} — Standard",
            standard_summary,
        )

        print_content_summary(
            f"{histogram_name} — MultiCell",
            multicell_summary,
        )

    return ComparisonResult(
        histogram=histogram_name,
        passed=passed,
        compared_categories=compared_categories,
        shape_mismatches=shape_mismatches,
        value_mismatches=value_mismatches,
        missing_in_standard=len(only_multicell),
        missing_in_multicell=len(only_standard),
        global_max_abs_diff=global_max_abs,
        global_max_rel_diff=global_max_rel,
        standard_summary=standard_summary,
        multicell_summary=multicell_summary,
        message="; ".join(messages),
    )


def print_single_result(
    result: ComparisonResult,
    standard_histogram: Any,
    multicell_histogram: Any,
    standard_pkl: Path,
    multicell_pkl: Path,
    rtol: float,
    atol: float,
):
    print("\n" + "=" * 78)
    print("HistEFT Standard vs MultiCell raw comparison")
    print("=" * 78)
    print("Standard PKL: ", standard_pkl)
    print("MultiCell PKL:", multicell_pkl)
    print("Histogram:    ", result.histogram)
    print("rtol:         ", rtol)
    print("atol:         ", atol)
    print()

    print(
        "Standard storage: ",
        get_storage_type(standard_histogram),
    )
    print(
        "MultiCell storage:",
        get_storage_type(multicell_histogram),
    )
    print(
        "Standard mode:    ",
        getattr(standard_histogram, "_use_multicell", None),
    )
    print(
        "MultiCell mode:   ",
        getattr(multicell_histogram, "_use_multicell", None),
    )

    if (
        result.standard_summary is not None
        and result.multicell_summary is not None
    ):
        print()
        print(
            "Categories in Standard: ",
            result.standard_summary.total_categories,
        )
        print(
            "Categories in MultiCell:",
            result.multicell_summary.total_categories,
        )
        print(
            "Same category count:    ",
            result.standard_summary.total_categories
            == result.multicell_summary.total_categories,
        )

    print()
    print("-" * 78)
    print("Comparison summary")
    print("-" * 78)
    print("Compared categories: ", result.compared_categories)
    print("Shape mismatches:    ", result.shape_mismatches)
    print("Value mismatches:    ", result.value_mismatches)
    print("Missing in Standard: ", result.missing_in_standard)
    print("Missing in MultiCell:", result.missing_in_multicell)
    print("Global max abs diff: ", result.global_max_abs_diff)
    print("Global max rel diff: ", result.global_max_rel_diff)
    print("All close:           ", result.passed)

    if result.message:
        print("Notes:               ", result.message)

    print("=" * 78)


def print_global_summary(
    results: list[ComparisonResult],
    skipped: list[tuple[str, str]],
):
    passed = [
        result
        for result in results
        if result.passed
    ]

    failed = [
        result
        for result in results
        if not result.passed
    ]

    print("\n" + "=" * 110)
    print("Global PKL comparison summary")
    print("=" * 110)

    header = (
        f"{'Histogram':<32}"
        f"{'Status':<10}"
        f"{'Categories':>12}"
        f"{'Shape':>9}"
        f"{'Values':>9}"
        f"{'Max abs':>16}"
        f"{'Max rel':>16}"
    )

    print(header)
    print("-" * 110)

    for result in results:
        status = "PASS" if result.passed else "FAIL"

        print(
            f"{result.histogram:<32}"
            f"{status:<10}"
            f"{result.compared_categories:>12}"
            f"{result.shape_mismatches:>9}"
            f"{result.value_mismatches:>9}"
            f"{result.global_max_abs_diff:>16.6g}"
            f"{result.global_max_rel_diff:>16.6g}"
        )

        if result.message:
            print(f"  ↳ {result.message}")

    print("-" * 110)
    print("Compared histograms:", len(results))
    print("Passed:             ", len(passed))
    print("Failed:             ", len(failed))
    print("Skipped:            ", len(skipped))

    if skipped:
        print("\nSkipped objects:")

        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    print("=" * 110)


def main():
    args = parse_args()

    standard_data = load_pkl(args.standard_pkl)
    multicell_data = load_pkl(args.multicell_pkl)

    if args.histogram:
        histogram_names = [args.histogram]
    else:
        standard_keys = set(standard_data)
        multicell_keys = set(multicell_data)

        only_standard = sorted(
            standard_keys - multicell_keys
        )

        only_multicell = sorted(
            multicell_keys - standard_keys
        )

        if only_standard:
            print("Keys only in Standard PKL:")
            for key in only_standard:
                print(" ", key)

        if only_multicell:
            print("Keys only in MultiCell PKL:")
            for key in only_multicell:
                print(" ", key)

        histogram_names = sorted(
            standard_keys & multicell_keys
        )

    results = []
    skipped = []

    for histogram_name in histogram_names:
        if histogram_name not in standard_data:
            results.append(
                ComparisonResult(
                    histogram=histogram_name,
                    passed=False,
                    compared_categories=0,
                    shape_mismatches=0,
                    value_mismatches=0,
                    missing_in_standard=1,
                    missing_in_multicell=0,
                    global_max_abs_diff=0.0,
                    global_max_rel_diff=0.0,
                    message="Histogram not found in Standard PKL",
                )
            )
            continue

        if histogram_name not in multicell_data:
            results.append(
                ComparisonResult(
                    histogram=histogram_name,
                    passed=False,
                    compared_categories=0,
                    shape_mismatches=0,
                    value_mismatches=0,
                    missing_in_standard=0,
                    missing_in_multicell=1,
                    global_max_abs_diff=0.0,
                    global_max_rel_diff=0.0,
                    message="Histogram not found in MultiCell PKL",
                )
            )
            continue

        standard_histogram = standard_data[histogram_name]
        multicell_histogram = multicell_data[histogram_name]

        if not is_supported_histogram(standard_histogram):
            skipped.append(
                (
                    histogram_name,
                    "Standard object is not a supported histogram",
                )
            )
            continue

        if not is_supported_histogram(multicell_histogram):
            skipped.append(
                (
                    histogram_name,
                    "MultiCell object is not a supported histogram",
                )
            )
            continue

        result = compare_histogram(
            histogram_name=histogram_name,
            standard_histogram=standard_histogram,
            multicell_histogram=multicell_histogram,
            rtol=args.rtol,
            atol=args.atol,
            allow_all_zero=args.allow_all_zero,
            show_content=args.show_content,
            show_mismatches=args.show_mismatches,
        )

        results.append(result)

        if args.histogram:
            print_single_result(
                result=result,
                standard_histogram=standard_histogram,
                multicell_histogram=multicell_histogram,
                standard_pkl=args.standard_pkl,
                multicell_pkl=args.multicell_pkl,
                rtol=args.rtol,
                atol=args.atol,
            )

    if args.all:
        print_global_summary(
            results,
            skipped,
        )

    failed = [
        result
        for result in results
        if not result.passed
    ]

    if failed:
        raise SystemExit(1)

    if not results:
        print("No compatible histograms were compared.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()