#!/usr/bin/env python3

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WC_POINTS = [
    ("SM", {}),
    ("ctG=+1", {"ctG": 1.0}),
    ("ctG=-1", {"ctG": -1.0}),
    ("ctW=+0.5", {"ctW": 0.5}),
    ("ctG=+0.5,ctW=-0.3", {"ctG": 0.5, "ctW": -0.3}),
    ("mixed", {"ctG": 1.2, "ctW": -0.8, "cpt": 0.4}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluated yields for shared 1D HistEFT objects and raw "
            "contents for shared 2D SparseHist objects."
        )
    )
    parser.add_argument(
        "--legacy",
        required=True,
        type=Path,
        help="Legacy TopEFT PKL file.",
    )
    parser.add_argument(
        "--multicell",
        required=True,
        type=Path,
        help="MultiCell + embedded nominal-sumw2 PKL file.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="Relative numerical tolerance.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute numerical tolerance.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"PKL file not found: {path}")

    with gzip.open(path, "rb") as file:
        output = pickle.load(file)

    if not isinstance(output, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, got {type(output).__name__}."
        )

    return output


def is_histeft(obj: Any) -> bool:
    return (
        hasattr(obj, "yield_coefficients")
        and hasattr(obj, "eval")
        and hasattr(obj, "wc_names")
    )


def is_sparse_hist(obj: Any) -> bool:
    return hasattr(obj, "view") and hasattr(obj, "axes")


def common_sparse_keys(
    legacy_dict: dict,
    multicell_dict: dict,
) -> tuple[set, list[str]]:
    legacy_keys = set(legacy_dict)
    multicell_keys = set(multicell_dict)

    messages = []

    only_legacy = legacy_keys - multicell_keys
    only_multicell = multicell_keys - legacy_keys

    if only_legacy:
        messages.append(
            f"{len(only_legacy)} sparse categories exist only in legacy."
        )

    if only_multicell:
        messages.append(
            f"{len(only_multicell)} sparse categories exist only in MultiCell."
        )

    return legacy_keys & multicell_keys, messages


def numerical_metrics(
    legacy_array: np.ndarray,
    multicell_array: np.ndarray,
    *,
    atol: float,
) -> tuple[float, float, int]:
    difference = np.abs(legacy_array - multicell_array)

    if difference.size == 0:
        return 0.0, 0.0, 0

    max_abs = float(np.max(difference))

    denominator = np.maximum(
        np.maximum(
            np.abs(legacy_array),
            np.abs(multicell_array),
        ),
        atol,
    )

    max_rel = float(np.max(difference / denominator))
    nonzero = int(np.count_nonzero(np.abs(legacy_array) > atol))

    return max_abs, max_rel, nonzero


def compatible_wc_points(
    wc_names: list[str],
) -> list[tuple[str, dict[str, float]]]:
    available = set(wc_names)
    resolved = []

    for label, point in DEFAULT_WC_POINTS:
        missing = set(point) - available

        if missing:
            continue

        resolved.append((label, point))

    # Exercise all available WCs with a deterministic, small mixed point.
    if wc_names:
        all_wc_point = {
            name: 0.05 * ((index % 7) - 3)
            for index, name in enumerate(wc_names)
        }
        resolved.append(("deterministic-all-WCs", all_wc_point))

    return resolved


def compare_histeft(
    name: str,
    legacy_hist: Any,
    multicell_hist: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    result = {
        "name": name,
        "kind": "HistEFT",
        "passed": True,
        "messages": [],
        "arrays": 0,
        "nonzero_entries": 0,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "wc_points": 0,
    }

    legacy_wc_names = list(legacy_hist.wc_names)
    multicell_wc_names = list(multicell_hist.wc_names)

    if legacy_wc_names != multicell_wc_names:
        result["passed"] = False
        result["messages"].append(
            "WC names/order differ: "
            f"legacy={legacy_wc_names}, "
            f"multicell={multicell_wc_names}"
        )
        return result

    wc_points = compatible_wc_points(legacy_wc_names)

    if not wc_points:
        result["passed"] = False
        result["messages"].append("No compatible WC points were constructed.")
        return result

    for point_label, point in wc_points:
        legacy_eval = legacy_hist.eval(point)
        multicell_eval = multicell_hist.eval(point)

        comparison_keys, key_messages = common_sparse_keys(
            legacy_eval,
            multicell_eval,
        )

        if key_messages:
            result["passed"] = False
            result["messages"].extend(
                f"{point_label}: {message}"
                for message in key_messages
            )

        result["wc_points"] += 1

        for sparse_key in comparison_keys:
            legacy_array = np.asarray(legacy_eval[sparse_key])
            multicell_array = np.asarray(multicell_eval[sparse_key])

            if legacy_array.shape != multicell_array.shape:
                result["passed"] = False
                result["messages"].append(
                    f"{point_label}, {sparse_key}: shape mismatch "
                    f"legacy={legacy_array.shape}, "
                    f"multicell={multicell_array.shape}"
                )
                continue

            if not np.all(np.isfinite(legacy_array)):
                result["passed"] = False
                result["messages"].append(
                    f"{point_label}, {sparse_key}: "
                    "legacy contains non-finite values."
                )
                continue

            if not np.all(np.isfinite(multicell_array)):
                result["passed"] = False
                result["messages"].append(
                    f"{point_label}, {sparse_key}: "
                    "MultiCell contains non-finite values."
                )
                continue

            max_abs, max_rel, nonzero = numerical_metrics(
                legacy_array,
                multicell_array,
                atol=atol,
            )

            result["arrays"] += 1
            result["nonzero_entries"] += nonzero
            result["max_abs"] = max(result["max_abs"], max_abs)
            result["max_rel"] = max(result["max_rel"], max_rel)

            if not np.allclose(
                legacy_array,
                multicell_array,
                rtol=rtol,
                atol=atol,
            ):
                failing_mask = ~np.isclose(
                    legacy_array,
                    multicell_array,
                    rtol=rtol,
                    atol=atol,
                )

                result["passed"] = False
                result["messages"].append(
                    f"{point_label}, {sparse_key}: "
                    f"{np.count_nonzero(failing_mask)}/"
                    f"{legacy_array.size} bins differ; "
                    f"max_abs={max_abs:.6e}, "
                    f"max_rel={max_rel:.6e}"
                )

    return result


def compare_sparsehist_raw(
    name: str,
    legacy_hist: Any,
    multicell_hist: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    result = {
        "name": name,
        "kind": "SparseHist raw",
        "passed": True,
        "messages": [],
        "arrays": 0,
        "nonzero_entries": 0,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "wc_points": 0,
    }

    legacy_axes = list(legacy_hist.axes.name)
    multicell_axes = list(multicell_hist.axes.name)

    if legacy_axes != multicell_axes:
        result["passed"] = False
        result["messages"].append(
            "Axis names differ: "
            f"legacy={legacy_axes}, "
            f"multicell={multicell_axes}"
        )

    legacy_view = legacy_hist.view(flow=True, as_dict=True)
    multicell_view = multicell_hist.view(flow=True, as_dict=True)

    comparison_keys, key_messages = common_sparse_keys(
        legacy_view,
        multicell_view,
    )

    if key_messages:
        result["passed"] = False
        result["messages"].extend(key_messages)

    for sparse_key in comparison_keys:
        legacy_array = np.asarray(legacy_view[sparse_key])
        multicell_array = np.asarray(multicell_view[sparse_key])

        if legacy_array.shape != multicell_array.shape:
            result["passed"] = False
            result["messages"].append(
                f"{sparse_key}: shape mismatch "
                f"legacy={legacy_array.shape}, "
                f"multicell={multicell_array.shape}"
            )
            continue

        if not np.all(np.isfinite(legacy_array)):
            result["passed"] = False
            result["messages"].append(
                f"{sparse_key}: legacy contains non-finite values."
            )
            continue

        if not np.all(np.isfinite(multicell_array)):
            result["passed"] = False
            result["messages"].append(
                f"{sparse_key}: MultiCell contains non-finite values."
            )
            continue

        max_abs, max_rel, nonzero = numerical_metrics(
            legacy_array,
            multicell_array,
            atol=atol,
        )

        result["arrays"] += 1
        result["nonzero_entries"] += nonzero
        result["max_abs"] = max(result["max_abs"], max_abs)
        result["max_rel"] = max(result["max_rel"], max_rel)

        if not np.allclose(
            legacy_array,
            multicell_array,
            rtol=rtol,
            atol=atol,
        ):
            failing_mask = ~np.isclose(
                legacy_array,
                multicell_array,
                rtol=rtol,
                atol=atol,
            )

            result["passed"] = False
            result["messages"].append(
                f"{sparse_key}: "
                f"{np.count_nonzero(failing_mask)}/"
                f"{legacy_array.size} bins differ; "
                f"max_abs={max_abs:.6e}, "
                f"max_rel={max_rel:.6e}"
            )

    return result


def main() -> None:
    args = parse_args()

    legacy_output = load_pickle(args.legacy)
    multicell_output = load_pickle(args.multicell)

    print("=" * 88)
    print("EVALUATED 1D HISTEFT + RAW 2D SPARSEHIST COMPARISON")
    print("=" * 88)

    common_names = sorted(set(legacy_output) & set(multicell_output))

    results = []
    skipped_legacy_companions = []
    skipped_unknown = []

    for name in common_names:
        legacy_obj = legacy_output[name]
        multicell_obj = multicell_output[name]

        legacy_is_histeft = is_histeft(legacy_obj)
        multicell_is_histeft = is_histeft(multicell_obj)

        # Skip the historical 1D HistEFT sumw2 companions.
        if name.endswith("_sumw2") and (
            legacy_is_histeft or multicell_is_histeft
        ):
            skipped_legacy_companions.append(name)
            continue

        if legacy_is_histeft and multicell_is_histeft:
            result = compare_histeft(
                name,
                legacy_obj,
                multicell_obj,
                rtol=args.rtol,
                atol=args.atol,
            )
            results.append(result)
            continue

        # The remaining shared objects should be ordinary 2D SparseHist
        # objects, including their explicit _sumw2 companions.
        if (
            is_sparse_hist(legacy_obj)
            and is_sparse_hist(multicell_obj)
            and not legacy_is_histeft
            and not multicell_is_histeft
        ):
            result = compare_sparsehist_raw(
                name,
                legacy_obj,
                multicell_obj,
                rtol=args.rtol,
                atol=args.atol,
            )
            results.append(result)
            continue

        skipped_unknown.append(name)

    for result in results:
        has_comparison_content = (
            result["arrays"] > 0
            and result["nonzero_entries"] > 0
        )

        if not result["passed"]:
            status = "FAIL"
        elif not has_comparison_content:
            status = "SKIP"
        else:
            status = "PASS"

        content_label = (
            f"WC points={result['wc_points']}"
            if result["kind"] == "HistEFT"
            else "raw"
        )

        if status == "SKIP":
            print(
                f"{status:<4} "
                f"{result['name']:<42} "
                f"{result['kind']:<14} "
                f"{content_label:<12} "
                "(no populated/nonzero comparison content)"
            )
        else:
            print(
                f"{status:<4} "
                f"{result['name']:<42} "
                f"{result['kind']:<14} "
                f"{content_label:<12} "
                f"arrays={result['arrays']:<5} "
                f"nonzero={result['nonzero_entries']:<8} "
                f"max_abs={result['max_abs']:.3e} "
                f"max_rel={result['max_rel']:.3e}"
            )
    failed = [result for result in results if not result["passed"]]

    skipped = [
        result
        for result in results
        if result["passed"]
        and (
            result["arrays"] == 0
            or result["nonzero_entries"] == 0
        )
    ]

    passed_with_content = [
        result
        for result in results
        if result["passed"]
        and result["arrays"] > 0
        and result["nonzero_entries"] > 0
    ]
    empty = [
        result
        for result in results
        if result["arrays"] == 0 or result["nonzero_entries"] == 0
    ]

    histeft_results = [
        result for result in results if result["kind"] == "HistEFT"
    ]
    sparse_results = [
        result for result in results if result["kind"] == "SparseHist raw"
    ]

    total_arrays = sum(result["arrays"] for result in results)
    total_nonzero = sum(result["nonzero_entries"] for result in results)

    global_max_abs = max(
        (result["max_abs"] for result in results),
        default=0.0,
    )
    global_max_rel = max(
        (result["max_rel"] for result in results),
        default=0.0,
    )

    print("\n" + "=" * 88)
    print("GLOBAL SUMMARY")
    print("=" * 88)

    print(f"Shared keys inspected:                 {len(common_names)}")
    print(f"1D HistEFT objects evaluated:          {len(histeft_results)}")
    print(f"2D SparseHist objects compared raw:    {len(sparse_results)}")
    print(
        "Skipped legacy 1D sumw2 companions:  "
        f"{len(skipped_legacy_companions)}"
    )
    print(f"Skipped unknown objects:               {len(skipped_unknown)}")
    print(f"Objects passed with content:           {len(passed_with_content)}")
    print(f"Objects skipped as empty:              {len(skipped)}")
    print(f"Objects failed:                        {len(failed)}")
    print(f"Compared arrays:                       {total_arrays}")
    print(f"Nonzero entries exercised:             {total_nonzero}")
    print(f"Global maximum absolute difference:    {global_max_abs:.6e}")
    print(f"Global maximum relative difference:    {global_max_rel:.6e}")

    if skipped_legacy_companions:
        print("\nSkipped legacy HistEFT companions:")
        for name in skipped_legacy_companions:
            print(" ", name)

    if skipped_unknown:
        print("\nSkipped objects with unsupported types:")
        for name in skipped_unknown:
            print(
                f"  {name}: "
                f"legacy={type(legacy_output[name]).__name__}, "
                f"multicell={type(multicell_output[name]).__name__}"
            )

    if empty:
        print("\nObjects with no populated/nonzero comparison content:")
        for result in empty:
            print(
                f"  {result['name']} "
                f"({result['kind']}, arrays={result['arrays']}, "
                f"nonzero={result['nonzero_entries']})"
            )

    if failed:
        print("\n" + "=" * 88)
        print("FAILURE DETAILS")
        print("=" * 88)

        for result in failed:
            print(f"\n{result['name']} ({result['kind']}):")
            for message in result["messages"]:
                print("  -", message)

        raise AssertionError(
            f"{len(failed)} histogram object(s) failed comparison."
        )

    nonzero_histeft = sum(
        result["nonzero_entries"]
        for result in histeft_results
    )

    if nonzero_histeft == 0:
        raise AssertionError(
            "All evaluated HistEFT results were zero; "
            "the WC-point validation was not meaningful."
        )

    if sparse_results and all(
        result["nonzero_entries"] == 0
        for result in sparse_results
    ):
        print(
            "\nNOTE: all shared 2D SparseHist objects are empty for this sample."
        )

    print("\nPASS: evaluated HistEFT yields contain nonzero content.")
    print(
        "ALL COMPARABLE EVALUATED YIELDS AND 2D RAW HISTOGRAMS MATCH."
    )


if __name__ == "__main__":
    main()