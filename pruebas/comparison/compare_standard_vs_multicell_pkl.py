#!/usr/bin/env python3
"""Compare Standard HistEFT and MultiCell HistEFT pickle outputs."""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path
from typing import Any, Hashable

import numpy as np


def load_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with gzip.open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dictionary, got {type(obj)!r}")
    return obj


def axis_names(histogram: Any) -> list[str]:
    return [axis.name for axis in histogram.axes]


def normalize_sparse_key(key: Any) -> Hashable:
    if isinstance(key, tuple):
        return tuple(key)
    try:
        return tuple(key)
    except TypeError:
        return key


def category_labels(histogram: Any, index_key: Any) -> tuple[Any, ...]:
    index_tuple = normalize_sparse_key(index_key)
    if not isinstance(index_tuple, tuple):
        return (index_tuple,)

    categorical_axes = getattr(histogram, "_categorical_axes", ())
    labels: list[Any] = []
    for axis, index in zip(categorical_axes, index_tuple):
        try:
            labels.append(axis.value(index))
        except Exception:
            labels.append(index)
    if len(labels) < len(index_tuple):
        labels.extend(index_tuple[len(labels):])
    return tuple(labels)


def normalize_raw_coefficients(
    parent_histogram: Any,
    dense_histogram: Any,
) -> np.ndarray:
    """
    Return raw coefficients with canonical shape:

        (n_eft_terms, n_dense_bins_including_flow)

    For Standard HistEFT, remove the underflow and overflow bins belonging
    specifically to the quadratic_term axis.

    For MultiCell HistEFT, the coefficient dimension is stored inside each
    cell and therefore has exactly n_eft_terms entries.
    """
    view = np.asarray(
        dense_histogram.view(flow=True),
        dtype=np.float64,
    )

    quad_count = int(getattr(parent_histogram, "_quad_count"))
    names = axis_names(dense_histogram)

    if "quadratic_term" in names:
        coefficient_axis = names.index("quadratic_term")
        axis = dense_histogram.axes[coefficient_axis]

        start = 1 if axis.traits.underflow else 0
        stop = view.shape[coefficient_axis]

        if axis.traits.overflow:
            stop -= 1

        indices = np.arange(start, stop)

        view = np.take(
            view,
            indices=indices,
            axis=coefficient_axis,
        )

    else:
        candidates = [
            index
            for index, size in enumerate(view.shape)
            if size == quad_count
        ]

        if not candidates:
            raise ValueError(
                "Could not identify the MultiCell coefficient dimension: "
                f"expected length {quad_count}, got shape {view.shape}."
            )

        coefficient_axis = candidates[0]

    canonical = np.moveaxis(
        view,
        coefficient_axis,
        0,
    )

    if canonical.shape[0] != quad_count:
        raise ValueError(
            f"Coefficient dimension has length {canonical.shape[0]}, "
            f"expected {quad_count}."
        )

    return canonical.reshape(quad_count, -1)

def raw_category_map(histogram: Any) -> dict[tuple[Any, ...], np.ndarray]:
    dense_hists = getattr(histogram, "_dense_hists", None)
    if not isinstance(dense_hists, dict):
        raise TypeError("Histogram does not expose a _dense_hists dictionary")

    result: dict[tuple[Any, ...], np.ndarray] = {}
    for index_key, dense_histogram in dense_hists.items():
        result[category_labels(histogram, index_key)] = normalize_raw_coefficients(
            histogram, dense_histogram
        )
    return result


def evaluated_view_map(
    histogram: Any,
    point: dict[str, float],
) -> dict[Hashable, np.ndarray]:
    """
    Evaluate HistEFT and normalize the result into:

        sparse_category -> NumPy array

    HistEFT.eval() may return:
      - a dictionary whose values are NumPy arrays,
      - a dictionary whose values are histogram-like objects,
      - a histogram-like object,
      - or a NumPy array.
    """
    evaluated = histogram.eval(point)

    def to_numpy(value: Any) -> np.ndarray:
        # NumPy arrays also have a .view() method, but it is not hist.view().
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64)

        # Histogram-like objects generally support view(flow=True).
        if hasattr(value, "view"):
            try:
                return np.asarray(
                    value.view(flow=True),
                    dtype=np.float64,
                )
            except TypeError:
                # Fallback for objects whose view() does not accept flow.
                return np.asarray(
                    value.view(),
                    dtype=np.float64,
                )

        return np.asarray(value, dtype=np.float64)

    if isinstance(evaluated, dict):
        return {
            normalize_sparse_key(key): to_numpy(value)
            for key, value in evaluated.items()
        }

    return {
        ("__all__",): to_numpy(evaluated)
    }


def compare_array_maps(
    reference: dict[Hashable, np.ndarray],
    candidate: dict[Hashable, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, str]:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    missing = reference_keys - candidate_keys
    extra = candidate_keys - reference_keys
    if missing or extra:
        return False, f"category mismatch: missing={len(missing)}, extra={len(extra)}"

    global_max_abs = 0.0
    global_max_rel = 0.0
    worst_key: Hashable | None = None

    for key in sorted(reference_keys, key=repr):
        ref = reference[key]
        cand = candidate[key]
        if ref.shape != cand.shape:
            return False, f"shape mismatch for {key!r}: {ref.shape} != {cand.shape}"

        difference = np.abs(ref - cand)
        max_abs = float(np.max(difference)) if difference.size else 0.0
        denominator = np.maximum(np.abs(ref), atol)
        max_rel = float(np.max(difference / denominator)) if difference.size else 0.0

        if max_abs > global_max_abs:
            global_max_abs = max_abs
            global_max_rel = max_rel
            worst_key = key

        if not np.allclose(ref, cand, rtol=rtol, atol=atol, equal_nan=True):
            return (
                False,
                f"numerical mismatch for {key!r}: "
                f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}",
            )

    return (
        True,
        f"max_abs={global_max_abs:.6e}, max_rel={global_max_rel:.6e}, "
        f"worst_category={worst_key!r}",
    )


def wc_names(histogram: Any) -> list[str]:
    names = getattr(histogram, "_wc_names", {})
    if isinstance(names, dict):
        try:
            return [name for name, _ in sorted(names.items(), key=lambda item: item[1])]
        except Exception:
            return list(names)
    return list(names)


def default_evaluation_points(histogram: Any) -> list[tuple[str, dict[str, float]]]:
    names = wc_names(histogram)
    points: list[tuple[str, dict[str, float]]] = [("SM", {})]
    if not names:
        return points

    preferred = "ctG" if "ctG" in names else names[0]
    points.append((f"{preferred}=1", {preferred: 1.0}))
    points.append((f"{preferred}=-1", {preferred: -1.0}))

    selected = names[: min(3, len(names))]
    rng = np.random.default_rng(42)
    random_point = {
        name: float(value)
        for name, value in zip(selected, rng.uniform(-1.0, 1.0, len(selected)))
    }
    points.append(("deterministic random point", random_point))
    return points


def compare_histogram(
    name: str,
    standard: Any,
    multicell: Any,
    *,
    rtol: float,
    atol: float,
) -> bool:
    print(f"\nHistogram: {name}")
    print("-" * (11 + len(name)))
    print(f"  Standard axes : {axis_names(standard)}")
    print(f"  MultiCell axes: {axis_names(multicell)}")
    print(f"  MultiCell flag: {getattr(multicell, '_use_multicell', '<missing>')}")

    try:
        raw_ok, raw_detail = compare_array_maps(
            raw_category_map(standard),
            raw_category_map(multicell),
            rtol=rtol,
            atol=atol,
        )
    except Exception as error:
        raw_ok = False
        raw_detail = f"{type(error).__name__}: {error}"

    print(f"  Raw coefficients: {'PASS' if raw_ok else 'FAIL'}")
    print(f"    {raw_detail}")

    eval_ok = True
    for label, point in default_evaluation_points(standard):
        try:
            point_ok, detail = compare_array_maps(
                evaluated_view_map(standard, point),
                evaluated_view_map(multicell, point),
                rtol=rtol,
                atol=atol,
            )
        except Exception as error:
            point_ok = False
            detail = f"{type(error).__name__}: {error}"

        eval_ok &= point_ok
        print(f"  eval({label}): {'PASS' if point_ok else 'FAIL'}")
        print(f"    point={point}")
        print(f"    {detail}")

    return raw_ok and eval_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Standard and MultiCell TopEFT pickle outputs."
    )
    parser.add_argument(
        "--standard",
        type=Path,
        default=Path("/users/ldiazlie/topeft/histos/plotsTopEFT.pkl.gz"),
    )
    parser.add_argument(
        "--multicell",
        type=Path,
        default=Path("/users/ldiazlie/topeft/histos/plotsTopEFT_multicell.pkl.gz"),
    )
    parser.add_argument(
        "--histogram",
        action="append",
        dest="histograms",
        help="Histogram key to compare; may be repeated. Default: all common HistEFT keys.",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    return parser.parse_args()


def is_histeft_like(obj: Any) -> bool:
    return hasattr(obj, "_dense_hists") and hasattr(obj, "_quad_count") and hasattr(obj, "eval")


def main() -> int:
    args = parse_args()
    print(f"Loading Standard : {args.standard}")
    standard_output = load_pickle(args.standard)
    print(f"Loading MultiCell: {args.multicell}")
    multicell_output = load_pickle(args.multicell)

    common_keys = set(standard_output) & set(multicell_output)
    if args.histograms:
        selected = args.histograms
        missing = [name for name in selected if name not in common_keys]
        if missing:
            print(f"ERROR: keys not present in both files: {missing}", file=sys.stderr)
            return 2
    else:
        selected = sorted(
            key
            for key in common_keys
            if is_histeft_like(standard_output[key])
            and is_histeft_like(multicell_output[key])
        )

    if not selected:
        print("ERROR: no common HistEFT histograms found.", file=sys.stderr)
        return 2

    print(f"Histograms selected: {', '.join(selected)}")
    print(f"Tolerances: rtol={args.rtol:.3e}, atol={args.atol:.3e}")

    overall_ok = True
    for name in selected:
        overall_ok &= compare_histogram(
            name,
            standard_output[name],
            multicell_output[name],
            rtol=args.rtol,
            atol=args.atol,
        )

    print("\n" + "=" * 72)
    print(f"OVERALL RESULT: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 72)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())