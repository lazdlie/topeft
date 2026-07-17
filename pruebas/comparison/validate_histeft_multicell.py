#!/usr/bin/env python

"""
Validate the MultiCell implementation of HistEFT.

This script compares:

1. Standard HistEFT against an independent NumPy calculation.
2. MultiCell HistEFT against an independent NumPy calculation.
3. Standard HistEFT directly against MultiCell HistEFT.
4. eval() results at several Wilson-coefficient points.

The toy dataset contains multiple:

- processes,
- channels,
- systematic variations,
- dense bins,
- underflow and overflow events.

The generated data are deterministic because a fixed random seed is used.
"""

import argparse
from itertools import product

import hist
import numpy as np

from topcoffea.modules.histEFT import HistEFT


# ----------------------------------------------------------------------
# Histogram and toy-dataset configuration
# ----------------------------------------------------------------------

WC_NAMES = ["ctG", "cQu8"]

# For two Wilson coefficients:
#
# 0: SM
# 1: SM * ctG
# 2: ctG^2
# 3: SM * cQu8
# 4: ctG * cQu8
# 5: cQu8^2
N_QUAD_TERMS = 6

PROCESSES = ["ttH", "ttW", "ttZ"]
CHANNELS = ["2lss", "3l", "4l"]
SYSTEMATICS = ["nominal", "JESUp", "JESDown"]

N_BINS = 5
BIN_START = 0.0
BIN_STOP = 5.0

EXPECTED_CATEGORY_COUNT = (
    len(PROCESSES)
    * len(CHANNELS)
    * len(SYSTEMATICS)
)


# ----------------------------------------------------------------------
# Toy-data generation and validation
# ----------------------------------------------------------------------

def generate_test_data(n_events=10_000, seed=42):
    """
    Generate a deterministic toy dataset.

    Every event contains:

    - process,
    - channel,
    - systematic,
    - njets,
    - event weight,
    - complete EFT coefficient vector.

    The same arrays will be used to fill the standard and MultiCell
    histogram implementations.
    """
    if n_events <= 0:
        raise ValueError("n_events must be positive")

    rng = np.random.default_rng(seed)

    process_values = rng.choice(
        PROCESSES,
        size=n_events,
    )

    channel_values = rng.choice(
        CHANNELS,
        size=n_events,
    )

    systematic_values = rng.choice(
        SYSTEMATICS,
        size=n_events,
    )

    # Generate values outside [0, 5) deliberately, so underflow and
    # overflow are included in the validation.
    njets = rng.uniform(
        -1.0,
        6.0,
        size=n_events,
    )

    # Positive, nontrivial event weights.
    weights = rng.uniform(
        0.5,
        1.5,
        size=n_events,
    )

    # One complete EFT coefficient vector per event.
    eft_coeff = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_events, N_QUAD_TERMS),
    )

    # Keep the SM term positive and nonzero.
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


def all_possible_categories():
    """
    Return all possible categorical combinations.

    Each category is represented as:

        (process, channel, systematic)
    """
    return list(
        product(
            PROCESSES,
            CHANNELS,
            SYSTEMATICS,
        )
    )


def category_combinations(data):
    """
    Return the sorted categorical combinations present in the dataset.
    """
    combinations = set(
        zip(
            data["process"],
            data["channel"],
            data["systematic"],
        )
    )

    return sorted(combinations)


def category_mask(data, category):
    """
    Construct an event mask for one sparse-category combination.
    """
    process_name, channel_name, systematic_name = category

    return (
        (data["process"] == process_name)
        & (data["channel"] == channel_name)
        & (data["systematic"] == systematic_name)
    )


def validate_input_data(data, n_events):
    """
    Verify that the generated dataset has the expected structure and
    contains all cases needed by the validation.
    """
    required_keys = {
        "process",
        "channel",
        "systematic",
        "njets",
        "weight",
        "eft_coeff",
    }

    missing_keys = required_keys.difference(data)

    if missing_keys:
        raise AssertionError(
            f"Missing input fields: {sorted(missing_keys)}"
        )

    # Check array dimensions.
    assert data["process"].shape == (n_events,)
    assert data["channel"].shape == (n_events,)
    assert data["systematic"].shape == (n_events,)
    assert data["njets"].shape == (n_events,)
    assert data["weight"].shape == (n_events,)
    assert data["eft_coeff"].shape == (
        n_events,
        N_QUAD_TERMS,
    )

    # Check that numerical inputs are finite.
    assert np.all(np.isfinite(data["njets"]))
    assert np.all(np.isfinite(data["weight"]))
    assert np.all(np.isfinite(data["eft_coeff"]))

    # Check nontrivial event weights and SM coefficients.
    assert np.all(data["weight"] > 0)
    assert np.all(data["eft_coeff"][:, 0] > 0)

    # Require both positive and negative BSM coefficients.
    bsm_coefficients = data["eft_coeff"][:, 1:]

    assert np.any(bsm_coefficients > 0)
    assert np.any(bsm_coefficients < 0)

    # Check that all configured categories appear.
    assert set(np.unique(data["process"])) == set(PROCESSES)
    assert set(np.unique(data["channel"])) == set(CHANNELS)
    assert set(np.unique(data["systematic"])) == set(SYSTEMATICS)

    combinations = category_combinations(data)

    assert len(combinations) == EXPECTED_CATEGORY_COUNT, (
        "Not all sparse-category combinations are present.\n"
        f"Expected: {EXPECTED_CATEGORY_COUNT}\n"
        f"Found:    {len(combinations)}"
    )

    # Validate underflow, regular-range, and overflow populations.
    underflow_count = np.count_nonzero(
        data["njets"] < BIN_START
    )

    in_range_count = np.count_nonzero(
        (data["njets"] >= BIN_START)
        & (data["njets"] < BIN_STOP)
    )

    overflow_count = np.count_nonzero(
        data["njets"] >= BIN_STOP
    )

    assert underflow_count > 0
    assert in_range_count > 0
    assert overflow_count > 0

    assert (
        underflow_count
        + in_range_count
        + overflow_count
        == n_events
    )


def print_dataset_summary(data):
    """
    Print a summary of the generated numerical inputs.
    """
    njets = data["njets"]
    weights = data["weight"]
    eft_coeff = data["eft_coeff"]

    bin_edges_with_flow = np.array(
        [
            -np.inf,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            np.inf,
        ]
    )

    occupancy, _ = np.histogram(
        njets,
        bins=bin_edges_with_flow,
    )

    print("\nToy dataset summary")
    print("-------------------")
    print(f"Events:                  {len(njets)}")
    print(f"eft_coeff shape:         {eft_coeff.shape}")

    print(
        "Weight range:            "
        f"[{weights.min():.6f}, {weights.max():.6f}]"
    )

    print(f"Mean weight:             {weights.mean():.6f}")
    print(
        f"Mean SM coefficient:     "
        f"{eft_coeff[:, 0].mean():.6f}"
    )

    print(
        "BSM coefficient range:   "
        f"[{eft_coeff[:, 1:].min():.6f}, "
        f"{eft_coeff[:, 1:].max():.6f}]"
    )

    print("\nDense-bin occupancy")
    print("-------------------")
    print(f"Underflow:               {occupancy[0]}")

    for bin_index in range(N_BINS):
        print(
            f"Bin [{bin_index}, {bin_index + 1}):"
            f"{occupancy[bin_index + 1]:>12}"
        )

    print(f"Overflow:                {occupancy[-1]}")
    print(f"Occupancy total:         {occupancy.sum()}")


def print_category_summary(data):
    """
    Print the number of events in every sparse-category combination.
    """
    combinations = category_combinations(data)

    print("\nSparse-category occupancy")
    print("-------------------------")
    print(
        f"{'Process':<9}"
        f"{'Channel':<9}"
        f"{'Systematic':<12}"
        f"{'Events':>8}"
    )
    print("-" * 38)

    total_count = 0

    for category in combinations:
        mask = category_mask(data, category)
        count = np.count_nonzero(mask)
        total_count += count

        process_name, channel_name, systematic_name = category

        print(
            f"{process_name:<9}"
            f"{channel_name:<9}"
            f"{systematic_name:<12}"
            f"{count:>8}"
        )

    print("-" * 38)
    print(f"Present combinations:    {len(combinations)}")
    print(f"Possible combinations:   {EXPECTED_CATEGORY_COUNT}")
    print(f"Category event total:    {total_count}")

    assert len(combinations) == EXPECTED_CATEGORY_COUNT
    assert total_count == len(data["njets"])


# ----------------------------------------------------------------------
# Histogram construction and filling
# ----------------------------------------------------------------------

def build_hist(use_multicell):
    """
    Construct HistEFT in either standard or MultiCell mode.
    """
    return HistEFT(
        hist.axis.StrCategory(
            [],
            name="process",
            growth=True,
        ),
        hist.axis.StrCategory(
            [],
            name="channel",
            growth=True,
        ),
        hist.axis.StrCategory(
            [],
            name="systematic",
            growth=True,
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


def fill_hist(histogram, data):
    """
    Fill HistEFT one sparse-category combination at a time.

    This mirrors the SparseHist model, where each unique categorical
    combination corresponds to a separate internal dense histogram.
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
# Independent reference calculation
# ----------------------------------------------------------------------

def dense_bin_indices(values):
    """
    Convert dense-axis values to bin indices including flow bins.

    Returned indices:

        0              underflow
        1 ... N_BINS   regular bins
        N_BINS + 1     overflow
    """
    values = np.asarray(values)

    bin_width = (
        BIN_STOP - BIN_START
    ) / N_BINS

    indices = np.floor(
        (values - BIN_START) / bin_width
    ).astype(int) + 1

    indices[values < BIN_START] = 0
    indices[values >= BIN_STOP] = N_BINS + 1

    return indices


def calculate_expected_raw_coefficients(data):
    """
    Independently calculate accumulated EFT coefficients for every
    sparse category.

    Returns
    -------
    dict
        Dictionary with keys:

            (process, channel, systematic)

        Each value has shape:

            (dense bins including flow, quadratic terms)
    """
    expected = {}

    event_bin_indices = dense_bin_indices(
        data["njets"]
    )

    weighted_coefficients = (
        data["eft_coeff"]
        * data["weight"][:, np.newaxis]
    )

    for category in category_combinations(data):
        mask = category_mask(data, category)

        category_expected = np.zeros(
            (
                N_BINS + 2,
                N_QUAD_TERMS,
            ),
            dtype=float,
        )

        np.add.at(
            category_expected,
            event_bin_indices[mask],
            weighted_coefficients[mask],
        )

        expected[category] = category_expected

    return expected


def calculate_expected_evaluation(coefficients, wc_point):
    """
    Independently evaluate the EFT polynomial for two Wilson
    coefficients.

    Coefficient ordering:

        0: SM
        1: SM * ctG
        2: ctG^2
        3: SM * cQu8
        4: ctG * cQu8
        5: cQu8^2
    """
    ctg = wc_point.get("ctG", 0.0)
    cqu8 = wc_point.get("cQu8", 0.0)

    return (
        coefficients[:, 0]
        + coefficients[:, 1] * ctg
        + coefficients[:, 2] * ctg**2
        + coefficients[:, 3] * cqu8
        + coefficients[:, 4] * ctg * cqu8
        + coefficients[:, 5] * cqu8**2
    )


# ----------------------------------------------------------------------
# Histogram-output extraction
# ----------------------------------------------------------------------

def normalize_sparse_key(sparse_key):
    """
    Convert a SparseHist named-tuple key to a plain tuple.

    Example:

        SparseHistTuple(...,
            process="ttH",
            channel="2lss",
            systematic="nominal",
        )

    becomes:

        ("ttH", "2lss", "nominal")
    """
    return tuple(sparse_key)


def extract_all_raw_coefficients(histogram):
    """
    Extract all raw EFT coefficients in a common logical layout:

        (dense bins including flow, quadratic terms)

    Standard raw layout:

        (
            dense bins including flow,
            quadratic-term bins including flow,
        )

    MultiCell raw layout:

        (
            quadratic terms,
            dense bins including flow,
        )
    """
    raw = histogram.view(
        flow=True,
        as_dict=True,
    )

    output = {}

    for sparse_key, raw_values in raw.items():
        values = np.asarray(raw_values)

        if histogram._use_multicell:
            coefficients = values.T
        else:
            coefficients = values[:, 1:-1]

        expected_shape = (
            N_BINS + 2,
            N_QUAD_TERMS,
        )

        if coefficients.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected coefficient layout for {sparse_key}.\n"
                f"Expected shape: {expected_shape}\n"
                f"Received shape: {coefficients.shape}"
            )

        category = normalize_sparse_key(sparse_key)

        if category in output:
            raise RuntimeError(
                f"Duplicate sparse category found: {category}"
            )

        output[category] = coefficients

    return output


def extract_all_evaluations(histogram, wc_point):
    """
    Evaluate all sparse categories and normalize their keys.
    """
    evaluated = histogram.eval(wc_point)

    return {
        normalize_sparse_key(sparse_key): np.asarray(values)
        for sparse_key, values in evaluated.items()
    }


# ----------------------------------------------------------------------
# Correctness comparisons
# ----------------------------------------------------------------------

def compare_against_expected(
    histogram,
    expected_coefficients,
    label,
    rtol=1e-12,
    atol=1e-12,
):
    """
    Compare one HistEFT implementation against the independent NumPy
    calculation for every sparse category.
    """
    actual_coefficients = extract_all_raw_coefficients(
        histogram
    )

    expected_keys = set(expected_coefficients)
    actual_keys = set(actual_coefficients)

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        raise AssertionError(
            f"{label} sparse-category mismatch.\n"
            f"Missing categories:    {sorted(missing)}\n"
            f"Unexpected categories: {sorted(unexpected)}"
        )

    global_max_difference = 0.0

    for category in sorted(expected_coefficients):
        expected = expected_coefficients[category]
        actual = actual_coefficients[category]

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"{label} raw coefficient mismatch "
                f"for category {category}"
            ),
        )

        max_difference = np.max(
            np.abs(actual - expected)
        )

        global_max_difference = max(
            global_max_difference,
            max_difference,
        )

    print(
        f"[PASS] {label} raw coefficients match the independent "
        f"expectation for {len(expected_coefficients)} categories"
    )

    print(
        "       Maximum absolute difference: "
        f"{global_max_difference:.3e}"
    )


def compare_raw_coefficients(
    standard_hist,
    multicell_hist,
    rtol=1e-12,
    atol=1e-12,
):
    """
    Compare Standard and MultiCell coefficient arrays for every sparse
    category.
    """
    standard_coefficients = extract_all_raw_coefficients(
        standard_hist
    )

    multicell_coefficients = extract_all_raw_coefficients(
        multicell_hist
    )

    standard_keys = set(standard_coefficients)
    multicell_keys = set(multicell_coefficients)

    if standard_keys != multicell_keys:
        raise AssertionError(
            "Standard and MultiCell sparse-category keys differ.\n"
            f"Standard only:  {sorted(standard_keys - multicell_keys)}\n"
            f"MultiCell only: {sorted(multicell_keys - standard_keys)}"
        )

    global_max_difference = 0.0

    for category in sorted(standard_coefficients):
        standard_values = standard_coefficients[category]
        multicell_values = multicell_coefficients[category]

        np.testing.assert_allclose(
            standard_values,
            multicell_values,
            rtol=rtol,
            atol=atol,
            err_msg=(
                "Standard-versus-MultiCell coefficient mismatch "
                f"for category {category}"
            ),
        )

        max_difference = np.max(
            np.abs(
                standard_values
                - multicell_values
            )
        )

        global_max_difference = max(
            global_max_difference,
            max_difference,
        )

    print(
        "[PASS] Standard and MultiCell raw coefficients match "
        f"for {len(standard_coefficients)} categories"
    )

    print(
        "       Maximum absolute difference: "
        f"{global_max_difference:.3e}"
    )


def compare_evaluations(
    standard_hist,
    multicell_hist,
    expected_coefficients,
    rtol=1e-12,
    atol=1e-12,
):
    """
    Compare eval() against the independent mathematical calculation for
    every sparse category and several Wilson-coefficient points.
    """
    wc_points = [
        {},
        {"ctG": 0.0, "cQu8": 0.0},
        {"ctG": 1.0, "cQu8": 0.0},
        {"ctG": 0.0, "cQu8": 1.0},
        {"ctG": 1.0, "cQu8": 1.0},
        {"ctG": -0.5, "cQu8": 0.75},
    ]

    expected_keys = set(expected_coefficients)

    for wc_point in wc_points:
        standard_evaluations = extract_all_evaluations(
            standard_hist,
            wc_point,
        )

        multicell_evaluations = extract_all_evaluations(
            multicell_hist,
            wc_point,
        )

        if set(standard_evaluations) != expected_keys:
            raise AssertionError(
                "Standard eval() returned unexpected sparse categories "
                f"at WC point {wc_point}"
            )

        if set(multicell_evaluations) != expected_keys:
            raise AssertionError(
                "MultiCell eval() returned unexpected sparse categories "
                f"at WC point {wc_point}"
            )

        global_max_difference = 0.0

        for category in sorted(expected_coefficients):
            expected_evaluation = calculate_expected_evaluation(
                expected_coefficients[category],
                wc_point,
            )

            standard_values = standard_evaluations[category]
            multicell_values = multicell_evaluations[category]

            np.testing.assert_allclose(
                standard_values,
                expected_evaluation,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"Standard eval() mismatch for {category} "
                    f"at WC point {wc_point}"
                ),
            )

            np.testing.assert_allclose(
                multicell_values,
                expected_evaluation,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"MultiCell eval() mismatch for {category} "
                    f"at WC point {wc_point}"
                ),
            )

            np.testing.assert_allclose(
                standard_values,
                multicell_values,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    "Standard-versus-MultiCell eval() mismatch "
                    f"for {category} at WC point {wc_point}"
                ),
            )

            category_max_difference = max(
                np.max(
                    np.abs(
                        standard_values
                        - expected_evaluation
                    )
                ),
                np.max(
                    np.abs(
                        multicell_values
                        - expected_evaluation
                    )
                ),
            )

            global_max_difference = max(
                global_max_difference,
                category_max_difference,
            )

        print(
            f"[PASS] eval() matches the independent expectation "
            f"at {wc_point} for {len(expected_coefficients)} categories"
        )

        print(
            "       Maximum absolute difference: "
            f"{global_max_difference:.3e}"
        )


# ----------------------------------------------------------------------
# Human-readable histogram summaries
# ----------------------------------------------------------------------

def print_storage_summary(histogram, label):
    """
    Print storage, axes, number of sparse categories, and a representative
    raw-array shape.
    """
    raw = histogram.view(
        flow=True,
        as_dict=True,
    )

    if not raw:
        raise RuntimeError(
            f"{label} contains no sparse categories"
        )

    first_key = next(iter(raw))
    first_raw_values = np.asarray(raw[first_key])

    dense_histogram = next(
        iter(histogram._dense_hists.values())
    )

    print(f"\n{label}")
    print("-" * len(label))
    print("use_multicell:", histogram._use_multicell)
    print("axes:", [axis.name for axis in histogram.axes])
    print("storage:", dense_histogram._hist._storage_type)
    print("sparse categories:", len(raw))
    print("representative key:", normalize_sparse_key(first_key))
    print("representative raw shape:", first_raw_values.shape)


def print_expected_example(expected_coefficients):
    """
    Print one category as an example of the independent accumulated
    coefficient result.
    """
    example_category = sorted(expected_coefficients)[0]
    example_values = expected_coefficients[example_category]

    print("\nIndependent expectation example")
    print("-------------------------------")
    print(f"Category: {example_category}")
    print(
        "Rows: underflow, regular njets bins, overflow\n"
        "Columns: quadratic EFT coefficients"
    )

    print(
        np.array2string(
            example_values,
            precision=6,
            suppress_small=False,
        )
    )


# ----------------------------------------------------------------------
# Main validation workflow
# ----------------------------------------------------------------------

def run_validation(
    n_events=10_000,
    seed=42,
    rtol=1e-12,
    atol=1e-12,
):
    """
    Run the complete HistEFT Standard-versus-MultiCell validation.
    """
    print("=" * 78)
    print("HistEFT MultiCell validation")
    print("=" * 78)
    print(f"Number of events:          {n_events}")
    print(f"Random seed:               {seed}")
    print(f"Processes:                 {len(PROCESSES)}")
    print(f"Channels:                  {len(CHANNELS)}")
    print(f"Systematics:               {len(SYSTEMATICS)}")
    print(f"Expected sparse categories:{EXPECTED_CATEGORY_COUNT:>8}")

    data = generate_test_data(
        n_events=n_events,
        seed=seed,
    )

    validate_input_data(
        data=data,
        n_events=n_events,
    )

    print(
        "\n[PASS] Input dataset structure, numerical content, "
        "flow-bin coverage, and category coverage"
    )

    print_dataset_summary(data)
    print_category_summary(data)

    expected_coefficients = (
        calculate_expected_raw_coefficients(data)
    )

    standard_hist = build_hist(
        use_multicell=False
    )

    multicell_hist = build_hist(
        use_multicell=True
    )

    fill_hist(
        standard_hist,
        data,
    )

    fill_hist(
        multicell_hist,
        data,
    )

    print_storage_summary(
        standard_hist,
        "Standard HistEFT",
    )

    print_storage_summary(
        multicell_hist,
        "MultiCell HistEFT",
    )

    print("\nValidation results")
    print("------------------")

    compare_against_expected(
        histogram=standard_hist,
        expected_coefficients=expected_coefficients,
        label="Standard HistEFT",
        rtol=rtol,
        atol=atol,
    )

    compare_against_expected(
        histogram=multicell_hist,
        expected_coefficients=expected_coefficients,
        label="MultiCell HistEFT",
        rtol=rtol,
        atol=atol,
    )

    compare_raw_coefficients(
        standard_hist=standard_hist,
        multicell_hist=multicell_hist,
        rtol=rtol,
        atol=atol,
    )

    compare_evaluations(
        standard_hist=standard_hist,
        multicell_hist=multicell_hist,
        expected_coefficients=expected_coefficients,
        rtol=rtol,
        atol=atol,
    )

    print_expected_example(
        expected_coefficients
    )

    print("\n" + "=" * 78)
    print(
        "All HistEFT MultiCell validation checks passed "
        f"for {EXPECTED_CATEGORY_COUNT} sparse categories."
    )
    print("=" * 78)

    return standard_hist, multicell_hist


def parse_arguments():
    """
    Parse optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate Standard and MultiCell HistEFT implementations "
            "using a deterministic multi-category toy dataset."
        )
    )

    parser.add_argument(
        "--events",
        type=int,
        default=10_000,
        help="Number of toy events to generate (default: 10000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate the dataset (default: 42).",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-12,
        help="Relative tolerance for numerical comparisons.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for numerical comparisons.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_validation(
        n_events=arguments.events,
        seed=arguments.seed,
        rtol=arguments.rtol,
        atol=arguments.atol,
    )