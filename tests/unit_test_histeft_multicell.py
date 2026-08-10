"""
Unit tests for the HistEFT MultiCell implementation.

These tests verify that:

1. The deterministic toy dataset has the expected structure.
2. Standard HistEFT reproduces an independent NumPy calculation.
3. MultiCell HistEFT reproduces an independent NumPy calculation.
4. Standard and MultiCell raw EFT coefficients are equivalent.
5. Standard and MultiCell eval() results reproduce an independent
   polynomial evaluation.
6. Multiple processes, channels, systematics, dense bins, underflow,
   and overflow are handled correctly.
"""

from itertools import product

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT


# ----------------------------------------------------------------------
# Test configuration
# ----------------------------------------------------------------------

WC_NAMES = ["ctG", "cQu8"]

# Coefficient ordering for two Wilson coefficients:
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

N_EVENTS = 10_000
RANDOM_SEED = 42

RTOL = 1e-12
ATOL = 1e-12

EXPECTED_CATEGORY_COUNT = (
    len(PROCESSES)
    * len(CHANNELS)
    * len(SYSTEMATICS)
)

EXPECTED_COEFFICIENT_SHAPE = (
    N_BINS + 2,
    N_QUAD_TERMS,
)


# ----------------------------------------------------------------------
# Toy-data helpers
# ----------------------------------------------------------------------

def generate_test_data(
    n_events=N_EVENTS,
    seed=RANDOM_SEED,
):
    """
    Generate a deterministic multi-category toy dataset.
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

    # Values outside the histogram range deliberately exercise
    # underflow and overflow.
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
    Return all configured sparse-category combinations.
    """
    return set(
        product(
            PROCESSES,
            CHANNELS,
            SYSTEMATICS,
        )
    )


def category_combinations(data):
    """
    Return all sparse-category combinations present in the data.
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
    Return the event mask for one sparse category.
    """
    process_name, channel_name, systematic_name = category

    return (
        (data["process"] == process_name)
        & (data["channel"] == channel_name)
        & (data["systematic"] == systematic_name)
    )


# ----------------------------------------------------------------------
# Histogram helpers
# ----------------------------------------------------------------------

def build_hist(use_multicell):
    """
    Construct Standard or MultiCell HistEFT.
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
    Fill one sparse-category combination at a time.
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


def normalize_sparse_key(sparse_key):
    """
    Convert a SparseHist key into a regular tuple.
    """
    return tuple(sparse_key)


# ----------------------------------------------------------------------
# Independent reference calculations
# ----------------------------------------------------------------------

def dense_bin_indices(values):
    """
    Map values to dense-bin indices including flow bins.

    Index convention:

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
    Independently accumulate weighted EFT coefficients for every sparse
    category.
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
            EXPECTED_COEFFICIENT_SHAPE,
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
    Independently evaluate the quadratic EFT polynomial.
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
# HistEFT-output extraction
# ----------------------------------------------------------------------

def extract_all_raw_coefficients(histogram):
    """
    Extract Standard and MultiCell coefficients in the common public layout:

        (dense bins including flow, EFT coefficients)
    """
    output = {}

    for sparse_key, values in histogram.yield_coefficients(flow=True).items():
        coefficients = np.asarray(values)
        assert coefficients.shape == EXPECTED_COEFFICIENT_SHAPE

        category = normalize_sparse_key(sparse_key)

        assert category not in output

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
# Pytest fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_data():
    """
    Generate the dataset once for the complete test module.
    """
    return generate_test_data()


@pytest.fixture(scope="module")
def expected_coefficients(toy_data):
    """
    Calculate the independent NumPy expectation once.
    """
    return calculate_expected_raw_coefficients(toy_data)


@pytest.fixture(scope="module")
def standard_hist(toy_data):
    """
    Build and fill the Standard HistEFT once.
    """
    histogram = build_hist(
        use_multicell=False
    )

    fill_hist(
        histogram,
        toy_data,
    )

    return histogram


@pytest.fixture(scope="module")
def multicell_hist(toy_data):
    """
    Build and fill the MultiCell HistEFT once.
    """
    histogram = build_hist(
        use_multicell=True
    )

    fill_hist(
        histogram,
        toy_data,
    )

    return histogram


# ----------------------------------------------------------------------
# Input and dataset tests
# ----------------------------------------------------------------------

def test_generated_data_shapes(toy_data):
    """
    Check that all event arrays have consistent dimensions.
    """
    assert toy_data["process"].shape == (N_EVENTS,)
    assert toy_data["channel"].shape == (N_EVENTS,)
    assert toy_data["systematic"].shape == (N_EVENTS,)
    assert toy_data["njets"].shape == (N_EVENTS,)
    assert toy_data["weight"].shape == (N_EVENTS,)

    assert toy_data["eft_coeff"].shape == (
        N_EVENTS,
        N_QUAD_TERMS,
    )


def test_generated_data_are_finite(toy_data):
    """
    Check that all numerical input values are finite.
    """
    assert np.all(np.isfinite(toy_data["njets"]))
    assert np.all(np.isfinite(toy_data["weight"]))
    assert np.all(np.isfinite(toy_data["eft_coeff"]))


def test_generated_data_are_nontrivial(toy_data):
    """
    Check that weights and EFT coefficients exercise nontrivial cases.
    """
    assert np.all(toy_data["weight"] > 0)
    assert np.all(toy_data["eft_coeff"][:, 0] > 0)

    bsm_coefficients = toy_data["eft_coeff"][:, 1:]

    assert np.any(bsm_coefficients > 0)
    assert np.any(bsm_coefficients < 0)


def test_all_categories_are_present(toy_data):
    """
    Check that all process-channel-systematic combinations exist.
    """
    present_categories = set(
        category_combinations(toy_data)
    )

    assert present_categories == all_possible_categories()

    assert len(present_categories) == EXPECTED_CATEGORY_COUNT


def test_flow_bins_are_populated(toy_data):
    """
    Check that underflow, regular bins, and overflow are all exercised.
    """
    underflow_count = np.count_nonzero(
        toy_data["njets"] < BIN_START
    )

    regular_count = np.count_nonzero(
        (toy_data["njets"] >= BIN_START)
        & (toy_data["njets"] < BIN_STOP)
    )

    overflow_count = np.count_nonzero(
        toy_data["njets"] >= BIN_STOP
    )

    assert underflow_count > 0
    assert regular_count > 0
    assert overflow_count > 0

    assert (
        underflow_count
        + regular_count
        + overflow_count
        == N_EVENTS
    )


# ----------------------------------------------------------------------
# Histogram-structure tests
# ----------------------------------------------------------------------

def test_standard_hist_axes(standard_hist):
    """
    Check the original Standard HistEFT axis structure.
    """
    assert [axis.name for axis in standard_hist.axes] == [
        "process",
        "channel",
        "systematic",
        "njets",
        "quadratic_term",
    ]


def test_multicell_hist_axes(multicell_hist):
    """
    Check that MultiCell removes the explicit quadratic-term axis.
    """
    assert [axis.name for axis in multicell_hist.axes] == [
        "process",
        "channel",
        "systematic",
        "njets",
    ]


def test_standard_raw_storage_shape(standard_hist):
    """
    Check the representative Standard raw storage shape.
    """
    raw = standard_hist.view(
        flow=True,
        as_dict=True,
    )

    assert len(raw) == EXPECTED_CATEGORY_COUNT

    representative_values = np.asarray(
        next(iter(raw.values()))
    )

    assert representative_values.shape == (
        N_BINS + 2,
        N_QUAD_TERMS + 2,
    )


def test_multicell_raw_storage_shape(multicell_hist):
    """
    Check the representative MultiCell raw storage shape.
    """
    raw = multicell_hist.view(
        flow=True,
        as_dict=True,
    )

    assert len(raw) == EXPECTED_CATEGORY_COUNT

    representative_values = np.asarray(
        next(iter(raw.values()))
    )

    assert representative_values.shape == (
        N_QUAD_TERMS,
        N_BINS + 2,
    )


def test_standard_matches_independent_reference(
    standard_hist,
    expected_coefficients,
):
    """
    Verify Standard HistEFT against the independent NumPy calculation.
    """
    actual = extract_all_raw_coefficients(
        standard_hist
    )

    assert set(actual) == set(expected_coefficients)

    for category in sorted(expected_coefficients):
        np.testing.assert_allclose(
            actual[category],
            expected_coefficients[category],
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "Standard HistEFT raw coefficient mismatch "
                f"for category {category}"
            ),
        )


def test_multicell_matches_independent_reference(
    multicell_hist,
    expected_coefficients,
):
    """
    Verify MultiCell HistEFT against the independent NumPy calculation.
    """
    actual = extract_all_raw_coefficients(
        multicell_hist
    )

    assert set(actual) == set(expected_coefficients)

    for category in sorted(expected_coefficients):
        np.testing.assert_allclose(
            actual[category],
            expected_coefficients[category],
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "MultiCell HistEFT raw coefficient mismatch "
                f"for category {category}"
            ),
        )


def test_standard_and_multicell_raw_coefficients_match(
    standard_hist,
    multicell_hist,
):
    """
    Compare Standard and MultiCell directly for every sparse category.
    """
    standard = extract_all_raw_coefficients(
        standard_hist
    )

    multicell = extract_all_raw_coefficients(
        multicell_hist
    )

    assert set(standard) == set(multicell)

    for category in sorted(standard):
        np.testing.assert_allclose(
            standard[category],
            multicell[category],
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "Standard-versus-MultiCell raw coefficient mismatch "
                f"for category {category}"
            ),
        )


# ----------------------------------------------------------------------
# EFT evaluation tests
# ----------------------------------------------------------------------

WC_TEST_POINTS = [
    {},
    {"ctG": 0.0, "cQu8": 0.0},
    {"ctG": 1.0, "cQu8": 0.0},
    {"ctG": 0.0, "cQu8": 1.0},
    {"ctG": 1.0, "cQu8": 1.0},
    {"ctG": -0.5, "cQu8": 0.75},
]


@pytest.mark.parametrize(
    "wc_point",
    WC_TEST_POINTS,
)
def test_standard_eval_matches_independent_reference(
    standard_hist,
    expected_coefficients,
    wc_point,
):
    """
    Verify Standard HistEFT eval() at several WC points.
    """
    actual = extract_all_evaluations(
        standard_hist,
        wc_point,
    )

    assert set(actual) == set(expected_coefficients)

    for category in sorted(expected_coefficients):
        expected = calculate_expected_evaluation(
            expected_coefficients[category],
            wc_point,
        )

        np.testing.assert_allclose(
            actual[category],
            expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"Standard eval() mismatch for category {category} "
                f"at WC point {wc_point}"
            ),
        )


@pytest.mark.parametrize(
    "wc_point",
    WC_TEST_POINTS,
)
def test_multicell_eval_matches_independent_reference(
    multicell_hist,
    expected_coefficients,
    wc_point,
):
    """
    Verify MultiCell HistEFT eval() at several WC points.
    """
    actual = extract_all_evaluations(
        multicell_hist,
        wc_point,
    )

    assert set(actual) == set(expected_coefficients)

    for category in sorted(expected_coefficients):
        expected = calculate_expected_evaluation(
            expected_coefficients[category],
            wc_point,
        )

        np.testing.assert_allclose(
            actual[category],
            expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"MultiCell eval() mismatch for category {category} "
                f"at WC point {wc_point}"
            ),
        )


@pytest.mark.parametrize(
    "wc_point",
    WC_TEST_POINTS,
)
def test_standard_and_multicell_eval_match(
    standard_hist,
    multicell_hist,
    wc_point,
):
    """
    Compare Standard and MultiCell eval() directly.
    """
    standard = extract_all_evaluations(
        standard_hist,
        wc_point,
    )

    multicell = extract_all_evaluations(
        multicell_hist,
        wc_point,
    )

    assert set(standard) == set(multicell)

    for category in sorted(standard):
        np.testing.assert_allclose(
            standard[category],
            multicell[category],
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                "Standard-versus-MultiCell eval() mismatch "
                f"for category {category} at WC point {wc_point}"
            ),
        )
