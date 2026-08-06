from __future__ import annotations

import hist
import numpy as np
import pytest
from coffea import processor

from topcoffea.modules.histEFT import HistEFT


WC_NAMES = ["ctG", "ctW"]
N_QUAD = 6
def _make_histeft(*, use_multicell: bool, bins: int = 2) -> HistEFT:
    return HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(bins, 0.0, float(bins), name="observable"),
        wc_names=WC_NAMES,
        use_multicell=use_multicell,
        store_sumw2=use_multicell,
    )


def _normalized(mapping):
    return {tuple(key): np.asarray(value) for key, value in mapping.items()}


def _fill_equivalent_toy_histograms():
    legacy = _make_histeft(use_multicell=False)
    multicell = _make_histeft(use_multicell=True)
    observable = np.array([-0.5, 0.25, 1.25, 2.5])
    weight = np.array([1.5, -2.0, 0.5, 3.0])
    eft_coeff = np.array(
        [
            [1.0, 0.5, 0.25, -0.5, 0.75, 0.125],
            [2.0, -1.0, 0.5, 0.25, -0.75, 1.5],
            [0.75, 0.25, -0.5, 1.0, 0.5, -0.25],
            [1.25, -0.5, 1.0, -1.5, 0.25, 0.75],
        ]
    )

    for systematic in ("nominal", "JERUp"):
        common = {
            "process": "ttH",
            "systematic": systematic,
            "observable": observable,
            "weight": weight,
            "eft_coeff": eft_coeff,
        }
        legacy.fill(**common)
        multicell.fill(
            **common,
            fill_sumw2=systematic == "nominal",
        )

    return legacy, multicell


@pytest.fixture
def equivalent_histograms():
    return _fill_equivalent_toy_histograms()


@pytest.mark.parametrize("flow,expected_dense_bins", [(True, 4), (False, 2)])
def test_public_yield_coefficients_match_between_backends(
    equivalent_histograms,
    flow,
    expected_dense_bins,
):
    legacy, multicell = equivalent_histograms
    legacy_coefficients = _normalized(legacy.yield_coefficients(flow=flow))
    multicell_coefficients = _normalized(
        multicell.yield_coefficients(flow=flow)
    )

    assert set(legacy_coefficients) == set(multicell_coefficients)
    assert set(legacy_coefficients) == {
        ("ttH", "nominal"),
        ("ttH", "JERUp"),
    }
    for sparse_key in legacy_coefficients:
        assert legacy_coefficients[sparse_key].shape == (
            expected_dense_bins,
            N_QUAD,
        )
        assert multicell_coefficients[sparse_key].shape == (
            expected_dense_bins,
            N_QUAD,
        )
        np.testing.assert_allclose(
            multicell_coefficients[sparse_key],
            legacy_coefficients[sparse_key],
        )

    stacked = np.concatenate(list(multicell_coefficients.values()), axis=0)
    assert np.any(stacked[..., 0] != 0.0)
    assert np.any(stacked[..., 1:] != 0.0)


@pytest.mark.parametrize(
    "wc_point",
    [
        {},
        {"ctG": 1.0},
        {"ctG": -1.0},
        {"ctG": 0.5, "ctW": -0.3},
    ],
)
def test_eval_matches_between_backends(equivalent_histograms, wc_point):
    legacy, multicell = equivalent_histograms
    legacy_evaluated = _normalized(legacy.eval(wc_point))
    multicell_evaluated = _normalized(multicell.eval(wc_point))

    assert set(legacy_evaluated) == set(multicell_evaluated)
    for sparse_key in legacy_evaluated:
        # eval() includes underflow and overflow for the physical dense axis.
        assert legacy_evaluated[sparse_key].shape == (4,)
        np.testing.assert_allclose(
            multicell_evaluated[sparse_key],
            legacy_evaluated[sparse_key],
        )
        np.testing.assert_allclose(
            multicell_evaluated[sparse_key][1:-1],
            legacy_evaluated[sparse_key][1:-1],
        )


def test_multicell_internal_layout_keeps_sumw2_out_of_public_yields():
    _, multicell = _fill_equivalent_toy_histograms()
    public_yields = _normalized(multicell.yield_coefficients(flow=False))
    public_sumw2 = _normalized(multicell.nominal_sumw2(flow=False))
    raw = _normalized(multicell.view(flow=False, as_dict=True))

    assert multicell._cell_count == multicell._quad_count + 1
    assert multicell._sumw2_index == multicell._quad_count
    for sparse_key, raw_values in raw.items():
        assert raw_values.shape == (N_QUAD + 1, 2)
        assert public_yields[sparse_key].shape == (2, N_QUAD)
        np.testing.assert_allclose(
            raw_values[:N_QUAD].T,
            public_yields[sparse_key],
        )
        np.testing.assert_allclose(
            raw_values[N_QUAD],
            public_sumw2[sparse_key],
        )


def test_embedded_nominal_sumw2_semantics_and_fill_gate():
    histogram = _make_histeft(use_multicell=True, bins=1)
    weight = np.array([3.0, -2.0])
    eft_coeff = np.array(
        [
            [2.0, 0.5, 0.25, 0.0, 0.0, 0.0],
            [1.0, -0.5, 0.25, 0.0, 0.0, 0.0],
        ]
    )

    for systematic, fill_sumw2 in (("nominal", True), ("JERUp", False)):
        histogram.fill(
            process="ttH",
            systematic=systematic,
            observable=np.array([0.5, 0.5]),
            weight=weight,
            eft_coeff=eft_coeff,
            fill_sumw2=fill_sumw2,
        )

    sumw2_without_flow = _normalized(histogram.nominal_sumw2(flow=False))
    np.testing.assert_allclose(
        sumw2_without_flow[("ttH", "nominal")],
        [6.0**2 + (-2.0) ** 2],
    )
    np.testing.assert_array_equal(
        sumw2_without_flow[("ttH", "JERUp")],
        [0.0],
    )

    sumw2_with_flow = _normalized(histogram.nominal_sumw2(flow=True))
    np.testing.assert_allclose(
        sumw2_with_flow[("ttH", "nominal")],
        [0.0, 40.0, 0.0],
    )


def test_disabled_sumw2_fill_keeps_fixed_layout_and_zero_scalar():
    histogram = _make_histeft(use_multicell=True, bins=1)
    histogram.fill(
        process="ttH",
        systematic="nominal",
        observable=np.array([0.5]),
        weight=np.array([4.0]),
        eft_coeff=np.array([[2.0, 1.0, 0.5, -1.0, 0.25, 0.125]]),
        fill_sumw2=False,
    )

    assert histogram.store_sumw2 is True
    assert histogram._cell_count == histogram._quad_count + 1
    assert histogram._sumw2_index == histogram._quad_count
    sumw2 = _normalized(histogram.nominal_sumw2(flow=False))
    np.testing.assert_array_equal(sumw2[("ttH", "nominal")], [0.0])


def test_missing_eft_coeff_uses_constant_sm_for_embedded_sumw2():
    histogram = _make_histeft(use_multicell=True, bins=1)
    weight = np.array([3.0, -2.0])
    histogram.fill(
        process="background",
        systematic="nominal",
        observable=np.array([0.5, 0.5]),
        weight=weight,
        eft_coeff=None,
        fill_sumw2=True,
    )

    sumw2 = _normalized(histogram.nominal_sumw2(flow=False))
    np.testing.assert_allclose(
        sumw2[("background", "nominal")],
        [np.sum(np.square(weight))],
    )


def test_legacy_companion_uses_squared_weight_with_original_eft_coefficients():
    companion = _make_histeft(use_multicell=False, bins=1)
    weight = np.array([2.0, -3.0])
    eft_coeff = np.array(
        [
            [1.0, 0.5, 0.25, -0.5, 0.75, 0.125],
            [2.0, -1.0, 0.5, 0.25, -0.75, 1.5],
        ]
    )
    companion.fill(
        process="ttH",
        systematic="nominal",
        observable=np.array([0.5, 0.5]),
        weight=np.square(weight),
        eft_coeff=eft_coeff,
    )

    coefficients = _normalized(companion.yield_coefficients(flow=False))
    expected = np.sum(np.square(weight)[:, None] * eft_coeff, axis=0)
    np.testing.assert_allclose(
        coefficients[("ttH", "nominal")],
        expected[None, :],
    )


def test_accumulating_processor_outputs_preserves_embedded_sumw2():
    left = _make_histeft(use_multicell=True, bins=1)
    right = _make_histeft(use_multicell=True, bins=1)
    fill_kwargs = {
        "process": "ttH",
        "systematic": "nominal",
        "observable": np.array([0.5]),
        "weight": np.array([3.0]),
        "eft_coeff": np.array([[2.0, 1.0, 0.5, -1.0, 0.25, 0.125]]),
        "fill_sumw2": True,
    }
    left.fill(**fill_kwargs)
    right.fill(**fill_kwargs)

    single_yields = _normalized(left.yield_coefficients(flow=False))
    single_sumw2 = _normalized(left.nominal_sumw2(flow=False))
    accumulated = processor.accumulate([{"njets": left}, {"njets": right}])
    combined = accumulated["njets"]
    combined_yields = _normalized(combined.yield_coefficients(flow=False))
    combined_sumw2 = _normalized(combined.nominal_sumw2(flow=False))

    sparse_key = ("ttH", "nominal")
    np.testing.assert_allclose(
        combined_yields[sparse_key],
        2.0 * single_yields[sparse_key],
    )
    np.testing.assert_allclose(
        combined_sumw2[sparse_key],
        2.0 * single_sumw2[sparse_key],
    )
