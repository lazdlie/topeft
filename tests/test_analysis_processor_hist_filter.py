import numpy as np
import pytest

cloudpickle = pytest.importorskip("cloudpickle")

from analysis.topeft_run2 import analysis_processor
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d


ONE_DIMENSIONAL_HIST = "njets"
TWO_DIMENSIONAL_HIST = "lepton_pt_vs_eta"


def _make_processor(hist_lst=None, fill_sumw2_hist=True, use_multicell=True):
    # The processor only needs a samples dictionary for instantiation; the
    # individual entries are accessed during processing, so the tests can use an
    # empty mapping here.
    return analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=["ctG", "ctW"],
        hist_lst=hist_lst,
        fill_sumw2_hist=fill_sumw2_hist,
        use_multicell=use_multicell,
    )


def _expected_keys(base_names, *, use_multicell, fill_sumw2_hist):
    expected = set(base_names)
    if not fill_sumw2_hist:
        return expected

    for name in base_names:
        if name in axes_info_2d or not use_multicell:
            expected.add(f"{name}_sumw2")
    return expected


@pytest.mark.parametrize("use_multicell", [False, True])
@pytest.mark.parametrize("fill_sumw2_hist", [False, True])
def test_accumulator_keys_without_hist_filter(use_multicell, fill_sumw2_hist):
    processor = _make_processor(
        fill_sumw2_hist=fill_sumw2_hist,
        use_multicell=use_multicell,
    )
    base_names = set(axes_info) | set(axes_info_2d)
    expected_keys = _expected_keys(
        base_names,
        use_multicell=use_multicell,
        fill_sumw2_hist=fill_sumw2_hist,
    )

    assert set(processor.accumulator) == expected_keys
    assert set(processor._hist_lst) == expected_keys
    assert set(processor._hist_axis_map) == expected_keys
    assert set(processor._hist_requires_eft) == expected_keys


@pytest.mark.parametrize("use_multicell", [False, True])
@pytest.mark.parametrize("fill_sumw2_hist", [False, True])
def test_one_and_two_dimensional_histogram_construction(
    use_multicell,
    fill_sumw2_hist,
):
    processor = _make_processor(
        hist_lst=[ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST],
        fill_sumw2_hist=fill_sumw2_hist,
        use_multicell=use_multicell,
    )
    expected_keys = _expected_keys(
        {ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST},
        use_multicell=use_multicell,
        fill_sumw2_hist=fill_sumw2_hist,
    )

    assert set(processor.accumulator) == expected_keys

    one_dimensional = processor.accumulator[ONE_DIMENSIONAL_HIST]
    assert isinstance(one_dimensional, HistEFT)
    assert one_dimensional._use_multicell is use_multicell
    assert one_dimensional.store_sumw2 is use_multicell
    assert ("quadratic_term" in one_dimensional.axes.name) is (not use_multicell)
    if use_multicell:
        assert one_dimensional._cell_count == one_dimensional._quad_count + 1
        assert one_dimensional._sumw2_index == one_dimensional._quad_count

    one_dimensional_sumw2 = f"{ONE_DIMENSIONAL_HIST}_sumw2"
    assert (one_dimensional_sumw2 in processor.accumulator) is (
        fill_sumw2_hist and not use_multicell
    )
    if one_dimensional_sumw2 in processor.accumulator:
        companion = processor.accumulator[one_dimensional_sumw2]
        assert isinstance(companion, HistEFT)
        assert companion._use_multicell is False
        assert companion.store_sumw2 is False
        assert "quadratic_term" in companion.axes.name

    two_dimensional = processor.accumulator[TWO_DIMENSIONAL_HIST]
    assert isinstance(two_dimensional, SparseHist)
    assert not isinstance(two_dimensional, HistEFT)
    assert not hasattr(two_dimensional, "store_sumw2")
    assert not hasattr(two_dimensional, "nominal_sumw2")
    assert (
        f"{TWO_DIMENSIONAL_HIST}_sumw2" in processor.accumulator
    ) is fill_sumw2_hist

    expected_mapping_keys = set()
    if fill_sumw2_hist:
        expected_mapping_keys.add(TWO_DIMENSIONAL_HIST)
        if not use_multicell:
            expected_mapping_keys.add(ONE_DIMENSIONAL_HIST)
    assert set(processor._hist_sumw2_axis_mapping) == expected_mapping_keys
    assert set(processor._hist_sumw2_axis_mapping) <= set(
        processor._base_hist_name_set
    )

    serialized = cloudpickle.dumps(processor.accumulator)
    restored = cloudpickle.loads(serialized)
    assert set(restored) == expected_keys


@pytest.mark.parametrize(
    "requested_hists,use_multicell,fill_sumw2_hist,expected_keys",
    [
        (["njets"], False, True, {"njets", "njets_sumw2"}),
        (["njets"], True, True, {"njets"}),
        (["njets_sumw2"], True, True, {"njets"}),
        (["njets_sumw2"], False, True, {"njets", "njets_sumw2"}),
        (["njets_sumw2"], False, False, {"njets"}),
        (
            [TWO_DIMENSIONAL_HIST],
            False,
            True,
            {TWO_DIMENSIONAL_HIST, f"{TWO_DIMENSIONAL_HIST}_sumw2"},
        ),
        (
            [f"{TWO_DIMENSIONAL_HIST}_sumw2"],
            True,
            True,
            {TWO_DIMENSIONAL_HIST, f"{TWO_DIMENSIONAL_HIST}_sumw2"},
        ),
        ([TWO_DIMENSIONAL_HIST], True, False, {TWO_DIMENSIONAL_HIST}),
    ],
)
def test_histogram_list_resolution(
    requested_hists,
    use_multicell,
    fill_sumw2_hist,
    expected_keys,
):
    processor = _make_processor(
        hist_lst=requested_hists,
        fill_sumw2_hist=fill_sumw2_hist,
        use_multicell=use_multicell,
    )

    assert set(processor.accumulator) == expected_keys
    assert set(processor._hist_lst) == expected_keys
    assert set(processor._hist_axis_map) == expected_keys
    assert set(processor._hist_requires_eft) == expected_keys

    expected_mapping_keys = {
        name
        for name in processor._base_hist_name_set
        if f"{name}_sumw2" in expected_keys
    }
    assert set(processor._hist_sumw2_axis_mapping) == expected_mapping_keys


@pytest.mark.parametrize(
    "fill_sumw2_hist,wgt_fluct,expected",
    [
        (True, "nominal", True),
        (True, "triggerSF_2022Up", False),
        (True, "JERUp", False),
        (False, "nominal", False),
    ],
)
def test_sumw2_fill_gate_only_allows_nominal_path(
    fill_sumw2_hist,
    wgt_fluct,
    expected,
):
    assert (
        analysis_processor.AnalysisProcessor._should_fill_sumw2_histogram(
            fill_sumw2_hist,
            wgt_fluct=wgt_fluct,
        )
        is expected
    )


class _FillRecorder:
    def __init__(self):
        self.calls = []

    def fill(self, **kwargs):
        self.calls.append(kwargs)


def _exercise_processor_fill_block(processor, *, systematic, weight, eft_coeff):
    """Exercise the fill-routing decisions with minimal histogram test doubles."""

    recorders = {name: _FillRecorder() for name in processor.accumulator}
    base_kwargs = {
        "process": "ttH",
        "channel": "2lss",
        "systematic": systematic,
        "appl": "isSR",
        ONE_DIMENSIONAL_HIST: np.array([1.0, 2.0]),
        "weight": weight,
        "eft_coeff": eft_coeff,
    }
    if processor._use_embedded_sumw2:
        base_kwargs["fill_sumw2"] = processor._should_fill_sumw2_histogram(
            processor._fill_sumw2_hist,
            wgt_fluct=systematic,
        )
    recorders[ONE_DIMENSIONAL_HIST].fill(**base_kwargs)

    companion_mapping = processor._hist_sumw2_axis_mapping.get(
        ONE_DIMENSIONAL_HIST
    )
    fill_companion = processor._should_fill_sumw2_histogram(
        processor._fill_sumw2_hist and bool(companion_mapping),
        wgt_fluct=systematic,
    )
    if fill_companion:
        companion_axis = next(iter(companion_mapping))
        recorders[f"{ONE_DIMENSIONAL_HIST}_sumw2"].fill(
            process="ttH",
            channel="2lss",
            systematic=systematic,
            appl="isSR",
            **{companion_axis: np.array([1.0, 2.0])},
            weight=np.square(weight),
            eft_coeff=eft_coeff,
        )

    return recorders


@pytest.mark.parametrize(
    "use_multicell,fill_sumw2_hist,systematic,expected_fill_sumw2,companion_calls",
    [
        (False, True, "nominal", None, 1),
        (False, True, "JERUp", None, 0),
        (True, True, "nominal", True, 0),
        (True, True, "JERUp", False, 0),
        (True, False, "nominal", False, 0),
    ],
)
def test_processor_fill_routing_contract(
    use_multicell,
    fill_sumw2_hist,
    systematic,
    expected_fill_sumw2,
    companion_calls,
):
    processor = _make_processor(
        hist_lst=[ONE_DIMENSIONAL_HIST],
        fill_sumw2_hist=fill_sumw2_hist,
        use_multicell=use_multicell,
    )
    weight = np.array([2.0, -3.0])
    eft_coeff = np.arange(12.0).reshape(2, 6)
    recorders = _exercise_processor_fill_block(
        processor,
        systematic=systematic,
        weight=weight,
        eft_coeff=eft_coeff,
    )

    assert len(recorders[ONE_DIMENSIONAL_HIST].calls) == 1
    base_call = recorders[ONE_DIMENSIONAL_HIST].calls[0]
    np.testing.assert_array_equal(base_call["weight"], weight)
    assert base_call["eft_coeff"] is eft_coeff
    if expected_fill_sumw2 is None:
        assert "fill_sumw2" not in base_call
    else:
        assert base_call["fill_sumw2"] is expected_fill_sumw2

    companion_name = f"{ONE_DIMENSIONAL_HIST}_sumw2"
    if companion_name not in recorders:
        assert companion_calls == 0
        return

    assert len(recorders[companion_name].calls) == companion_calls
    if companion_calls:
        companion_call = recorders[companion_name].calls[0]
        np.testing.assert_array_equal(companion_call["weight"], np.square(weight))
        assert companion_call["eft_coeff"] is eft_coeff
