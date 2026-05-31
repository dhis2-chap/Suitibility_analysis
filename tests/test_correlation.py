"""Tests for analysis/correlation.py: CorrelationAnalysis and _safe_float.

INTENDED LOGIC:
  _safe_float(): converts NaN/Inf to None; passes valid floats through; None → None.

  CorrelationAnalysis.run():
    - Used internally by run_lag_sweep to compute per-lag correlations.
    - Returns overall_correlation (Spearman/Pearson between composite score and cases)
      and continuous_variables (same for each raw climate variable).

  _overall_correlation():
    - Returns a 'note' and no correlation keys when fewer than 3 observations.
    - Computes spearman_r, pearson_r, r_squared.
    - r_squared must equal pearson_r ** 2.

  _analyze_continuous():
    - Correlates each raw climate variable with cases.
    - Skips rows where climate value is NaN (in addition to NaN cases).
    - Returns 'Too few observations' note when fewer than 3 rows remain.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.correlation import CorrelationAnalysis, _safe_float


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_pos_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _safe_float(float("-inf")) is None

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_valid_float_passes_through(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_zero_passes_through(self):
        assert _safe_float(0.0) == 0.0

    def test_negative_float_passes_through(self):
        assert _safe_float(-1.5) == pytest.approx(-1.5)

    def test_returns_float_type(self):
        result = _safe_float(1)
        assert isinstance(result, float)

    def test_numpy_nan_returns_none(self):
        assert _safe_float(np.nan) is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    return SuitabilityModel(
        name="test",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
            ThresholdComponent("precipitation", "rainfall", 1.0, None),
        ],
    )


@pytest.fixture
def simple_df():
    """10 rows: half suitable, half not, with controlled cases."""
    return pd.DataFrame({
        "mean_temperature": [25.0, 25.0, 25.0, 25.0, 25.0,
                             10.0, 10.0, 10.0, 10.0, 10.0],
        "rainfall":         [5.0,  5.0,  5.0,  5.0,  5.0,
                             5.0,  5.0,  5.0,  5.0,  5.0],
        "disease_cases":    [100., 80., 90., 110., 95.,
                              10., 15.,  5., 20., 12.],
        "population":       [100_000] * 10,
        "location":         ["A", "B", "A", "B", "A",
                              "B", "A", "B", "A", "B"],
        "time_period":      ["2010-01", "2010-02", "2010-03", "2010-04", "2010-05",
                              "2010-06", "2010-07", "2010-08", "2010-09", "2010-10"],
    })


# ---------------------------------------------------------------------------
# _overall_correlation
# ---------------------------------------------------------------------------

class TestOverallCorrelation:

    def test_requires_at_least_3_observations(self, simple_model):
        df = pd.DataFrame({
            "mean_temperature": [25.0, 10.0],
            "rainfall":         [5.0, 5.0],
            "disease_cases":    [100., 10.],
            "location":         ["A", "B"],
            "time_period":      ["2010-01", "2010-02"],
        })
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(df)
        oc = results["overall_correlation"]
        # With only 2 rows, a 'note' is returned instead of correlations.
        assert "note" in oc

    def test_r_squared_equals_pearson_r_squared(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        oc = results["overall_correlation"]
        if oc.get("pearson_r") is not None and oc.get("r_squared") is not None:
            assert oc["r_squared"] == pytest.approx(oc["pearson_r"] ** 2, rel=1e-6)

    def test_spearman_r_in_range_minus1_to_1(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        oc = results["overall_correlation"]
        if oc.get("spearman_r") is not None:
            assert -1.0 <= oc["spearman_r"] <= 1.0

    def test_pearson_r_in_range_minus1_to_1(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        oc = results["overall_correlation"]
        if oc.get("pearson_r") is not None:
            assert -1.0 <= oc["pearson_r"] <= 1.0

    def test_perfect_positive_correlation(self, simple_model):
        # score=2 → cases=100, score=0 → cases=1, score=1 → cases=50
        df = pd.DataFrame({
            "mean_temperature": [25.0, 25.0, 10.0, 10.0, 25.0, 10.0],
            "rainfall":         [5.0,  5.0,  0.1,  0.1,  0.1,  5.0],
            "disease_cases":    [100., 100., 1.,   1.,   1.,   50.],
            "location":         ["A", "B", "C", "D", "E", "F"],
            "time_period":      [f"2010-0{i}" for i in range(1, 7)],
        })
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(df)
        oc = results["overall_correlation"]
        if oc.get("spearman_r") is not None:
            assert oc["spearman_r"] > 0.8


# ---------------------------------------------------------------------------
# _analyze_continuous
# ---------------------------------------------------------------------------

class TestAnalyzeContinuous:

    def test_continuous_correlations_computed(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        cont = results["continuous_variables"]
        assert "mean_temperature" in cont
        assert "rainfall" in cont

    def test_continuous_contains_spearman_and_pearson(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        for col, stats in results["continuous_variables"].items():
            if "note" not in stats:
                assert "spearman_r" in stats
                assert "pearson_r" in stats

    def test_nan_climate_rows_excluded(self, simple_model):
        # One NaN in mean_temperature should not crash and should reduce effective n.
        df = pd.DataFrame({
            "mean_temperature": [25.0, float("nan"), 25.0, 10.0, 10.0, 10.0],
            "rainfall":         [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "disease_cases":    [100., 90., 80., 10., 20., 30.],
            "location":         list("ABCDEF"),
            "time_period":      [f"2010-0{i}" for i in range(1, 7)],
        })
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(df)
        temp_cont = results["continuous_variables"].get("mean_temperature", {})
        # Should succeed (5 valid rows) without raising.
        assert "note" not in temp_cont or "Too few" not in temp_cont.get("note", "")

    def test_too_few_rows_returns_note(self, simple_model):
        df = pd.DataFrame({
            "mean_temperature": [25.0, 10.0],
            "rainfall":         [5.0, 5.0],
            "disease_cases":    [100., 10.],
            "location":         ["A", "B"],
            "time_period":      ["2010-01", "2010-02"],
        })
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(df)
        temp_cont = results["continuous_variables"].get("mean_temperature", {})
        assert "note" in temp_cont


# ---------------------------------------------------------------------------
# Top-level run() keys
# ---------------------------------------------------------------------------

class TestRunKeys:

    def test_run_returns_expected_top_level_keys(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        for key in ["model_name", "n_observations", "overall_correlation",
                    "continuous_variables"]:
            assert key in results

    def test_model_name_matches(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        assert results["model_name"] == "test"

    def test_n_observations_matches_len_df(self, simple_model, simple_df):
        analysis = CorrelationAnalysis(simple_model)
        results = analysis.run(simple_df)
        assert results["n_observations"] == len(simple_df)
