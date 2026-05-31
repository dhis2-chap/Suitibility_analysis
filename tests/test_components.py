"""Tests for analysis/components.py: analyze_leave_one_out.

INTENDED LOGIC:
  analyze_leave_one_out():
  - Computes Spearman r for the full model and each leave-one-out reduced model.
  - delta_r = full_r - reduced_r. If either is None, delta_r is None.
  - Reduced models contain all components EXCEPT the removed one.
  - The full model results are stored under "full_model" key.
  - Each LOO result is stored under "without" -> component_name.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.components import analyze_leave_one_out, _safe


# ---------------------------------------------------------------------------
# _safe
# ---------------------------------------------------------------------------

class TestSafeHelper:

    def test_nan_returns_none(self):
        assert _safe(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe(float("inf")) is None
        assert _safe(float("-inf")) is None

    def test_none_returns_none(self):
        assert _safe(None) is None

    def test_valid_float_passes_through(self):
        assert _safe(3.14) == pytest.approx(3.14)

    def test_zero_passes_through(self):
        assert _safe(0.0) == 0.0

    def test_numpy_nan_returns_none(self):
        assert _safe(np.nan) is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_comp_model():
    return SuitabilityModel(
        name="test",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
            ThresholdComponent("precipitation", "rainfall", 1.0, None),
        ],
    )


@pytest.fixture
def three_comp_model():
    return SuitabilityModel(
        name="test3",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
            ThresholdComponent("precipitation", "rainfall", 1.0, None),
            ThresholdComponent("humidity", "mean_relative_humidity", 50.0, 80.0),
        ],
    )


def make_df(n=30, seed=42):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01", periods=n, freq="MS")
    return pd.DataFrame({
        "time_period": [m.strftime("%Y-%m") for m in months],
        "location": (["A"] * (n // 2) + ["B"] * (n - n // 2)),
        "disease_cases": rng.integers(0, 100, n).astype(float),
        "population": [100_000] * n,
        "mean_temperature": rng.uniform(15, 35, n),
        "rainfall": rng.uniform(0, 15, n),
        "mean_relative_humidity": rng.uniform(40, 90, n),
    })


# ---------------------------------------------------------------------------
# analyze_leave_one_out — structure
# ---------------------------------------------------------------------------

class TestAnalyzeLeaveOneOut:

    @pytest.fixture
    def temporal_df(self):
        """Multi-location DataFrame with enough months for rolling windows."""
        months = pd.date_range("2010-01", periods=30, freq="MS")
        rows = []
        for loc in ["A", "B", "C"]:
            for i, m in enumerate(months):
                rows.append({
                    "time_period": m.strftime("%Y-%m"),
                    "location": loc,
                    "disease_cases": float(10 * (i % 12 + 1)),
                    "mean_temperature": 20.0 + 5 * np.sin(i * np.pi / 6),
                    "rainfall": 3.0 + 4 * max(0, np.sin(i * np.pi / 6)),
                    "mean_relative_humidity": 60.0 + 10 * np.sin(i * np.pi / 6),
                })
        return pd.DataFrame(rows)

    def test_returns_full_model_and_without_keys(self, three_comp_model, temporal_df):
        result = analyze_leave_one_out(temporal_df, three_comp_model)
        assert "full_model" in result
        assert "without" in result

    def test_full_model_has_n_components(self, three_comp_model, temporal_df):
        result = analyze_leave_one_out(temporal_df, three_comp_model)
        assert result["full_model"]["n_components"] == 3

    def test_without_has_entry_per_component(self, three_comp_model, temporal_df):
        result = analyze_leave_one_out(temporal_df, three_comp_model)
        for comp in three_comp_model.components:
            assert comp.name in result["without"]

    def test_without_entry_has_required_keys(self, three_comp_model, temporal_df):
        result = analyze_leave_one_out(temporal_df, three_comp_model)
        for comp_name, comp_result in result["without"].items():
            assert "spearman_r" in comp_result
            assert "spearman_p" in comp_result
            assert "delta_r" in comp_result
            assert "n_windows" in comp_result

    def test_delta_r_is_none_when_full_r_none(self, two_comp_model):
        # With very little data, full_r may be None → delta_r must also be None.
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02", "2010-03"],
            "location": ["A", "A", "A"],
            "disease_cases": [10.0, 20.0, 30.0],
            "mean_temperature": [25.0, 25.0, 25.0],
            "rainfall": [5.0, 5.0, 5.0],
        })
        result = analyze_leave_one_out(df, two_comp_model)
        full_r = result["full_model"]["spearman_r"]
        if full_r is None:
            for comp_result in result["without"].values():
                assert comp_result["delta_r"] is None

    def test_delta_r_equals_full_r_minus_reduced_r(self, three_comp_model, temporal_df):
        result = analyze_leave_one_out(temporal_df, three_comp_model)
        full_r = result["full_model"]["spearman_r"]
        for comp_name, comp_result in result["without"].items():
            full_r_val = full_r
            reduced_r = comp_result["spearman_r"]
            delta = comp_result["delta_r"]
            if full_r_val is not None and reduced_r is not None and delta is not None:
                assert delta == pytest.approx(full_r_val - reduced_r, abs=1e-9)

    def test_n_windows_positive_when_enough_data(self, three_comp_model, temporal_df):
        result = analyze_leave_one_out(temporal_df, three_comp_model)
        assert result["full_model"]["n_windows"] > 0
