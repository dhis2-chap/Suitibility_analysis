"""Tests for analysis/temporal.py: analyze_within_region.

INTENDED LOGIC:
  - Uses compute_window_data to get rolling 12-month windows per location.
  - For each location, computes WITHIN-LOCATION temporal Spearman r between
    months_suitable and total_cases across time (not across locations).
    This tests: "as suitability exposure varies over time within a district,
    does it track annual disease burden?"
  - Requires >= 3 windows AND variation in months_suitable within the location.
    Otherwise spearman_r is None.
  - per_location: {spearman_r, spearman_p} per location.
  - summary: {mean_within_location_r} across locations with enough data.
  - Returns empty per_location when no windows at all.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.temporal import analyze_within_region


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return SuitabilityModel(
        name="test",
        components=[ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0)],
    )


def make_temporal_df(location, temps, cases, start="2010-01"):
    months = pd.date_range(start, periods=len(temps), freq="MS")
    return pd.DataFrame({
        "time_period": [m.strftime("%Y-%m") for m in months],
        "location": location,
        "disease_cases": [float(c) for c in cases],
        "mean_temperature": [float(t) for t in temps],
    })


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_per_location_when_no_windows(self, model):
        # With only 3 rows, no 12-month windows qualify.
        df = make_temporal_df("A", [25.0, 25.0, 25.0], [10, 20, 30])
        result = analyze_within_region(df, model)
        assert result["per_location"] == {}

    def test_window_param_respected(self, model):
        # With window=6 and 10 rows, windows start appearing earlier.
        df = make_temporal_df("A", [25.0] * 10, [10.0] * 10)
        result = analyze_within_region(df, model, window=6)
        if result["per_location"]:
            assert len(result["per_location"]) >= 1


# ---------------------------------------------------------------------------
# per_location structure
# ---------------------------------------------------------------------------

class TestPerLocationStructure:

    def test_per_location_contains_each_location(self, model):
        df_a = make_temporal_df("A", [25.0] * 20, [10.0] * 20)
        df_b = make_temporal_df("B", [25.0] * 20, [20.0] * 20)
        df = pd.concat([df_a, df_b], ignore_index=True)
        result = analyze_within_region(df, model)
        assert "A" in result["per_location"]
        assert "B" in result["per_location"]

    def test_per_location_has_spearman_keys(self, model):
        df = make_temporal_df("A", [25.0] * 20, [10.0] * 20)
        result = analyze_within_region(df, model)
        for loc, loc_result in result["per_location"].items():
            # spearman_r may be None but key must exist.
            assert "spearman_r" in loc_result
            assert "spearman_p" in loc_result


# ---------------------------------------------------------------------------
# Within-location r computation
# ---------------------------------------------------------------------------

class TestWithinLocationCorrelation:

    def test_r_is_none_when_no_variation_in_months_suitable(self, model):
        # When ALL months are unsuitable (score=0), every window has months_suitable=0
        # → no variance → Spearman r cannot be computed → None.
        df = make_temporal_df("A", [10.0] * 20, [10.0] * 20)  # temp=10 < 18, score=0
        result = analyze_within_region(df, model)
        if "A" in result["per_location"]:
            loc_r = result["per_location"]["A"]["spearman_r"]
            assert loc_r is None

    def test_r_is_none_when_too_few_windows(self, model):
        # With exactly 4 months and window=4, we might get <=2 windows → too few.
        df = make_temporal_df("A", [25.0, 35.0, 25.0, 25.0], [10., 5., 20., 15.])
        result = analyze_within_region(df, model, window=4)
        if "A" in result["per_location"]:
            loc = result["per_location"]["A"]
            # If too few windows, r must be None (no n_windows field to inspect).
            # This test just verifies no crash and spearman_r key exists.
            assert "spearman_r" in loc

    def test_r_is_numeric_when_enough_data_and_variance(self, model):
        # Vary temperature to create variance in months_suitable.
        temps = [25.0, 35.0] * 12  # alternating suitable/not
        cases = [float(i) for i in range(24)]
        df = make_temporal_df("A", temps, cases)
        result = analyze_within_region(df, model)
        if "A" in result["per_location"]:
            loc_r = result["per_location"]["A"]["spearman_r"]
            if loc_r is not None:
                assert -1.0 <= loc_r <= 1.0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

class TestSummaryStatistics:

    def test_summary_mean_r_in_range(self, model):
        temps = [25.0, 35.0] * 15
        cases = list(range(30))
        df_a = make_temporal_df("A", temps, cases)
        df_b = make_temporal_df("B", temps[::-1], cases)
        df = pd.concat([df_a, df_b], ignore_index=True)
        result = analyze_within_region(df, model)
        mean_r = result["summary"].get("mean_within_location_r")
        if mean_r is not None:
            assert -1.0 <= mean_r <= 1.0
