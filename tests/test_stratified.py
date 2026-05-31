"""Tests for analysis/stratified.py: analyze_by_location.

INTENDED LOGIC:
  analyze_by_location():
  - Calls compute_window_data to get rolling window observations per location.
  - For each location: reports mean_months_suitable.
  - Returns an empty dict when no qualifying windows exist.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.stratified import analyze_by_location


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return SuitabilityModel(
        name="test",
        components=[ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0)],
    )


def make_df(locs_temps_cases, n_months=20, start_year=2010):
    """Build a multi-location DataFrame.

    locs_temps_cases: list of (location, base_temp, base_cases) tuples.
    Temperature oscillates sinusoidally; cases similarly.
    """
    months = pd.date_range(f"{start_year}-01", periods=n_months, freq="MS")
    rows = []
    for month_idx, month in enumerate(months):
        for loc, base_temp, base_cases in locs_temps_cases:
            rows.append({
                "time_period": month.strftime("%Y-%m"),
                "location": loc,
                "disease_cases": float(base_cases * (1 + 0.3 * np.sin(month_idx * np.pi / 6))),
                "mean_temperature": base_temp + 5 * np.sin(month_idx * np.pi / 6),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# analyze_by_location — structure
# ---------------------------------------------------------------------------

class TestAnalyzeByLocationStructure:

    def test_returns_dict(self, model):
        df = make_df([("A", 25.0, 50.0)], n_months=20)
        result = analyze_by_location(df, model)
        assert isinstance(result, dict)

    def test_each_location_has_entry(self, model):
        df = make_df([("A", 25.0, 50.0), ("B", 25.0, 30.0)], n_months=20)
        result = analyze_by_location(df, model)
        assert "A" in result
        assert "B" in result

    def test_per_location_required_fields(self, model):
        df = make_df([("A", 25.0, 50.0)], n_months=20)
        result = analyze_by_location(df, model)
        for loc, stats in result.items():
            assert "mean_months_suitable" in stats

    def test_empty_dict_when_no_qualifying_windows(self, model):
        # Only 3 rows of data → no 12-month windows qualify.
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02", "2010-03"],
            "location": ["A", "A", "A"],
            "disease_cases": [10.0, 20.0, 30.0],
            "mean_temperature": [25.0, 25.0, 25.0],
        })
        result = analyze_by_location(df, model)
        assert result == {}


