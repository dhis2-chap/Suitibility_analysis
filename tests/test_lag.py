"""Tests for analysis/lag.py: create_lagged_df and run_lag_sweep.

INTENDED LOGIC:
  create_lagged_df():
  - lag=0: returns an identical copy of the DataFrame (no structural changes).
  - lag=N: each row gets climate data from N months EARLIER (same location).
    - disease_cases, population, incidence stay at the CURRENT row's time.
    - climate columns (everything not in the non-climate exclusion set) come from
      the row N months earlier in the same location.
    - Rows where the lagged data doesn't exist are DROPPED (inner join).
    - The resulting DataFrame has no _dt or _climate_dt helper columns.
  - Non-climate columns identified by exclusion from:
    {"disease_cases", "population", "incidence", location_col, time_col,
     "_dt", "_climate_dt"}.
  - The original DataFrame is NOT modified.

  run_lag_sweep():
  - Runs CorrelationAnalysis at each lag from 0 to max_lag (inclusive).
  - "best_lag" is the lag with the highest SIGNED (not absolute) Spearman r.
    A negative r is not "best" even if it has the highest magnitude.
  - summary is a list of dicts with one entry per lag.
  - continuous_by_lag tracks per-variable Spearman r across lags.
  - Skips lags that produce fewer than 5 rows.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.lag import create_lagged_df, run_lag_sweep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return SuitabilityModel(
        name="test",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
        ],
    )


def make_df(n_months=24, n_locations=2):
    """Build a multi-location monthly DataFrame with known climate and cases."""
    months = pd.date_range("2010-01", periods=n_months, freq="MS")
    rows = []
    for loc_idx in range(n_locations):
        loc = chr(ord("A") + loc_idx)
        for i, month in enumerate(months):
            rows.append({
                "time_period": month.strftime("%Y-%m"),
                "location": loc,
                "disease_cases": float(10 * (i % 12 + 1)),
                "population": 100_000,
                "mean_temperature": 20.0 + loc_idx + i * 0.1,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# create_lagged_df — lag=0
# ---------------------------------------------------------------------------

class TestCreateLaggedDfLagZero:

    def test_lag_0_returns_same_shape(self):
        df = make_df()
        result = create_lagged_df(df, 0)
        assert len(result) == len(df)

    def test_lag_0_returns_copy_not_same_object(self):
        df = make_df()
        result = create_lagged_df(df, 0)
        assert result is not df

    def test_lag_0_does_not_modify_original(self):
        df = make_df()
        original_len = len(df)
        original_cols = list(df.columns)
        _ = create_lagged_df(df, 0)
        assert len(df) == original_len
        assert list(df.columns) == original_cols

    def test_lag_0_preserves_all_values(self):
        df = make_df(n_months=6, n_locations=1)
        result = create_lagged_df(df, 0)
        # All climate values should be unchanged.
        pd.testing.assert_series_equal(
            df["mean_temperature"].reset_index(drop=True),
            result["mean_temperature"].reset_index(drop=True),
        )

    def test_lag_0_no_helper_columns_in_output(self):
        df = make_df()
        result = create_lagged_df(df, 0)
        assert "_dt" not in result.columns
        assert "_climate_dt" not in result.columns


# ---------------------------------------------------------------------------
# create_lagged_df — lag=N
# ---------------------------------------------------------------------------

class TestCreateLaggedDfLagN:

    def test_lag_drops_first_n_rows_per_location(self):
        # With lag=1, the first month of each location has no prior data → dropped.
        df = make_df(n_months=6, n_locations=2)
        result = create_lagged_df(df, lag_months=1)
        # Each location loses 1 row → total 2*(6-1)=10 rows
        assert len(result) == 10

    def test_lag_2_drops_first_2_rows_per_location(self):
        df = make_df(n_months=6, n_locations=2)
        result = create_lagged_df(df, lag_months=2)
        assert len(result) == 2 * (6 - 2)

    def test_climate_comes_from_lagged_month(self):
        # Build a 3-row single-location df where temperature increases by 1 each month.
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02", "2010-03"],
            "location": ["A", "A", "A"],
            "disease_cases": [10.0, 20.0, 30.0],
            "population": [100_000] * 3,
            "mean_temperature": [20.0, 21.0, 22.0],
        })
        result = create_lagged_df(df, lag_months=1)
        # Lag=1: row for 2010-02 gets climate from 2010-01 (temp=20.0)
        #        row for 2010-03 gets climate from 2010-02 (temp=21.0)
        assert len(result) == 2
        temps = result.sort_values("time_period")["mean_temperature"].tolist()
        assert temps == pytest.approx([20.0, 21.0])

    def test_cases_stay_at_current_time(self):
        # Cases should NOT be lagged — they remain at the original row's time.
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02", "2010-03"],
            "location": ["A", "A", "A"],
            "disease_cases": [10.0, 20.0, 30.0],
            "population": [100_000] * 3,
            "mean_temperature": [20.0, 21.0, 22.0],
        })
        result = create_lagged_df(df, lag_months=1)
        cases = result.sort_values("time_period")["disease_cases"].tolist()
        # 2010-02 keeps cases=20, 2010-03 keeps cases=30
        assert cases == pytest.approx([20.0, 30.0])

    def test_no_helper_columns_in_output(self):
        df = make_df(n_months=6)
        result = create_lagged_df(df, lag_months=1)
        assert "_dt" not in result.columns
        assert "_climate_dt" not in result.columns

    def test_does_not_modify_original_df(self):
        df = make_df(n_months=6)
        original_cols = list(df.columns)
        _ = create_lagged_df(df, lag_months=1)
        assert list(df.columns) == original_cols

    def test_population_stays_at_current_time(self):
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02", "2010-03"],
            "location": ["A", "A", "A"],
            "disease_cases": [10.0, 20.0, 30.0],
            "population": [100_000.0, 200_000.0, 300_000.0],
            "mean_temperature": [20.0, 21.0, 22.0],
        })
        result = create_lagged_df(df, lag_months=1)
        pops = result.sort_values("time_period")["population"].tolist()
        # 2010-02 keeps pop=200_000, 2010-03 keeps pop=300_000
        assert pops == pytest.approx([200_000.0, 300_000.0])

    def test_cross_location_mixing_does_not_occur(self):
        # Lag must only look up climate within the same location.
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02", "2010-01", "2010-02"],
            "location":    ["A",       "A",       "B",       "B"],
            "disease_cases": [10.0, 20.0, 30.0, 40.0],
            "population": [100_000] * 4,
            "mean_temperature": [20.0, 21.0, 99.0, 100.0],
        })
        result = create_lagged_df(df, lag_months=1)
        # A at 2010-02 should get temp from A at 2010-01 (=20), NOT from B.
        a_row = result[(result["location"] == "A") & (result["time_period"] == "2010-02")]
        assert len(a_row) == 1
        assert a_row.iloc[0]["mean_temperature"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# run_lag_sweep — structure
# ---------------------------------------------------------------------------

class TestRunLagSweep:

    def test_returns_required_keys(self, model):
        df = make_df(n_months=24, n_locations=3)
        result = run_lag_sweep(df, model, max_lag=2)
        for key in ["summary", "best_lag", "continuous_by_lag"]:
            assert key in result

    def test_summary_length_matches_max_lag_plus_1(self, model):
        df = make_df(n_months=24, n_locations=3)
        result = run_lag_sweep(df, model, max_lag=2)
        # Summary has one entry per lag (if data is sufficient).
        assert len(result["summary"]) == 3

    def test_summary_contains_lag_field(self, model):
        df = make_df(n_months=24, n_locations=3)
        result = run_lag_sweep(df, model, max_lag=2)
        for row in result["summary"]:
            assert "lag" in row

# ---------------------------------------------------------------------------
# run_lag_sweep — best_lag logic (signed Spearman r)
# ---------------------------------------------------------------------------

class TestBestLagLogic:

    def test_best_lag_is_lag_with_highest_signed_r(self, model):
        # The best lag must be the one with the highest signed Spearman r,
        # not the highest absolute r.
        df = make_df(n_months=24, n_locations=3)
        result = run_lag_sweep(df, model, max_lag=3)
        summary = [s for s in result["summary"] if s["spearman_r"] is not None]
        if summary:
            expected_best = max(summary, key=lambda s: s["spearman_r"])["lag"]
            assert result["best_lag"] == expected_best

    def test_best_lag_is_zero_when_no_valid_correlations(self):
        # When no correlation can be computed, default best_lag is 0.
        # Build data with fewer than 5 rows to force all lags to be skipped.
        model = SuitabilityModel(
            name="test",
            components=[ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0)],
        )
        df = pd.DataFrame({
            "time_period": ["2010-01", "2010-02"],
            "location": ["A", "A"],
            "disease_cases": [10.0, 20.0],
            "population": [100_000, 100_000],
            "mean_temperature": [25.0, 25.0],
        })
        result = run_lag_sweep(df, model, max_lag=1)
        assert result["best_lag"] == 0
