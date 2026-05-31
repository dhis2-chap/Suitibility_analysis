"""Tests for analysis/annual.py: compute_window_data and analyze_months_suitable.

INTENDED LOGIC:
  compute_window_data():
  - For each location, sorts rows by time.
  - For each row in the location (including rows with NaN cases), builds a
    trailing window of `window` months ending at that row's time.
  - Only rows with non-NaN cases are counted in months_suitable and total_cases
    (NaN rows are excluded from the valid set).
  - Windows with fewer than 9 valid rows are dropped, ensuring every window
    represents a near-complete year and avoiding the build-up confound where
    both months_suitable and total_cases would spuriously co-vary with window size.
  - months_suitable = count of valid rows in window where composite == max_score.
  - total_cases = sum of cases for valid rows in window.
  - Locations are processed independently.
  - Returns empty DataFrame with correct columns if no qualifying windows exist.

  analyze_months_suitable():
  - Calls compute_window_data.
  - Per-location mean statistics (mean_months_suitable, mean_total_cases).
  - Returns early dict with 'note' if fewer than 3 window observations.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data, analyze_months_suitable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standard_model():
    return SuitabilityModel(
        name="standard",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
        ],
    )


def make_single_location_df(n_months, suitable_mask, cases_per_month=10,
                             start="2010-01", location="A"):
    months = pd.date_range(start, periods=n_months, freq="MS")
    rows = []
    for i, month in enumerate(months):
        rows.append({
            "time_period": month.strftime("%Y-%m"),
            "location": location,
            "disease_cases": float(cases_per_month),
            "mean_temperature": 25.0 if suitable_mask[i] else 35.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_window_data — basic window structure
# ---------------------------------------------------------------------------

class TestComputeWindowDataBasic:

    def test_returns_dataframe_with_required_columns(self):
        model = standard_model()
        df = make_single_location_df(15, [True] * 15)
        result = compute_window_data(df, model)
        assert set(result.columns) == {
            "location", "time_period", "months_suitable", "total_cases"
        }

    def test_fraction_suitable_column_does_not_exist(self):
        # fraction_suitable was removed; the column must not appear.
        model = standard_model()
        df = make_single_location_df(15, [True] * 15)
        result = compute_window_data(df, model)
        assert "fraction_suitable" not in result.columns

    def test_returns_empty_df_when_no_qualifying_windows(self):
        # With only 8 rows, every window has n_valid=8 < 9 → all dropped.
        model = standard_model()
        df = make_single_location_df(8, [True] * 8)
        result = compute_window_data(df, model)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {
            "location", "time_period", "months_suitable", "total_cases"
        }

    def test_exactly_8_valid_rows_does_not_qualify(self):
        # n_valid < 9 → dropped. 8 valid rows is below the threshold.
        model = standard_model()
        df = make_single_location_df(8, [True] * 8)
        result = compute_window_data(df, model)
        assert len(result) == 0

    def test_exactly_9_valid_rows_qualifies(self):
        # n_valid >= 9 → qualifies. Use window=9 so the 9th row's window
        # contains exactly 9 valid rows.
        model = standard_model()
        df = make_single_location_df(9, [True] * 9)
        result = compute_window_data(df, model, window=9)
        assert len(result) >= 1

    def test_window_12_requires_at_least_9_valid_months(self):
        # With 8 months and window=12, n_valid <= 8 < 9 for all rows → no output.
        model = standard_model()
        df = make_single_location_df(8, [True] * 8)
        result = compute_window_data(df, model, window=12)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# compute_window_data — months_suitable counting
# ---------------------------------------------------------------------------

class TestMonthsSuitableCounting:

    def test_all_suitable_months_suitable_equals_n_valid(self):
        # When every row is suitable, months_suitable should equal n_valid in every
        # qualifying window. For full 12-month windows, months_suitable == 12.
        model = standard_model()
        df = make_single_location_df(15, [True] * 15)
        result = compute_window_data(df, model, window=12)
        assert len(result) > 0
        # All full 12-month windows: months_suitable == 12
        full_windows = result[result["months_suitable"] == 12.0]
        assert len(full_windows) > 0

    def test_no_suitable_months_suitable_is_zero(self):
        # When no row is suitable, months_suitable must be 0 for every window.
        model = standard_model()
        df = make_single_location_df(15, [False] * 15)
        result = compute_window_data(df, model, window=12)
        assert (result["months_suitable"] == 0.0).all()

    def test_known_mixed_suitability_window(self):
        # 14 months: first 6 unsuitable, next 8 suitable.
        # The window ending at month 14 (window=12) covers months 3-14:
        # months 3-6 = 4 unsuitable, months 7-14 = 8 suitable → months_suitable=8.
        suitable_mask = [False] * 6 + [True] * 8
        model = standard_model()
        df = make_single_location_df(14, suitable_mask)
        result = compute_window_data(df, model, window=12)
        last = result[result["time_period"] == "2011-02"]
        assert len(last) == 1
        assert last.iloc[0]["months_suitable"] == 8.0

    def test_months_suitable_bounded_by_n_valid(self):
        # months_suitable can never exceed the number of valid rows in the window.
        model = standard_model()
        df = make_single_location_df(20, [True, False] * 10)
        result = compute_window_data(df, model, window=12)
        assert len(result) > 0
        assert (result["months_suitable"] <= 12.0).all()
        assert (result["months_suitable"] >= 0.0).all()


# ---------------------------------------------------------------------------
# compute_window_data — total_cases counting
# ---------------------------------------------------------------------------

class TestTotalCasesCounting:

    def test_full_window_total_cases_correct(self):
        # 15 months of 10 cases each. Full 12-month windows sum to 120.
        model = standard_model()
        df = make_single_location_df(15, [True] * 15, cases_per_month=10)
        result = compute_window_data(df, model, window=12)
        assert result["total_cases"].max() == pytest.approx(120.0)

    def test_varying_cases_summed_correctly(self):
        # 14 months with cases 1..14. Window ending at month 14 covers months 3-14:
        # sum = 3+4+…+14 = 102.
        months = pd.date_range("2010-01", periods=14, freq="MS")
        cases = list(range(1, 15))
        df = pd.DataFrame([
            {"time_period": m.strftime("%Y-%m"), "location": "A",
             "disease_cases": float(c), "mean_temperature": 25.0}
            for m, c in zip(months, cases)
        ])
        model = standard_model()
        result = compute_window_data(df, model, window=12)
        last = result[result["time_period"] == "2011-02"]
        assert len(last) == 1
        assert last.iloc[0]["total_cases"] == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# compute_window_data — NaN cases handling
# ---------------------------------------------------------------------------

class TestNaNHandling:

    def test_nan_cases_row_excluded_from_total_cases(self):
        # Insert a NaN at month 7. Window ending at month 14 covers months 3-14
        # (month 7 excluded): n_valid=11, total_cases=110 (not 120).
        months = pd.date_range("2010-01", periods=15, freq="MS")
        cases = [10.0] * 15
        cases[6] = float("nan")  # month 7
        df = pd.DataFrame({
            "time_period": [m.strftime("%Y-%m") for m in months],
            "location": ["A"] * 15,
            "disease_cases": cases,
            "mean_temperature": [25.0] * 15,
        })
        model = standard_model()
        result = compute_window_data(df, model, window=12)
        last = result[result["time_period"] == "2011-03"]
        assert len(last) == 1
        assert last.iloc[0]["total_cases"] == pytest.approx(110.0)

    def test_nan_cases_row_excluded_from_months_suitable(self):
        # NaN row is also excluded from months_suitable count.
        months = pd.date_range("2010-01", periods=15, freq="MS")
        cases = [10.0] * 15
        cases[6] = float("nan")  # month 7 is suitable but NaN cases
        df = pd.DataFrame({
            "time_period": [m.strftime("%Y-%m") for m in months],
            "location": ["A"] * 15,
            "disease_cases": cases,
            "mean_temperature": [25.0] * 15,  # all suitable
        })
        model = standard_model()
        result = compute_window_data(df, model, window=12)
        # Window ending at 2011-03: n_valid=11, all suitable → months_suitable=11
        last = result[result["time_period"] == "2011-03"]
        assert len(last) == 1
        assert last.iloc[0]["months_suitable"] == pytest.approx(11.0)

    def test_nan_reduces_effective_n_valid_below_threshold(self):
        # Many NaN values can drop n_valid below 9, causing window to be dropped.
        months = pd.date_range("2010-01", periods=12, freq="MS")
        cases = [float("nan")] * 4 + [10.0] * 8  # first 4 NaN → n_valid=8 < 9
        df = pd.DataFrame({
            "time_period": [m.strftime("%Y-%m") for m in months],
            "location": ["A"] * 12,
            "disease_cases": cases,
            "mean_temperature": [25.0] * 12,
        })
        model = standard_model()
        result = compute_window_data(df, model, window=12)
        # The window ending at month 12 has only 8 valid rows → dropped.
        assert len(result) == 0

    def test_all_nan_location_produces_no_rows(self):
        months = pd.date_range("2010-01", periods=15, freq="MS")
        df = pd.DataFrame({
            "time_period": [m.strftime("%Y-%m") for m in months],
            "location": ["A"] * 15,
            "disease_cases": [float("nan")] * 15,
            "mean_temperature": [25.0] * 15,
        })
        model = standard_model()
        result = compute_window_data(df, model)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# compute_window_data — multiple locations
# ---------------------------------------------------------------------------

class TestMultipleLocations:

    def test_locations_processed_independently(self):
        df_a = make_single_location_df(15, [True] * 15, location="A")
        df_b = make_single_location_df(15, [True] * 15, location="B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        model = standard_model()
        result = compute_window_data(df, model, window=12)
        assert "A" in set(result["location"])
        assert "B" in set(result["location"])

    def test_location_column_is_string(self):
        df = make_single_location_df(15, [True] * 15)
        result = compute_window_data(df, standard_model(), window=12)
        assert pd.api.types.is_string_dtype(result["location"])


# ---------------------------------------------------------------------------
# analyze_months_suitable — output structure
# ---------------------------------------------------------------------------

class TestAnalyzeMonthsSuitable:

    def test_returns_note_when_too_few_observations(self):
        model = standard_model()
        df = make_single_location_df(8, [True] * 8)
        result = analyze_months_suitable(df, model, window=12)
        assert "note" in result

    def test_contains_expected_keys_with_enough_data(self):
        df_a = make_single_location_df(15, [True] * 15, location="A")
        df_b = make_single_location_df(15, [False] * 15, location="B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        model = standard_model()
        result = analyze_months_suitable(df, model, window=12)
        if "note" not in result:
            assert "per_location" in result
            assert "window_data" in result

    def test_window_data_has_no_fraction_suitable(self):
        # fraction_suitable must not appear in window_data records.
        df_a = make_single_location_df(15, [True] * 15, location="A")
        df_b = make_single_location_df(15, [False] * 15, location="B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        model = standard_model()
        result = analyze_months_suitable(df, model, window=12)
        for record in result.get("window_data", []):
            assert "fraction_suitable" not in record

    def test_per_location_keys(self):
        df = make_single_location_df(15, [True] * 15)
        model = standard_model()
        result = analyze_months_suitable(df, model, window=12)
        if "per_location" in result:
            for loc_key, stats in result["per_location"].items():
                assert "mean_months_suitable" in stats
                assert "mean_total_cases" in stats
                assert "fraction_suitable" not in stats

    def test_window_data_is_list_of_dicts(self):
        df = make_single_location_df(15, [True] * 15)
        result = analyze_months_suitable(df, standard_model(), window=12)
        assert isinstance(result["window_data"], list)


# ---------------------------------------------------------------------------
# _safe helper
# ---------------------------------------------------------------------------

class TestSafeHelper:

    def test_safe_returns_none_for_nan(self):
        from analysis.annual import _safe
        assert _safe(float("nan")) is None

    def test_safe_returns_none_for_inf(self):
        from analysis.annual import _safe
        assert _safe(float("inf")) is None
        assert _safe(float("-inf")) is None

    def test_safe_returns_float_for_valid(self):
        from analysis.annual import _safe
        assert _safe(1.5) == 1.5

    def test_safe_returns_none_for_none(self):
        from analysis.annual import _safe
        assert _safe(None) is None
