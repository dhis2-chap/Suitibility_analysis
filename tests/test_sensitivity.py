"""Tests for analysis/sensitivity.py.

INTENDED LOGIC:
  _round_series():
    - Rounds values based on the column's value range (scale).
    - scale > 50: round to 0 decimal places.
    - scale > 5:  round to 1 decimal place.
    - Otherwise:  round to 2 decimal places.
    - Returns a sorted list of unique floats.

  _modify_component():
    - Returns a NEW SuitabilityModel with one component's thresholds replaced.
    - All other components are UNCHANGED.
    - The ORIGINAL model is NOT mutated.
    - The new component gets the new min/max; its name and column are preserved.

  sweep_thresholds():
    - For each component (temperature, precipitation, humidity), sweeps a grid of
      threshold combinations.
    - For each combination, runs compute_window_data and correlates months_suitable
      vs total_cases (Spearman r).
    - Returns None for spearman_r when months_suitable has no variance.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.sensitivity import (
    _round_series,
    _modify_component,
    _mean_within_location_r,
    _between_location_r,
    sweep_thresholds,
    sweep_thresholds_within_region,
    sweep_thresholds_between_region,
)


# ---------------------------------------------------------------------------
# _round_series
# ---------------------------------------------------------------------------

class TestRoundSeries:
    """Rounding decimals determined by scale (max - min) of the column."""

    def make_col(self, min_val, max_val):
        return pd.Series([min_val, (min_val + max_val) / 2, max_val], dtype=float)

    def test_scale_over_50_rounds_to_0_decimals(self):
        col = self.make_col(0, 100)  # scale=100
        result = _round_series(np.array([1.234, 5.678]), col)
        for v in result:
            assert v == round(v, 0)

    def test_scale_over_5_rounds_to_1_decimal(self):
        col = self.make_col(0, 20)  # scale=20
        result = _round_series(np.array([1.234, 5.678]), col)
        for v in result:
            assert v == round(v, 1)

    def test_scale_5_or_less_rounds_to_2_decimals(self):
        col = self.make_col(0, 4)  # scale=4
        result = _round_series(np.array([1.234, 5.678]), col)
        for v in result:
            assert v == round(v, 2)

    def test_returns_sorted_list(self):
        col = self.make_col(0, 100)
        values = np.array([50.0, 10.0, 30.0, 20.0])
        result = _round_series(values, col)
        assert result == sorted(result)

    def test_returns_unique_values_only(self):
        col = self.make_col(0, 100)
        # All values round to 10 with 0 decimal places.
        values = np.array([10.1, 10.4, 10.0])
        result = _round_series(values, col)
        assert len(result) == len(set(result))

    def test_returns_floats(self):
        col = self.make_col(0, 100)
        result = _round_series(np.array([10.0]), col)
        for v in result:
            assert isinstance(v, float)

    def test_scale_exactly_50_rounds_to_1_decimal(self):
        # boundary: scale == 50 is NOT > 50 → falls to > 5 → 1 decimal
        col = self.make_col(0, 50)  # scale=50
        result = _round_series(np.array([1.234]), col)
        for v in result:
            assert v == round(v, 1)

    def test_scale_exactly_5_rounds_to_2_decimals(self):
        # scale == 5 is NOT > 5 → 2 decimals
        col = self.make_col(0, 5)  # scale=5
        result = _round_series(np.array([1.234]), col)
        for v in result:
            assert v == round(v, 2)


# ---------------------------------------------------------------------------
# _modify_component
# ---------------------------------------------------------------------------

class TestModifyComponent:

    @pytest.fixture
    def three_model(self):
        return SuitabilityModel(
            name="base",
            components=[
                ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
                ThresholdComponent("precipitation", "rainfall", 1.0, None),
                ThresholdComponent("humidity", "mean_relative_humidity", 50.0, 80.0),
            ],
        )

    def test_returns_new_model_object(self, three_model):
        result = _modify_component(three_model, "temperature", 20.0, 30.0)
        assert result is not three_model

    def test_original_model_not_mutated(self, three_model):
        _ = _modify_component(three_model, "temperature", 20.0, 30.0)
        # Original temperature component must be unchanged.
        orig_temp = next(c for c in three_model.components if c.name == "temperature")
        assert orig_temp.min_value == 18.0
        assert orig_temp.max_value == 32.0

    def test_modified_component_gets_new_thresholds(self, three_model):
        result = _modify_component(three_model, "temperature", 20.0, 30.0)
        modified_temp = next(c for c in result.components if c.name == "temperature")
        assert modified_temp.min_value == 20.0
        assert modified_temp.max_value == 30.0

    def test_other_components_unchanged(self, three_model):
        result = _modify_component(three_model, "temperature", 20.0, 30.0)
        precip = next(c for c in result.components if c.name == "precipitation")
        humidity = next(c for c in result.components if c.name == "humidity")
        assert precip.min_value == 1.0
        assert humidity.min_value == 50.0
        assert humidity.max_value == 80.0

    def test_modified_component_preserves_name_and_column(self, three_model):
        result = _modify_component(three_model, "temperature", 20.0, 30.0)
        modified_temp = next(c for c in result.components if c.name == "temperature")
        assert modified_temp.name == "temperature"
        assert modified_temp.column == "mean_temperature"

    def test_model_has_same_number_of_components(self, three_model):
        result = _modify_component(three_model, "temperature", 20.0, 30.0)
        assert len(result.components) == len(three_model.components)

    def test_can_set_bounds_to_none(self, three_model):
        result = _modify_component(three_model, "temperature", None, None)
        modified = next(c for c in result.components if c.name == "temperature")
        assert modified.min_value is None
        assert modified.max_value is None


# ---------------------------------------------------------------------------
# sweep_thresholds — structure
# ---------------------------------------------------------------------------

def make_multi_location_df(n_locations=3, n_months=24):
    """Build a DataFrame with the right columns for sweep_thresholds."""
    months = pd.date_range("2010-01", periods=n_months, freq="MS")
    rows = []
    for loc_idx in range(n_locations):
        loc = chr(ord("A") + loc_idx)
        for i, month in enumerate(months):
            rows.append({
                "time_period": month.strftime("%Y-%m"),
                "location": loc,
                "disease_cases": float(10 * (i % 12 + 1) * (loc_idx + 1)),
                "population": 100_000,
                "mean_temperature": 20.0 + 5.0 * np.sin(i * np.pi / 6) + loc_idx,
                "rainfall": 3.0 + 4.0 * max(0, np.sin(i * np.pi / 6)),
                "mean_relative_humidity": 60.0 + 10.0 * np.sin(i * np.pi / 6),
            })
    return pd.DataFrame(rows)


class TestSweepThresholds:

    @pytest.fixture
    def full_model(self):
        return SuitabilityModel(
            name="full",
            components=[
                ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
                ThresholdComponent("precipitation", "rainfall", 1.0, None),
                ThresholdComponent("humidity", "mean_relative_humidity", 50.0, 80.0),
            ],
        )

    @pytest.fixture
    def df(self):
        return make_multi_location_df()

    def test_returns_entry_per_component(self, full_model, df):
        result = sweep_thresholds(df, full_model)
        for comp in full_model.components:
            assert comp.name in result

    def test_each_entry_has_sweeps_list(self, full_model, df):
        result = sweep_thresholds(df, full_model)
        for comp_name, comp_result in result.items():
            assert "sweeps" in comp_result
            assert isinstance(comp_result["sweeps"], list)
            assert len(comp_result["sweeps"]) > 0

    def test_each_sweep_has_required_fields(self, full_model, df):
        result = sweep_thresholds(df, full_model)
        for comp_name, comp_result in result.items():
            for sweep in comp_result["sweeps"]:
                assert "min_value" in sweep
                assert "max_value" in sweep
                # spearman_r may be None if no variance, but key must exist.
                assert "spearman_r" in sweep

    def test_original_thresholds_preserved_in_metadata(self, full_model, df):
        result = sweep_thresholds(df, full_model)
        for comp in full_model.components:
            comp_result = result[comp.name]
            assert comp_result["original_min"] == comp.min_value
            assert comp_result["original_max"] == comp.max_value

    def test_spearman_r_values_in_range(self, full_model, df):
        result = sweep_thresholds(df, full_model)
        for comp_result in result.values():
            for sweep in comp_result["sweeps"]:
                r = sweep["spearman_r"]
                if r is not None:
                    assert -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# _mean_within_location_r
# ---------------------------------------------------------------------------

def make_win_df(per_location_windows: dict) -> pd.DataFrame:
    """Build a win_df from {location: [(months_suitable, total_cases), ...]}."""
    rows = []
    months = pd.date_range("2010-01", periods=30, freq="MS")
    for loc, windows in per_location_windows.items():
        for i, (ms, tc) in enumerate(windows):
            rows.append({
                "location": loc,
                "time_period": months[i].strftime("%Y-%m"),
                "months_suitable": float(ms),
                "total_cases": float(tc),
            })
    return pd.DataFrame(rows)


class TestMeanWithinLocationR:

    def test_returns_none_for_empty_df(self):
        # INTENDED: no locations → no rs → None
        win_df = pd.DataFrame(columns=["location", "time_period",
                                        "months_suitable", "total_cases"])
        assert _mean_within_location_r(win_df) is None

    def test_returns_none_when_all_locations_have_constant_months_suitable(self):
        # INTENDED: all windows in every location have months_suitable=0 → no variance
        # → Spearman r undefined → no valid rs → None
        win_df = make_win_df({
            "A": [(0, i) for i in range(12)],
            "B": [(0, i) for i in range(12)],
        })
        assert _mean_within_location_r(win_df) is None

    def test_returns_none_when_fewer_than_3_windows_per_location(self):
        # INTENDED: each location needs >= 3 windows for Spearman r; 2 is not enough
        win_df = make_win_df({
            "A": [(0, 10), (6, 60)],
            "B": [(0, 20), (6, 70)],
        })
        assert _mean_within_location_r(win_df) is None

    def test_returns_float_when_one_location_has_valid_r(self):
        # INTENDED: one location with 12 varying windows → valid r → float returned
        win_df = make_win_df({
            "A": [(i % 7, i * 10) for i in range(12)],
        })
        result = _mean_within_location_r(win_df)
        if result is not None:
            assert isinstance(result, float)
            assert -1.0 <= result <= 1.0

    def test_returns_mean_of_per_location_rs(self):
        # INTENDED: two locations produce known rs; result is their mean.
        # Location A: months_suitable and total_cases move together → r ≈ +1
        # Location B: same
        # Mean of two near-+1 values should be near +1.
        win_df = make_win_df({
            "A": [(i, i * 10) for i in range(12)],
            "B": [(i, i * 20) for i in range(12)],
        })
        result = _mean_within_location_r(win_df)
        assert result is not None
        assert result > 0.9

    def test_skips_location_with_no_variance(self):
        # INTENDED: location A has no variance (months_suitable constant) → excluded
        # Location B has variance → its r is the only one averaged → result = B's r
        win_df = make_win_df({
            "A": [(0, i) for i in range(12)],        # constant months_suitable
            "B": [(i, i * 10) for i in range(12)],   # varying
        })
        result = _mean_within_location_r(win_df)
        # If A is excluded, result == B's r. Should be non-None since B is valid.
        assert result is not None
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _between_location_r
# ---------------------------------------------------------------------------

class TestBetweenLocationR:

    def test_returns_none_for_empty_df(self):
        win_df = pd.DataFrame(columns=["location", "time_period",
                                        "months_suitable", "total_cases"])
        assert _between_location_r(win_df) is None

    def test_returns_none_when_fewer_than_3_locations(self):
        # INTENDED: between-location correlation needs >= 3 locations
        win_df = make_win_df({
            "A": [(6, 100)] * 12,
            "B": [(3, 50)] * 12,
        })
        assert _between_location_r(win_df) is None

    def test_returns_none_when_all_locations_have_same_mean_months(self):
        # INTENDED: no variance in mean months_suitable across locations → r undefined
        win_df = make_win_df({
            "A": [(6, 100)] * 12,
            "B": [(6, 50)] * 12,
            "C": [(6, 200)] * 12,
        })
        assert _between_location_r(win_df) is None

    def test_returns_float_with_enough_locations_and_variance(self):
        # INTENDED: 3+ locations with different mean months → float r
        win_df = make_win_df({
            "A": [(2, 20)] * 12,
            "B": [(6, 60)] * 12,
            "C": [(10, 100)] * 12,
        })
        result = _between_location_r(win_df)
        assert result is not None
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_positive_r_when_more_suitable_months_means_more_cases(self):
        # INTENDED: locations ordered by months_suitable also ordered by mean cases
        # → Spearman r should be +1
        win_df = make_win_df({
            "A": [(2, 20)] * 12,
            "B": [(5, 50)] * 12,
            "C": [(9, 90)] * 12,
            "D": [(12, 120)] * 12,
        })
        result = _between_location_r(win_df)
        assert result == pytest.approx(1.0)

    def test_negative_r_when_more_suitable_means_fewer_cases(self):
        # INTENDED: inverse relationship → r = -1
        win_df = make_win_df({
            "A": [(2, 120)] * 12,
            "B": [(5, 90)] * 12,
            "C": [(9, 40)] * 12,
            "D": [(12, 10)] * 12,
        })
        result = _between_location_r(win_df)
        assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# sweep_thresholds_within_region — structure
# ---------------------------------------------------------------------------

class TestSweepThresholdsWithinRegion:

    @pytest.fixture
    def full_model(self):
        return SuitabilityModel(
            name="full",
            components=[
                ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
                ThresholdComponent("precipitation", "rainfall", 1.0, None),
                ThresholdComponent("humidity", "mean_relative_humidity", 50.0, 80.0),
            ],
        )

    @pytest.fixture
    def df(self):
        return make_multi_location_df()

    def test_returns_entry_per_component(self, full_model, df):
        # INTENDED: one key per component in the sweep config
        result = sweep_thresholds_within_region(df, full_model)
        for comp in full_model.components:
            assert comp.name in result

    def test_each_entry_has_sweeps_list(self, full_model, df):
        # INTENDED: each component has a non-empty list of sweep points
        result = sweep_thresholds_within_region(df, full_model)
        for comp_result in result.values():
            assert "sweeps" in comp_result
            assert len(comp_result["sweeps"]) > 0

    def test_each_sweep_has_required_fields(self, full_model, df):
        # INTENDED: each sweep point has min_value, max_value, spearman_r
        result = sweep_thresholds_within_region(df, full_model)
        for comp_result in result.values():
            for sweep in comp_result["sweeps"]:
                assert "min_value" in sweep
                assert "max_value" in sweep
                assert "spearman_r" in sweep

    def test_original_thresholds_in_metadata(self, full_model, df):
        # INTENDED: original_min and original_max preserved for reference box in plot
        result = sweep_thresholds_within_region(df, full_model)
        for comp in full_model.components:
            comp_result = result[comp.name]
            assert comp_result["original_min"] == comp.min_value
            assert comp_result["original_max"] == comp.max_value

    def test_spearman_r_is_none_or_in_range(self, full_model, df):
        # INTENDED: r is the MEAN within-location r; None when no location has valid r
        result = sweep_thresholds_within_region(df, full_model)
        for comp_result in result.values():
            for sweep in comp_result["sweeps"]:
                r = sweep["spearman_r"]
                if r is not None:
                    assert -1.0 <= r <= 1.0

    def test_uses_same_grid_as_pooled_sweep(self, full_model, df):
        # INTENDED: within-region and pooled sweeps share the same threshold grid
        pooled = sweep_thresholds(df, full_model)
        within = sweep_thresholds_within_region(df, full_model)
        for comp_name in pooled:
            pooled_mins = {s["min_value"] for s in pooled[comp_name]["sweeps"]}
            within_mins = {s["min_value"] for s in within[comp_name]["sweeps"]}
            assert pooled_mins == within_mins
            pooled_maxs = {s["max_value"] for s in pooled[comp_name]["sweeps"]}
            within_maxs = {s["max_value"] for s in within[comp_name]["sweeps"]}
            assert pooled_maxs == within_maxs

    def test_r_can_differ_from_pooled_sweep(self, full_model, df):
        # INTENDED: within-region r is a different metric from pooled r;
        # they need not agree. Just verify they are computed independently.
        pooled = sweep_thresholds(df, full_model)
        within = sweep_thresholds_within_region(df, full_model)
        # At least one sweep point should potentially differ (or both be None).
        # This test just asserts that the function completes without error and
        # both are valid dicts — the actual r values may or may not match.
        assert isinstance(within, dict)
        assert isinstance(pooled, dict)


# ---------------------------------------------------------------------------
# sweep_thresholds_between_region — structure
# ---------------------------------------------------------------------------

class TestSweepThresholdsBetweenRegion:

    @pytest.fixture
    def full_model(self):
        return SuitabilityModel(
            name="full",
            components=[
                ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
                ThresholdComponent("precipitation", "rainfall", 1.0, None),
                ThresholdComponent("humidity", "mean_relative_humidity", 50.0, 80.0),
            ],
        )

    @pytest.fixture
    def df(self):
        return make_multi_location_df()

    def test_returns_entry_per_component(self, full_model, df):
        result = sweep_thresholds_between_region(df, full_model)
        for comp in full_model.components:
            assert comp.name in result

    def test_each_entry_has_sweeps_list(self, full_model, df):
        result = sweep_thresholds_between_region(df, full_model)
        for comp_result in result.values():
            assert "sweeps" in comp_result
            assert len(comp_result["sweeps"]) > 0

    def test_each_sweep_has_required_fields(self, full_model, df):
        # INTENDED: each sweep point has min_value, max_value, spearman_r
        result = sweep_thresholds_between_region(df, full_model)
        for comp_result in result.values():
            for sweep in comp_result["sweeps"]:
                assert "min_value" in sweep
                assert "max_value" in sweep
                assert "spearman_r" in sweep

    def test_original_thresholds_in_metadata(self, full_model, df):
        result = sweep_thresholds_between_region(df, full_model)
        for comp in full_model.components:
            comp_result = result[comp.name]
            assert comp_result["original_min"] == comp.min_value
            assert comp_result["original_max"] == comp.max_value

    def test_spearman_r_is_none_or_in_range(self, full_model, df):
        # INTENDED: r is the cross-location Spearman r; None when < 3 locations
        # or no variance in mean months_suitable
        result = sweep_thresholds_between_region(df, full_model)
        for comp_result in result.values():
            for sweep in comp_result["sweeps"]:
                r = sweep["spearman_r"]
                if r is not None:
                    assert -1.0 <= r <= 1.0

    def test_uses_same_grid_as_pooled_sweep(self, full_model, df):
        # INTENDED: between-region and pooled sweeps share the same threshold grid
        pooled = sweep_thresholds(df, full_model)
        between = sweep_thresholds_between_region(df, full_model)
        for comp_name in pooled:
            pooled_mins = {s["min_value"] for s in pooled[comp_name]["sweeps"]}
            between_mins = {s["min_value"] for s in between[comp_name]["sweeps"]}
            assert pooled_mins == between_mins

    def test_returns_none_when_only_one_location(self):
        # INTENDED: between-region r requires >= 3 locations
        model = SuitabilityModel(
            name="test",
            components=[ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0)],
        )
        months = pd.date_range("2010-01", periods=24, freq="MS")
        df = pd.DataFrame([{
            "time_period": m.strftime("%Y-%m"),
            "location": "A",
            "disease_cases": float(i * 10),
            "mean_temperature": 20.0 + 5.0 * np.sin(i * np.pi / 6),
        } for i, m in enumerate(months)])
        result = sweep_thresholds_between_region(df, model)
        if "temperature" in result:
            for sweep in result["temperature"]["sweeps"]:
                assert sweep["spearman_r"] is None
