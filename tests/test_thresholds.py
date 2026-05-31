"""Tests for ThresholdComponent.

INTENDED LOGIC:
  - A ThresholdComponent evaluates whether values fall within [min_value, max_value].
  - Both bounds are INCLUSIVE (ND-A): value exactly equal to min or max counts as "suitable" → 1.
  - Either bound can be None for one-sided thresholds.
  - When BOTH bounds are None, every value is suitable → 1.
  - The result Series must be int (0 or 1), not bool.
  - NaN values in the input are treated as not meeting the threshold (comparison with NaN is False).
  - Serialization to/from dict must be a perfect roundtrip.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(min_val, max_val, values):
    comp = ThresholdComponent(name="x", column="x", min_value=min_val, max_value=max_val)
    s = pd.Series(values, dtype=float)
    return comp.evaluate(s)


# ---------------------------------------------------------------------------
# Both-bound evaluation
# ---------------------------------------------------------------------------

class TestBothBounds:
    """min_value=18, max_value=32 (temperature-like threshold)."""

    def test_value_in_range_is_1(self):
        # Values strictly inside [18, 32] should be 1.
        result = evaluate(18.0, 32.0, [20.0, 25.0, 30.0])
        assert list(result) == [1, 1, 1]

    def test_value_below_min_is_0(self):
        # Values below the minimum threshold should be 0.
        result = evaluate(18.0, 32.0, [0.0, 10.0, 17.9])
        assert list(result) == [0, 0, 0]

    def test_value_above_max_is_0(self):
        # Values above the maximum threshold should be 0.
        result = evaluate(18.0, 32.0, [32.1, 40.0, 100.0])
        assert list(result) == [0, 0, 0]

    def test_exactly_at_min_bound_is_1(self):
        # Boundary is inclusive: exactly min_value should be 1.
        result = evaluate(18.0, 32.0, [18.0])
        assert list(result) == [1]

    def test_exactly_at_max_bound_is_1(self):
        # Boundary is inclusive: exactly max_value should be 1.
        result = evaluate(18.0, 32.0, [32.0])
        assert list(result) == [1]

    def test_just_below_min_is_0(self):
        # One float epsilon below the minimum should still be 0.
        result = evaluate(18.0, 32.0, [17.999])
        assert list(result) == [0]

    def test_just_above_max_is_0(self):
        # One float epsilon above the maximum should still be 0.
        result = evaluate(18.0, 32.0, [32.001])
        assert list(result) == [0]

    def test_mixed_values(self):
        # Mix of suitable, too-cold, and too-hot values.
        result = evaluate(18.0, 32.0, [15.0, 18.0, 25.0, 32.0, 33.0])
        assert list(result) == [0, 1, 1, 1, 0]

    def test_negative_range(self):
        # Works for ranges that include negative numbers.
        result = evaluate(-5.0, 5.0, [-10.0, -5.0, 0.0, 5.0, 6.0])
        assert list(result) == [0, 1, 1, 1, 0]


# ---------------------------------------------------------------------------
# Min-only evaluation
# ---------------------------------------------------------------------------

class TestMinOnly:
    """min_value=1.0, max_value=None (precipitation-like threshold)."""

    def test_value_above_min_is_1(self):
        # Values >= min are suitable.
        result = evaluate(1.0, None, [1.0, 5.0, 100.0])
        assert list(result) == [1, 1, 1]

    def test_exactly_at_min_is_1(self):
        # Inclusive: exactly at min_value is suitable.
        result = evaluate(1.0, None, [1.0])
        assert list(result) == [1]

    def test_value_below_min_is_0(self):
        # Values < min are not suitable.
        result = evaluate(1.0, None, [0.0, 0.5, 0.999])
        assert list(result) == [0, 0, 0]

    def test_zero_min_value(self):
        # min_value=0 means any non-negative value is suitable.
        result = evaluate(0.0, None, [0.0, 0.001, 1.0])
        assert list(result) == [1, 1, 1]

    def test_negative_min_value(self):
        # Negative min: negative values can be suitable.
        result = evaluate(-10.0, None, [-10.0, -5.0, 0.0])
        assert list(result) == [1, 1, 1]


# ---------------------------------------------------------------------------
# Max-only evaluation
# ---------------------------------------------------------------------------

class TestMaxOnly:
    """max_value=40.0, min_value=None."""

    def test_value_below_max_is_1(self):
        result = evaluate(None, 40.0, [0.0, 20.0, 39.9])
        assert list(result) == [1, 1, 1]

    def test_exactly_at_max_is_1(self):
        # Inclusive: exactly max_value is suitable.
        result = evaluate(None, 40.0, [40.0])
        assert list(result) == [1]

    def test_value_above_max_is_0(self):
        result = evaluate(None, 40.0, [40.1, 50.0, 100.0])
        assert list(result) == [0, 0, 0]


# ---------------------------------------------------------------------------
# No-bounds evaluation
# ---------------------------------------------------------------------------

class TestNoBounds:
    """min_value=None, max_value=None — every value is suitable."""

    def test_all_values_are_1(self):
        # With no bounds, the entire series should be 1.
        result = evaluate(None, None, [0.0, 1.0, -999.0, 1e10])
        assert list(result) == [1, 1, 1, 1]

    def test_empty_series_returns_empty(self):
        result = evaluate(None, None, [])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """NaN values in the input should yield 0 (comparison with NaN is False)."""

    def test_nan_with_both_bounds_returns_0(self):
        result = evaluate(18.0, 32.0, [float("nan")])
        assert list(result) == [0]

    def test_nan_with_min_only_returns_0(self):
        result = evaluate(1.0, None, [float("nan")])
        assert list(result) == [0]

    def test_nan_with_max_only_returns_0(self):
        result = evaluate(None, 40.0, [float("nan")])
        assert list(result) == [0]

    def test_nan_mixed_with_valid_values(self):
        # NaN at index 1; other values normal.
        result = evaluate(18.0, 32.0, [25.0, float("nan"), 25.0])
        assert list(result) == [1, 0, 1]


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------

class TestOutputDtype:
    """Result must be an integer Series (0/1), not boolean."""

    def test_dtype_is_int_for_both_bounds(self):
        result = evaluate(18.0, 32.0, [20.0])
        assert result.dtype in (np.int32, np.int64, int)

    def test_dtype_is_int_for_min_only(self):
        result = evaluate(1.0, None, [5.0])
        assert result.dtype in (np.int32, np.int64, int)

    def test_dtype_is_int_for_no_bounds(self):
        result = evaluate(None, None, [5.0])
        assert result.dtype in (np.int32, np.int64, int)

    def test_values_are_0_or_1_not_bool(self):
        result = evaluate(18.0, 32.0, [20.0, 35.0])
        assert set(result.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------

class TestIndexPreservation:
    """The output index must match the input index."""

    def test_preserves_default_integer_index(self):
        s = pd.Series([20.0, 35.0, 25.0])
        comp = ThresholdComponent(name="t", column="c", min_value=18.0, max_value=32.0)
        result = comp.evaluate(s)
        pd.testing.assert_index_equal(result.index, s.index)

    def test_preserves_custom_index(self):
        s = pd.Series([20.0, 35.0, 25.0], index=[10, 20, 30])
        comp = ThresholdComponent(name="t", column="c", min_value=18.0, max_value=32.0)
        result = comp.evaluate(s)
        pd.testing.assert_index_equal(result.index, s.index)


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    """to_dict / from_dict must be a perfect roundtrip."""

    def test_roundtrip_with_both_bounds(self):
        original = ThresholdComponent("temp", "mean_temperature", 18.0, 32.0)
        restored = ThresholdComponent.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.column == original.column
        assert restored.min_value == original.min_value
        assert restored.max_value == original.max_value

    def test_roundtrip_with_none_bounds(self):
        original = ThresholdComponent("precip", "rainfall", 1.0, None)
        restored = ThresholdComponent.from_dict(original.to_dict())
        assert restored.min_value == original.min_value
        assert restored.max_value is None

    def test_roundtrip_with_all_none(self):
        original = ThresholdComponent("open", "col", None, None)
        restored = ThresholdComponent.from_dict(original.to_dict())
        assert restored.min_value is None
        assert restored.max_value is None

    def test_to_dict_contains_all_fields(self):
        comp = ThresholdComponent("t", "col", 10.0, 20.0)
        d = comp.to_dict()
        assert set(d.keys()) == {"name", "column", "min_value", "max_value"}
        assert d["name"] == "t"
        assert d["column"] == "col"
        assert d["min_value"] == 10.0
        assert d["max_value"] == 20.0

    def test_from_dict_evaluation_matches_original(self):
        # Deserialised component must evaluate identically to the original.
        original = ThresholdComponent("temp", "mean_temperature", 18.0, 32.0)
        restored = ThresholdComponent.from_dict(original.to_dict())
        s = pd.Series([10.0, 18.0, 25.0, 32.0, 35.0])
        pd.testing.assert_series_equal(
            original.evaluate(s), restored.evaluate(s)
        )
