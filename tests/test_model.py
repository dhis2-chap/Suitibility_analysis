"""Tests for SuitabilityModel.

INTENDED LOGIC:
  - SuitabilityModel is a named container of ThresholdComponents.
  - compute_component_scores() evaluates each component against its designated
    column and returns a DataFrame with one binary (0/1) column per component.
  - compute_composite_score() returns the row-wise sum of all component scores.
    A value of N means all N thresholds are met simultaneously.
  - required_columns() returns the unique list of column names needed.
  - to_dict / from_dict are a perfect serialisation roundtrip.
  - An empty model (no components) is valid: composite score is always 0.
"""

import pandas as pd
import numpy as np
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_component_model():
    return SuitabilityModel(
        name="two_comp",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
            ThresholdComponent("precipitation", "rainfall", 1.0, None),
        ],
    )


@pytest.fixture
def three_component_model():
    return SuitabilityModel(
        name="three_comp",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
            ThresholdComponent("precipitation", "rainfall", 1.0, None),
            ThresholdComponent("humidity", "mean_relative_humidity", 50.0, 80.0),
        ],
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "mean_temperature": [25.0, 10.0, 25.0, 10.0],
        "rainfall":         [5.0,  5.0,  0.5,  0.5],
        "mean_relative_humidity": [65.0, 65.0, 65.0, 65.0],
    })


# ---------------------------------------------------------------------------
# compute_component_scores
# ---------------------------------------------------------------------------

class TestComputeComponentScores:

    def test_returns_dataframe_with_one_column_per_component(self, two_component_model, sample_df):
        # There must be exactly as many columns as components.
        scores = two_component_model.compute_component_scores(sample_df)
        assert isinstance(scores, pd.DataFrame)
        assert len(scores.columns) == 2

    def test_column_names_match_component_names(self, two_component_model, sample_df):
        # Column names must be the component names, not the column names.
        scores = two_component_model.compute_component_scores(sample_df)
        assert list(scores.columns) == ["temperature", "precipitation"]

    def test_index_matches_input_df(self, two_component_model):
        df = pd.DataFrame(
            {"mean_temperature": [25.0], "rainfall": [5.0]},
            index=[42],
        )
        scores = two_component_model.compute_component_scores(df)
        pd.testing.assert_index_equal(scores.index, df.index)

    def test_correct_binary_values_for_two_components(self, two_component_model, sample_df):
        # Row 0: temp=25 (ok), rain=5 (ok) → [1, 1]
        # Row 1: temp=10 (low), rain=5 (ok) → [0, 1]
        # Row 2: temp=25 (ok), rain=0.5 (low) → [1, 0]
        # Row 3: temp=10 (low), rain=0.5 (low) → [0, 0]
        scores = two_component_model.compute_component_scores(sample_df)
        assert list(scores["temperature"]) == [1, 0, 1, 0]
        assert list(scores["precipitation"]) == [1, 1, 0, 0]

    def test_empty_model_returns_empty_dataframe(self, sample_df):
        # A model with no components should return an empty-column DataFrame
        # with the same index as the input.
        model = SuitabilityModel(name="empty", components=[])
        scores = model.compute_component_scores(sample_df)
        assert len(scores.columns) == 0
        assert len(scores) == len(sample_df)


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

class TestComputeCompositeScore:

    def test_returns_series(self, two_component_model, sample_df):
        composite = two_component_model.compute_composite_score(sample_df)
        assert isinstance(composite, pd.Series)

    def test_correct_composite_values(self, two_component_model, sample_df):
        # Row 0: [1,1] → sum=2  (all met)
        # Row 1: [0,1] → sum=1
        # Row 2: [1,0] → sum=1
        # Row 3: [0,0] → sum=0  (none met)
        composite = two_component_model.compute_composite_score(sample_df)
        assert list(composite) == [2, 1, 1, 0]

    def test_max_possible_score_equals_n_components(self, three_component_model):
        df = pd.DataFrame({
            "mean_temperature": [25.0],
            "rainfall": [5.0],
            "mean_relative_humidity": [65.0],
        })
        composite = three_component_model.compute_composite_score(df)
        assert composite.iloc[0] == 3

    def test_empty_model_score_is_zero(self, sample_df):
        model = SuitabilityModel(name="empty", components=[])
        composite = model.compute_composite_score(sample_df)
        assert list(composite) == [0, 0, 0, 0]

    def test_score_range_is_0_to_n_components(self, three_component_model):
        # Build data spanning all possible scores
        df = pd.DataFrame({
            "mean_temperature": [25.0, 10.0, 25.0, 25.0],
            "rainfall":         [5.0,  5.0,  0.5,  5.0],
            "mean_relative_humidity": [65.0, 65.0, 65.0, 45.0],  # last row: humidity fail
        })
        composite = three_component_model.compute_composite_score(df)
        assert composite.min() >= 0
        assert composite.max() <= 3

    def test_composite_is_sum_of_component_scores(self, three_component_model, sample_df):
        # Explicit verification: composite must equal manual row-sum of components.
        df = pd.DataFrame({
            "mean_temperature": [25.0, 10.0],
            "rainfall": [5.0, 5.0],
            "mean_relative_humidity": [65.0, 45.0],
        })
        scores = three_component_model.compute_component_scores(df)
        expected = scores.sum(axis=1)
        composite = three_component_model.compute_composite_score(df)
        pd.testing.assert_series_equal(composite, expected)


# ---------------------------------------------------------------------------
# required_columns
# ---------------------------------------------------------------------------

class TestRequiredColumns:

    def test_returns_all_component_columns(self, three_component_model):
        cols = three_component_model.required_columns()
        assert set(cols) == {"mean_temperature", "rainfall", "mean_relative_humidity"}

    def test_order_matches_component_order(self, three_component_model):
        cols = three_component_model.required_columns()
        assert cols == ["mean_temperature", "rainfall", "mean_relative_humidity"]

    def test_empty_model_returns_empty_list(self):
        model = SuitabilityModel(name="empty", components=[])
        assert model.required_columns() == []

    def test_single_component(self):
        model = SuitabilityModel(
            name="single",
            components=[ThresholdComponent("t", "mean_temperature", 18.0, 32.0)],
        )
        assert model.required_columns() == ["mean_temperature"]


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------

class TestSerialisationRoundtrip:

    def test_to_dict_contains_name_and_components(self, two_component_model):
        d = two_component_model.to_dict()
        assert "name" in d
        assert "components" in d
        assert d["name"] == "two_comp"
        assert len(d["components"]) == 2

    def test_from_dict_restores_model_name(self, two_component_model):
        restored = SuitabilityModel.from_dict(two_component_model.to_dict())
        assert restored.name == two_component_model.name

    def test_from_dict_restores_all_components(self, three_component_model):
        restored = SuitabilityModel.from_dict(three_component_model.to_dict())
        assert len(restored.components) == 3

    def test_from_dict_restores_component_attributes(self, three_component_model):
        restored = SuitabilityModel.from_dict(three_component_model.to_dict())
        for orig, rest in zip(three_component_model.components, restored.components):
            assert rest.name == orig.name
            assert rest.column == orig.column
            assert rest.min_value == orig.min_value
            assert rest.max_value == orig.max_value

    def test_roundtrip_produces_identical_scores(self, three_component_model):
        df = pd.DataFrame({
            "mean_temperature": [20.0, 35.0],
            "rainfall": [5.0, 0.1],
            "mean_relative_humidity": [65.0, 45.0],
        })
        restored = SuitabilityModel.from_dict(three_component_model.to_dict())
        pd.testing.assert_series_equal(
            three_component_model.compute_composite_score(df),
            restored.compute_composite_score(df),
        )
