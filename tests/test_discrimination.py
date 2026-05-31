"""Tests for analysis/discrimination.py: analyze_discrimination.

INTENDED LOGIC:
  A pure frequency table of composite suitability score levels. For each level
  0..max_score, counts how many observation-months land at that level and what
  percentage of the total they represent. Disease cases are irrelevant — this
  only asks "how often does the model fire at each score level?"

  - All score levels 0..max_score always appear, even with n=0.
  - pct = n / total_observations * 100.
  - pct values sum to 100; n values sum to len(df).
  - Only the "distribution" key is returned.
"""

import pandas as pd
import pytest

from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel
from analysis.discrimination import analyze_discrimination


@pytest.fixture
def model():
    return SuitabilityModel(
        name="test",
        components=[
            ThresholdComponent("temperature", "mean_temperature", 18.0, 32.0),
            ThresholdComponent("precipitation", "rainfall", 1.0, None),
        ],
    )


def make_df(temps, rains):
    n = len(temps)
    return pd.DataFrame({
        "mean_temperature": temps,
        "rainfall":         rains,
        "disease_cases":    [float("nan")] * n,  # cases irrelevant
        "location":         [chr(65 + i) for i in range(n)],
        "time_period":      [f"2010-{i+1:02d}" for i in range(n)],
    })


class TestScoreDistribution:

    def test_returns_only_distribution_key(self, model):
        result = analyze_discrimination(make_df([25.0], [5.0]), model)
        assert set(result.keys()) == {"distribution"}

    def test_distribution_covers_all_score_levels(self, model):
        # Levels 0, 1, 2 must all appear even if one has n=0.
        df = make_df([25.0, 10.0, 25.0, 10.0], [5.0, 5.0, 0.5, 0.5])
        result = analyze_discrimination(df, model)
        score_levels = {d["score"] for d in result["distribution"]}
        assert score_levels == {0, 1, 2}

    def test_n_sums_to_len_df(self, model):
        df = make_df([25.0, 10.0, 25.0], [5.0, 5.0, 0.5])
        result = analyze_discrimination(df, model)
        assert sum(d["n"] for d in result["distribution"]) == len(df)

    def test_pct_sums_to_100(self, model):
        df = make_df([25.0, 10.0], [5.0, 0.5])
        result = analyze_discrimination(df, model)
        total_pct = sum(d["pct"] for d in result["distribution"])
        assert total_pct == pytest.approx(100.0)

    def test_pct_equals_n_over_total_times_100(self, model):
        df = make_df([25.0, 10.0, 25.0, 10.0], [5.0, 5.0, 0.5, 0.5])
        result = analyze_discrimination(df, model)
        n_total = len(df)
        for d in result["distribution"]:
            assert d["pct"] == pytest.approx(d["n"] / n_total * 100)

    def test_n_correct_for_known_scores(self, model):
        # Row 0: temp=25 (ok), rain=5 (ok) → score=2
        # Row 1: temp=10 (fail), rain=5 (ok) → score=1
        # Row 2: temp=10 (fail), rain=0.5 (fail) → score=0
        df = make_df([25.0, 10.0, 10.0], [5.0, 5.0, 0.5])
        result = analyze_discrimination(df, model)
        dist = {d["score"]: d for d in result["distribution"]}
        assert dist[2]["n"] == 1
        assert dist[1]["n"] == 1
        assert dist[0]["n"] == 1

    def test_level_with_zero_observations_has_n_zero(self, model):
        # temp=10 fails temperature → score never reaches 2.
        df = make_df([10.0, 10.0, 10.0], [5.0, 5.0, 5.0])
        result = analyze_discrimination(df, model)
        dist = {d["score"]: d for d in result["distribution"]}
        assert dist[2]["n"] == 0
        assert dist[2]["pct"] == pytest.approx(0.0)

    def test_no_mean_cases_or_median_cases_field(self, model):
        # Disease cases must not appear in the output at all.
        df = make_df([25.0, 10.0], [5.0, 5.0])
        result = analyze_discrimination(df, model)
        for d in result["distribution"]:
            assert "mean_cases" not in d
            assert "median_cases" not in d
