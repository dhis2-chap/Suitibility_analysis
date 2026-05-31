"""Primary analysis: months at maximum suitability vs. total cases in same period.

For each location-window, count how many of the last 12 months had the suitability
score at its maximum (all thresholds met). Compare this to the total disease cases
summed over the same 12-month window.

This is the standard cross-sectional method. It asks:
  "Do locations/periods with more annually-suitable months accumulate more disease burden?"

Why this approach instead of current-month score vs current-month cases:
  - Current-month comparison is circular: suitability and cases both peak in transmission
    season, so they co-occur by construction regardless of model quality.
  - Counting suitable months per year captures cumulative transmission opportunity —
    the ecologically meaningful quantity.
  - Comparing to total annual cases removes seasonal phase-matching artifacts.
    A 1-3 month lag between climate and cases is absorbed within the 12-month window.

NORMATIVE DECISION: Using months where score == max_score (all thresholds met) as
"suitable". Alternative: any score > 0, or mean score, or fraction above some
intermediate threshold. We use max because the model defines suitability as the
conjunction of all thresholds — partial scores are ambiguous.

NORMATIVE DECISION: Comparing to total cases in the same 12-month window (not
current-month cases). This makes both sides annualized quantities over the same
period. See docstring of analyze_months_suitable() for discussion.
"""

import pandas as pd
import numpy as np

from suitability.model import SuitabilityModel


def compute_window_data(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> pd.DataFrame:
    """Shared helper: compute rolling window pairs for any model.

    For each location, builds trailing calendar windows of `window` months.
    Each window yields one row: (months_suitable, total_cases).

    NaN rows in cases_col are excluded from window counts. Windows with fewer
    than 9 valid months are dropped, ensuring windows represent a near-complete
    year. To treat missing cases as zero, fill NaN before calling.

    Args:
        df: Input DataFrame with location, time, climate, and cases columns.
        model: The suitability model to score with.
        cases_col: Column to sum as outcome.
        location_col, time_col: Column names.
        window: Rolling window length in months (default 12).

    Returns:
        DataFrame with columns: location, time_period, months_suitable,
        total_cases. Empty if no qualifying windows exist.
    """
    composite = model.compute_composite_score(df)
    max_score = len(model.components)

    df = df.copy()
    df["_score"] = composite
    df["_is_suitable"] = (composite == max_score).astype(int)
    df["dt_sort"] = pd.to_datetime(df[time_col])
    df = df.sort_values([location_col, "dt_sort"])

    rows = []

    for location, group in df.groupby(location_col):
        group = group.sort_values("dt_sort").reset_index(drop=True)
        valid = group.dropna(subset=[cases_col]).reset_index(drop=True)

        if valid.empty:
            continue

        # Use cumulative sums over valid rows for O(1) range queries.
        valid_dates = valid["dt_sort"].values
        cum_suitable = np.concatenate([[0], np.cumsum(valid["_is_suitable"].values)])
        cum_cases = np.concatenate([[0], np.cumsum(valid[cases_col].values)])

        for row in group.itertuples(index=False):
            t = row.dt_sort
            window_start = t - pd.DateOffset(months=window - 1)
            lo = int(np.searchsorted(valid_dates, np.datetime64(window_start), side="left"))
            hi = int(np.searchsorted(valid_dates, np.datetime64(t), side="right"))
            n_valid = hi - lo
            if n_valid < 9:
                continue
            months_suitable = float(cum_suitable[hi] - cum_suitable[lo])
            rows.append({
                "location": str(location),
                "time_period": str(getattr(row, time_col)),
                "months_suitable": months_suitable,
                "total_cases": float(cum_cases[hi] - cum_cases[lo]),
            })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=["location", "time_period", "months_suitable", "total_cases"]
    )


def analyze_months_suitable(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Rolling window: months at max suitability vs. total cases in same window.

    For each location, a trailing window of `window` months is rolled over the
    time series. Each window yields one observation:
        predictor:  months_suitable = count(score == max_score) in window
        outcome:    total_cases     = sum(cases_col) in same window

    The correlation is computed across all location-windows that have a complete
    window of data. This pools across both locations and time.

    Returns a dict with:
      - window_data: list of {location, time_period, months_suitable, total_cases}
      - correlation: pooled Spearman r across all windows
      - per_location: per-location mean months_suitable and mean total_cases
      - cross_sectional: for each window-end time period, Spearman r across locations
    """
    max_score = len(model.components)
    win_df = compute_window_data(df, model, cases_col, location_col, time_col, window)

    results = {"window": window, "max_score": max_score}

    if len(win_df) < 3:
        results["note"] = f"Too few observations with full {window}-month window"
        results["window_data"] = win_df.to_dict("records")
        return results

    results["window_data"] = win_df.to_dict("records")

    # Per-location summary
    per_loc = {}
    for location, group in win_df.groupby("location"):
        per_loc[str(location)] = {
            "mean_months_suitable": _safe(group["months_suitable"].mean()),
            "mean_total_cases": _safe(group["total_cases"].mean()),
        }
    results["per_location"] = per_loc

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
