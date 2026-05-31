"""Lag analysis: how does the climate-to-cases correlation change with time lag?

Climate conditions affect malaria cases with a delay (mosquito breeding,
parasite incubation, symptom onset, health facility visit). The typical
range in the literature is 1-3 months. This module runs the correlation
analysis at multiple lags to find which lag best aligns suitability
scores with observed cases.

Approach: for lag=N, we match each location's climate data from month T
with disease cases from month T+N. Within each location, we shift the
climate/score columns backward by N months relative to cases.
"""

import pandas as pd
from scipy import stats

from suitability.model import SuitabilityModel
from analysis.correlation import CorrelationAnalysis


def create_lagged_df(
    df: pd.DataFrame,
    lag_months: int,
    location_col: str = "location",
    time_col: str = "time_period",
) -> pd.DataFrame:
    """Create a DataFrame where climate columns are lagged relative to cases.

    For lag=N, each row's climate values come from N months earlier,
    while disease_cases and population stay at the original time.
    Rows where the lagged climate data doesn't exist are dropped.

    Returns a new DataFrame (original is not modified).
    """
    if lag_months == 0:
        return df.copy()

    df = df.copy()
    df["_dt"] = pd.to_datetime(df[time_col])

    # For each row, compute the time_period we need climate data from
    df["_climate_dt"] = df["_dt"] - pd.DateOffset(months=lag_months)

    # Build a lookup of climate data keyed by (location, month)
    # NORMATIVE DECISION (ND-D): Climate columns are identified by exclusion — any
    # column not in the fixed non-climate set is treated as a climate variable.
    # This works correctly for well-formed data but will silently lag unexpected
    # columns (e.g., a "notes" column) if they appear in the DataFrame.
    # Confirm that input DataFrames contain only the expected columns.
    non_climate = {
        "disease_cases", "population", "incidence",
        location_col, time_col, "_dt", "_climate_dt",
    }
    climate_cols = [c for c in df.columns if c not in non_climate]

    climate_lookup = df.set_index([location_col, "_dt"])[climate_cols]

    # For each row, look up climate from the lagged month
    merged = df[[location_col, time_col, "_dt", "_climate_dt"]].copy()

    # Keep non-climate columns from original time
    for col in ["disease_cases", "population", "incidence"]:
        if col in df.columns:
            merged[col] = df[col].values

    # Join climate data from the lagged time
    merged = merged.merge(
        climate_lookup.reset_index().rename(columns={"_dt": "_climate_dt"}),
        on=[location_col, "_climate_dt"],
        how="inner",
    )

    # Clean up
    merged = merged.drop(columns=["_dt", "_climate_dt"])
    return merged.reset_index(drop=True)


def run_lag_sweep(
    df: pd.DataFrame,
    model: SuitabilityModel,
    max_lag: int = 3,
    cases_col: str = "disease_cases",
    population_col: str = "population",
    location_col: str = "location",
    time_col: str = "time_period",
) -> dict:
    """Run correlation analysis at each lag from 0 to max_lag months.

    Returns a dict with:
      - "summary": list of {lag, n, spearman_r, spearman_p, pearson_r, ...}
      - "best_lag": the lag with highest signed Spearman r (see ND-B comment below)
      - "continuous_by_lag": {column_name: [{lag, spearman_r, ...}, ...]}
    """
    summary = []
    continuous_by_lag = {c.column: [] for c in model.components}

    for lag in range(max_lag + 1):
        lagged_df = create_lagged_df(df, lag, location_col, time_col)

        if len(lagged_df) < 5:
            continue

        analysis = CorrelationAnalysis(model)
        results = analysis.run(lagged_df, cases_col)

        # Extract summary row
        overall = results.get("overall_correlation", {})
        summary.append({
            "lag": lag,
            "n": results.get("n_observations", 0),
            "spearman_r": overall.get("spearman_r"),
            "spearman_p": overall.get("spearman_p"),
            "pearson_r": overall.get("pearson_r"),
            "pearson_p": overall.get("pearson_p"),
            "r_squared": overall.get("r_squared"),
        })

        # Per-continuous-variable tracking
        cont_analysis = results.get("continuous_variables", {})
        for col, col_stats in cont_analysis.items():
            if col in continuous_by_lag:
                continuous_by_lag[col].append({
                    "lag": lag,
                    "spearman_r": col_stats.get("spearman_r"),
                    "spearman_p": col_stats.get("spearman_p"),
                })

    # Find best lag
    # NORMATIVE DECISION (ND-B): Best lag is the lag with the highest *signed*
    # Spearman r, not the highest absolute r. Our hypothesis is directional:
    # more climate suitability should predict more cases (positive correlation).
    # A lag that produces a strong negative correlation is not "best" for our
    # purposes — it would imply suitability predicts fewer cases, contradicting
    # the model's logic. Using signed r makes this assumption explicit.
    valid = [s for s in summary if s["spearman_r"] is not None]
    best_lag = max(valid, key=lambda s: s["spearman_r"])["lag"] if valid else 0

    return {
        "summary": summary,
        "best_lag": best_lag,
        "continuous_by_lag": continuous_by_lag,
    }
