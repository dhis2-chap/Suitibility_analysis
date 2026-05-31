"""Threshold sensitivity analysis using the annual rolling approach.

For each component, sweeps threshold values and tracks Spearman r for three metrics:
  - Pooled: months_suitable vs total_cases pooled across all (location, window) pairs.
  - Within-region: mean within-location temporal r (same logic as analyze_within_region).
  - Between-region: spatial r across locations between mean months_suitable and mean total_cases.
"""

import pandas as pd
import numpy as np
from scipy import stats

from suitability.model import SuitabilityModel
from suitability.thresholds import ThresholdComponent
from analysis.annual import compute_window_data


def sweep_thresholds(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Sweep thresholds, tracking pooled Spearman r (months_suitable vs total_cases)."""
    return _run_sweep(df, model, _pooled_r, cases_col, location_col, time_col, window)


def sweep_thresholds_within_region(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Sweep thresholds, tracking mean within-location temporal Spearman r.

    For each threshold combination, computes the Spearman r between months_suitable
    and total_cases within each location across time, then averages across locations.
    Answers: does changing this threshold strengthen or weaken the year-to-year
    correlation within individual districts?
    """
    return _run_sweep(df, model, _mean_within_location_r, cases_col, location_col, time_col, window)


def sweep_thresholds_between_region(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Sweep thresholds, tracking between-location spatial Spearman r.

    For each threshold combination, computes the Spearman r across locations between
    each location's mean months_suitable and mean total_cases.
    Answers: does changing this threshold strengthen or weaken the spatial pattern
    where districts with more suitable months also have more disease burden?
    """
    return _run_sweep(df, model, _between_location_r, cases_col, location_col, time_col, window)


def _run_sweep(
    df: pd.DataFrame,
    model: SuitabilityModel,
    metric_fn,
    cases_col: str,
    location_col: str,
    time_col: str,
    window: int,
) -> dict:
    """Generic sweep runner. Calls metric_fn(win_df) for each threshold combination."""
    sweep_config = _default_sweep_config(model, df)
    results = {}

    for comp in model.components:
        if comp.name not in sweep_config:
            continue

        config = sweep_config[comp.name]
        comp_results = {
            "original_min": comp.min_value,
            "original_max": comp.max_value,
            "sweeps": [],
        }

        for min_val in config.get("min_values", [comp.min_value]):
            for max_val in config.get("max_values", [comp.max_value]):
                modified_model = _modify_component(model, comp.name, min_val, max_val)
                win_df = compute_window_data(
                    df, modified_model, cases_col, location_col, time_col, window
                )
                comp_results["sweeps"].append({
                    "min_value": min_val,
                    "max_value": max_val,
                    "spearman_r": metric_fn(win_df),
                })

        results[comp.name] = comp_results

    return results


def _pooled_r(win_df: pd.DataFrame) -> float | None:
    """Spearman r between months_suitable and total_cases pooled across all windows."""
    if len(win_df) >= 3 and win_df["months_suitable"].nunique() > 1:
        r, _ = stats.spearmanr(win_df["months_suitable"], win_df["total_cases"])
        return _safe(r)
    return None


def _mean_within_location_r(win_df: pd.DataFrame) -> float | None:
    """Mean Spearman r across locations of within-location temporal correlation."""
    all_rs = []
    for _, group in win_df.groupby("location"):
        if len(group) >= 3 and group["months_suitable"].nunique() > 1:
            r, _ = stats.spearmanr(group["months_suitable"], group["total_cases"])
            if r is not None and not np.isnan(r):
                all_rs.append(float(r))
    return _safe(np.mean(all_rs)) if all_rs else None


def _between_location_r(win_df: pd.DataFrame) -> float | None:
    """Spearman r across locations between mean months_suitable and mean total_cases."""
    per_loc = win_df.groupby("location").agg(
        mean_months=("months_suitable", "mean"),
        mean_cases=("total_cases", "mean"),
    ).reset_index()
    if len(per_loc) < 3 or per_loc["mean_months"].nunique() <= 1:
        return None
    r, _ = stats.spearmanr(per_loc["mean_months"], per_loc["mean_cases"])
    return _safe(r)


def _default_sweep_config(model: SuitabilityModel, df: pd.DataFrame) -> dict:
    """Derive sweep ranges from the actual column data distribution.

    Uses percentiles of each column so the ranges are automatically unit-correct
    (e.g. mm/day vs mm/month, different humidity distributions across countries).
    5 values for min thresholds and 5 for max thresholds, giving 25 combinations
    per component (or 5 for one-sided thresholds).
    """
    config = {}
    for comp in model.components:
        col = df[comp.column].dropna()
        quantiles = [0.05, 0.15, 0.25, 0.35, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
        p = {q: col.quantile(q) for q in quantiles}

        if comp.name == "temperature":
            # NORMATIVE DECISION: Sweep centered on the original threshold values
            # (±4°C in 2°C steps), so the original threshold always falls in the
            # middle cell of the heatmap grid. The previous approach used data-distribution
            # percentiles, which placed the original threshold at the edge of the sweep
            # range (the red reference box appeared in a corner).
            # Alternative: keep percentile approach but widen the range so the original
            # threshold is never in the outer 20% of the grid.
            if comp.min_value is not None and comp.max_value is not None:
                step = 2.0
                mins = _round_series(
                    np.linspace(comp.min_value - 2 * step, comp.min_value + 2 * step, 5), col
                )
                maxs = _round_series(
                    np.linspace(comp.max_value - 2 * step, comp.max_value + 2 * step, 5), col
                )
            else:
                mins = _round_series(np.linspace(p[0.05], p[0.50], 5), col)
                maxs = _round_series(np.linspace(p[0.60], col.max() + 5, 5), col)
            # Always include original thresholds (already at centre when min/max are set above)
            if comp.min_value is not None:
                mins = sorted(set(mins + [float(round(comp.min_value, 1))]))
            if comp.max_value is not None:
                maxs = sorted(set(maxs + [float(round(comp.max_value, 1))]))
            config["temperature"] = {"min_values": mins, "max_values": maxs}

        elif comp.name == "precipitation":
            # One-sided (min only): sweep from near-zero to p80.
            # NORMATIVE DECISION: Sweep range derived from data distribution, not
            # hardcoded mm/month values. This ensures correct scaling regardless
            # of whether data is in mm/day, mm/month, or other units.
            mins = _round_series(np.linspace(p[0.05], p[0.80], 9), col)
            if comp.min_value is not None:
                mins = sorted(set(mins + [float(round(comp.min_value, 2))]))
            config["precipitation"] = {
                "min_values": mins,
                "max_values": [comp.max_value],
            }

        elif comp.name == "humidity":
            # Fixed sweep ranges: min threshold 42–58, max threshold 72–88.
            # Gaps between ranges ensure min < max for all combinations.
            # Current thresholds [50, 80] are centred within each range.
            mins = _round_series(np.linspace(42, 58, 5), col)
            maxs = _round_series(np.linspace(72, 88, 5), col)
            if comp.min_value is not None:
                mins = sorted(set(mins + [float(round(comp.min_value, 1))]))
            if comp.max_value is not None:
                maxs = sorted(set(maxs + [float(round(comp.max_value, 1))]))
            config["humidity"] = {"min_values": mins, "max_values": maxs}

    return config


def _round_series(values: np.ndarray, col: pd.Series) -> list:
    """Round sweep values sensibly based on the column's scale."""
    scale = col.max() - col.min()
    if scale > 50:
        decimals = 0
    elif scale > 5:
        decimals = 1
    else:
        decimals = 2
    return sorted(set(float(round(v, decimals)) for v in values))


def _modify_component(
    model: SuitabilityModel,
    comp_name: str,
    new_min: float | None,
    new_max: float | None,
) -> SuitabilityModel:
    """Return a copy of the model with one component's thresholds changed."""
    new_components = []
    for comp in model.components:
        if comp.name == comp_name:
            new_components.append(ThresholdComponent(
                name=comp.name,
                column=comp.column,
                min_value=new_min,
                max_value=new_max,
            ))
        else:
            new_components.append(comp)
    return SuitabilityModel(name=model.name, components=new_components)


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
