"""Shared analysis pipeline logic for train.py and report.py."""

import sys

import pandas as pd

from analysis.correlation import CorrelationAnalysis
from analysis.stratified import analyze_by_location
from analysis.lag import run_lag_sweep
from analysis.temporal import analyze_within_region
from analysis.components import analyze_leave_one_out
from analysis.sensitivity import (
    sweep_thresholds,
    sweep_thresholds_within_region,
    sweep_thresholds_between_region,
)
from analysis.discrimination import analyze_discrimination
from analysis.annual import analyze_months_suitable


def load_and_clean(csv_path: str, model, missing_strategy: str = "drop") -> tuple:
    """Load CSV, validate columns, and return (df, outcome_col).

    Args:
        csv_path: Path to the input CSV.
        model: SuitabilityModel to validate required columns against.
        missing_strategy: "drop" or "fill_zero".

    Returns:
        (df, outcome_col) where outcome_col is "incidence" or "disease_cases".
    """
    df = pd.read_csv(csv_path)

    if "disease_cases" in df.columns and "location" in df.columns:
        all_nan_locs = (
            df.groupby("location")["disease_cases"]
            .apply(lambda s: s.isna().all())
        )
        all_nan_locs = all_nan_locs[all_nan_locs].index.tolist()
        if all_nan_locs:
            print(f"NOTE: Excluding locations with no disease data: {all_nan_locs}")
            df = df[~df["location"].isin(all_nan_locs)].reset_index(drop=True)

    if missing_strategy == "fill_zero":
        if "disease_cases" in df.columns:
            n_missing = df["disease_cases"].isna().sum()
            if n_missing > 0:
                print(f"NOTE: Filling {n_missing} missing disease_cases with 0 "
                      f"({n_missing / len(df) * 100:.1f}% of remaining data)")
                df["disease_cases"] = df["disease_cases"].fillna(0)
    else:
        n_missing = df["disease_cases"].isna().sum() if "disease_cases" in df.columns else 0
        if n_missing > 0:
            print(f"NOTE: {n_missing} missing disease_cases will be excluded within "
                  f"each rolling window (missing_strategy='drop')")

    missing = [c for c in model.required_columns() if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if "disease_cases" not in df.columns:
        print("ERROR: Missing 'disease_cases' column in training data")
        sys.exit(1)

    if "population" in df.columns and (df["population"] > 0).all():
        df["incidence"] = df["disease_cases"] / df["population"] * 1000
        outcome_col = "incidence"
        print("Using incidence (cases per 1,000 population) as primary outcome\n")
    else:
        outcome_col = "disease_cases"
        print("WARNING: No population data — using raw disease cases.\n"
              "Cross-location comparisons will be confounded by population size.\n")

    return df, outcome_col


def run_all_analyses(df: pd.DataFrame, model, outcome_col: str) -> dict:
    """Run all analysis modules and return results dict.

    No file I/O — pure computation.

    Returns dict with keys: results, annual_results, temporal_results,
    lag_results, leave_one_out_results, sensitivity_results,
    within_region_sensitivity, between_region_sensitivity, discrimination_results.
    """
    annual_results = None
    if "time_period" in df.columns and "location" in df.columns:
        annual_results = analyze_months_suitable(df, model, cases_col=outcome_col)

    analysis = CorrelationAnalysis(model)
    results = analysis.run(df, cases_col=outcome_col)
    results["outcome_metric"] = outcome_col

    if "location" in df.columns and "time_period" in df.columns:
        by_location = analyze_by_location(
            df, model, cases_col=outcome_col,
            location_col="location", time_col="time_period",
        )
        results["by_location"] = by_location

    temporal_results = None
    if "time_period" in df.columns and "location" in df.columns:
        temporal_results = analyze_within_region(
            df, model, cases_col=outcome_col,
            location_col="location", time_col="time_period",
        )

    lag_results = None
    if "time_period" in df.columns and "location" in df.columns:
        lag_results = run_lag_sweep(df, model, max_lag=3, cases_col=outcome_col)
        results["lag_analysis"] = {
            "summary": lag_results["summary"],
            "best_lag": lag_results["best_lag"],
            "continuous_by_lag": lag_results["continuous_by_lag"],
        }

    leave_one_out_results = analyze_leave_one_out(
        df, model, cases_col=outcome_col,
        location_col="location", time_col="time_period",
    )

    sensitivity_results = sweep_thresholds(
        df, model, cases_col=outcome_col,
        location_col="location", time_col="time_period",
    )

    within_region_sensitivity = sweep_thresholds_within_region(
        df, model, cases_col=outcome_col,
        location_col="location", time_col="time_period",
    )

    between_region_sensitivity = sweep_thresholds_between_region(
        df, model, cases_col=outcome_col,
        location_col="location", time_col="time_period",
    )

    discrimination_results = analyze_discrimination(df, model, cases_col=outcome_col)

    return {
        "results": results,
        "annual_results": annual_results,
        "temporal_results": temporal_results,
        "lag_results": lag_results,
        "leave_one_out_results": leave_one_out_results,
        "sensitivity_results": sensitivity_results,
        "within_region_sensitivity": within_region_sensitivity,
        "between_region_sensitivity": between_region_sensitivity,
        "discrimination_results": discrimination_results,
    }
