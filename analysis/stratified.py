import pandas as pd

from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data


def analyze_by_location(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Per-location mean months_suitable from rolling windows.

    Returns a dict keyed by location with one field:
      - mean_months_suitable: mean of months_suitable across all qualifying windows
    """
    win_df = compute_window_data(df, model, cases_col, location_col, time_col, window)
    results = {}

    for location, group in win_df.groupby("location"):
        results[str(location)] = {
            "mean_months_suitable": _safe(group["months_suitable"].mean()),
        }

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    import numpy as np
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
