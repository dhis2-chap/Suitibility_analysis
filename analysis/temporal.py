"""Q2: Within-location temporal analysis using the annual rolling approach.

For each location, computes rolling 12-month windows of (months_suitable,
total_annual_cases) and correlates them. This answers:

  "Within a given district, does annual suitability exposure predict
   annual disease burden as the window advances through time?"
"""

import numpy as np
from scipy import stats
import pandas as pd

from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data


def analyze_within_region(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Per-location within-location Spearman r of rolling annual suitability vs. cases.

    Returns:
        - per_location: {location: {spearman_r, spearman_p}}
        - summary: {mean_within_location_r}
    """
    win_df = compute_window_data(df, model, cases_col, location_col, time_col, window)

    results = {"per_location": {}, "summary": {}}

    if win_df.empty:
        return results

    all_rs = []

    for location, group in win_df.groupby("location"):
        if len(group) >= 3 and group["months_suitable"].nunique() > 1:
            r, p = stats.spearmanr(group["months_suitable"], group["total_cases"])
            loc_result = {
                "spearman_r": _safe(r),
                "spearman_p": _safe(p),
            }
            if r is not None and not np.isnan(r):
                all_rs.append(r)
        else:
            loc_result = {"spearman_r": None, "spearman_p": None}

        results["per_location"][str(location)] = loc_result

    valid_rs = [r for r in all_rs if r is not None and not np.isnan(r)]
    if valid_rs:
        results["summary"]["mean_within_location_r"] = _safe(np.mean(valid_rs))

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
