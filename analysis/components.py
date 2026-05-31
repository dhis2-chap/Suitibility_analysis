"""Q5: Leave-one-out component contribution analysis."""

import numpy as np
import pandas as pd
from scipy import stats

from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data


def analyze_leave_one_out(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Compare full model vs. model without each component, using annual rolling approach.

    For each model variant (full and each leave-one-out), computes rolling window
    data (months_suitable, total_annual_cases) and correlates them. This is
    consistent with the primary analysis method.

    "months_suitable" for a reduced model = months where the reduced score reaches
    its maximum (i.e., all remaining thresholds met).

    If removing a component doesn't change the annual correlation, that component
    may be redundant or uninformative for predicting annual disease burden.
    """
    # Full model
    win_df_full = compute_window_data(df, model, cases_col, location_col, time_col, window)

    full_r, full_p = (None, None)
    if len(win_df_full) >= 3 and win_df_full["months_suitable"].nunique() > 1:
        full_r, full_p = stats.spearmanr(win_df_full["months_suitable"], win_df_full["total_cases"])

    results = {
        "full_model": {
            "n_components": len(model.components),
            "spearman_r": _safe(full_r),
            "spearman_p": _safe(full_p),
            "n_windows": len(win_df_full),
        },
        "without": {},
    }

    for comp in model.components:
        # Build a reduced model without this component
        remaining_components = [c for c in model.components if c.name != comp.name]
        reduced_model = SuitabilityModel(name=model.name, components=remaining_components)

        win_df_reduced = compute_window_data(
            df, reduced_model, cases_col, location_col, time_col, window
        )

        if len(win_df_reduced) >= 3 and win_df_reduced["months_suitable"].nunique() > 1:
            r, p = stats.spearmanr(win_df_reduced["months_suitable"], win_df_reduced["total_cases"])
            fr = _safe(full_r)
            rr = _safe(r)
            results["without"][comp.name] = {
                "spearman_r": rr,
                "spearman_p": _safe(p),
                # delta_r is undefined (None) if either correlation is unavailable,
                # rather than silently substituting 0 for a missing full_r.
                "delta_r": _safe(fr - rr) if fr is not None and rr is not None else None,
                "n_windows": len(win_df_reduced),
            }
        else:
            results["without"][comp.name] = {
                "spearman_r": None,
                "spearman_p": None,
                "delta_r": None,
                "n_windows": len(win_df_reduced),
                "note": "No variance in months_suitable for reduced model",
            }

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
