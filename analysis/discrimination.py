"""Q8: Score distribution — frequency table of composite suitability score levels."""

import numpy as np

from suitability.model import SuitabilityModel
import pandas as pd


def analyze_discrimination(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
) -> dict:
    """Count how many observation-months fall at each composite score level.

    Returns a distribution list with one entry per level from 0 to max_score.
    Disease cases are not used — this is purely a frequency table of scores.
    """
    composite = model.compute_composite_score(df)
    max_score = len(model.components)
    n_total = len(composite)

    distribution = []
    for level in range(max_score + 1):
        n = int((composite == level).sum())
        distribution.append({
            "score": level,
            "n": n,
            "pct": _safe(n / n_total * 100),
        })

    return {"distribution": distribution}


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
