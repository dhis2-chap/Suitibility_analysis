import numpy as np
import pandas as pd
from scipy import stats

from suitability.model import SuitabilityModel


class CorrelationAnalysis:
    """Correlates suitability scores with disease cases per lag in run_lag_sweep."""

    def __init__(self, model: SuitabilityModel):
        self.model = model

    def run(
        self,
        df: pd.DataFrame,
        cases_col: str = "disease_cases",
    ) -> dict:
        """Run correlation analyses. Returns nested dict of results."""
        composite = self.model.compute_composite_score(df)

        results = {}
        results["model_name"] = self.model.name
        results["n_observations"] = len(df)
        results["overall_correlation"] = self._overall_correlation(
            composite, df[cases_col]
        )
        results["continuous_variables"] = self._analyze_continuous(df, cases_col)

        return results

    def _overall_correlation(
        self, scores: pd.Series, cases: pd.Series
    ) -> dict:
        """Spearman, Pearson, and R² between score and cases."""
        result = {}

        valid = cases.notna()
        scores_v = scores[valid]
        cases_v = cases[valid]

        if len(scores_v) < 3:
            result["note"] = "Too few observations for correlation"
            return result

        spearman_r, spearman_p = stats.spearmanr(scores_v, cases_v)
        result["spearman_r"] = _safe_float(spearman_r)
        result["spearman_p"] = _safe_float(spearman_p)

        pearson_r, pearson_p = stats.pearsonr(scores_v, cases_v)
        result["pearson_r"] = _safe_float(pearson_r)
        result["pearson_p"] = _safe_float(pearson_p)
        result["r_squared"] = _safe_float(pearson_r**2)

        return result

    def _analyze_continuous(self, df: pd.DataFrame, cases_col: str) -> dict:
        """Correlation between raw climate variables and cases."""
        results = {}
        cases = df[cases_col]
        valid = cases.notna()

        for component in self.model.components:
            col = component.column
            if col not in df.columns:
                continue

            values = df[col][valid]
            cases_v = cases[valid]
            # also exclude rows where the climate column itself is NaN; without this
            # spearmanr silently returns NaN for the whole result even if only one
            # row is missing — most visible at lag 0 where no inner-join pre-cleans the data
            climate_valid = values.notna()
            values = values[climate_valid]
            cases_v = cases_v[climate_valid]
            if len(values) < 3:
                results[col] = {"note": "Too few observations"}
                continue

            spearman_r, spearman_p = stats.spearmanr(values, cases_v)
            pearson_r, pearson_p = stats.pearsonr(values, cases_v)

            results[col] = {
                "spearman_r": _safe_float(spearman_r),
                "spearman_p": _safe_float(spearman_p),
                "pearson_r": _safe_float(pearson_r),
                "pearson_p": _safe_float(pearson_p),
            }

        return results


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN/inf."""
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
