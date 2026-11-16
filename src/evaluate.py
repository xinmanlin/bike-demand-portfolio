"""Evaluation metrics: point accuracy and interval quality.

Interval methods are compared with the Winkler (interval) score, which
penalises both miss-coverage and excessive width. Comparing raw average
width alone is invalid when the methods achieve different coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error


def point_metrics(y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        p = np.clip(np.asarray(p, dtype=float), 1e-6, None)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y, p),
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "Poisson deviance": mean_poisson_deviance(y, p),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  alpha: float = 0.10) -> float:
    """Mean Winkler score for a (1-alpha) interval; lower is better."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    width = hi - lo
    penalty = (2.0 / alpha) * np.maximum(0.0, lo - y) + \
              (2.0 / alpha) * np.maximum(0.0, y - hi)
    return float(np.mean(width + penalty))


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      alpha: float = 0.10) -> dict:
    covered = (y >= lo) & (y <= hi)
    return {
        "Coverage": float(covered.mean()),
        "Average width": float(np.mean(hi - lo)),
        "Winkler score": winkler_score(y, lo, hi, alpha),
        "Coverage below target": bool(covered.mean() < 1 - alpha),
    }


def coverage_by(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                groups: np.ndarray, label: str) -> pd.DataFrame:
    d = pd.DataFrame({"group": groups, "lo": lo, "hi": hi, "y": y})
    d["covered"] = (d.y >= d.lo) & (d.y <= d.hi)
    d["width"] = d["hi"] - d["lo"]
    out = d.groupby("group").agg(
        Coverage=("covered", "mean"), Mean_width=("width", "mean"),
    )
    out.index.name = label
    return out


def worst_group_coverage(y, lo, hi, groups) -> float:
    d = pd.DataFrame({"g": groups, "c": (y >= lo) & (y <= hi)})
    return float(d.groupby("g")["c"].mean().min())


def peak_hour_mask(hours: np.ndarray) -> np.ndarray:
    return np.isin(hours, [7, 8, 9, 17, 18, 19])


def hour_group(hours: np.ndarray) -> np.ndarray:
    """Four coarse groups keep Mondrian calibration sample sizes usable."""
    h = np.asarray(hours)
    return np.select(
        [h <= 5, h <= 9, h <= 16],
        ["Night (0-5)", "Morning (6-9)", "Midday (10-16)"],
        default="Evening (17-23)",
    )


def residual_acf(y: np.ndarray, mu: np.ndarray, alpha: float,
                 nlags: int = 24) -> np.ndarray:
    from statsmodels.tsa.stattools import acf
    r = (y - mu) / np.sqrt(np.maximum(mu + alpha * mu ** 2, 1e-6))
    return np.asarray(acf(r, nlags=nlags), dtype=float)
