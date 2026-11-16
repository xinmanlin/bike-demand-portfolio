"""Data loading, integrity checks and leakage-safe feature engineering.

The UCI ``hour.csv`` is not a complete hourly grid (some hours are missing).
We reindex onto a full hourly grid so that ``cnt.shift(168)`` means exactly
"the same hour one week ago" instead of "the 168th previous *record*".
Rows without the required history are excluded from modelling.

Two feature sets are provided:

* ``conditional``  - calendar + same-hour observed weather. This is demand
  MODELLING conditional on weather, not a deployable forecast.
* ``forecasting``  - calendar + lagged demand only. Every column is known
  at time t-1 or earlier, so the model could be run in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CALENDAR_CATEGORICAL = ["hr", "weekday", "workingday", "holiday", "season", "mnth", "yr"]
WEATHER_CONDITIONAL = ["temp", "hum", "windspeed", "weather"]
LAG_NUMERIC = ["cnt_lag_24", "cnt_lag_168", "cnt_roll24", "log_lag168", "log_roll24"]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp


@dataclass(frozen=True)
class BacktestFold:
    name: str
    train: pd.DataFrame
    horizon: pd.DataFrame
    train_end: pd.Timestamp


def load_hourly(path: str | Path) -> pd.DataFrame:
    """Return a complete hourly grid with calendar and lagged features."""
    raw = pd.read_csv(path)
    raw["datetime"] = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    raw = raw.sort_values("datetime").drop_duplicates("datetime", keep="last")

    grid = pd.date_range(raw["datetime"].min(), raw["datetime"].max(), freq="h")
    df = raw.set_index("datetime").reindex(grid)
    df.index.name = "datetime"
    df["observed"] = df["cnt"].notna()

    # Missing hours are never modelled; ffill only keeps calendar columns
    # well-formed for the lag calculations below.
    for col in CALENDAR_CATEGORICAL + ["temp", "atemp", "hum", "windspeed", "weathersit"]:
        df[col] = df[col].ffill().bfill()
    for col in CALENDAR_CATEGORICAL + ["weathersit"]:
        df[col] = df[col].astype(int)

    # Weather level 4 has 3 observations in the whole dataset -> merge into 3.
    df["weather"] = df["weathersit"].clip(upper=3)

    # Lags are computed on the complete grid, so they are strictly a
    # function of past hours. shift(1) keeps the rolling mean causal.
    df["cnt_lag_1"] = df["cnt"].shift(1)
    df["cnt_lag_24"] = df["cnt"].shift(24)
    df["cnt_lag_168"] = df["cnt"].shift(168)
    df["cnt_roll24"] = df["cnt"].shift(1).rolling(24, min_periods=18).mean()
    df["log_lag168"] = np.log1p(df["cnt_lag_168"])
    df["log_roll24"] = np.log1p(df["cnt_roll24"])

    df["date"] = df.index.normalize()
    df["month_of_year"] = df.index.month
    df["year_index"] = df.index.year - df.index.year.min()
    return df


def integrity_report(df: pd.DataFrame) -> dict:
    missing = df.index[~df["observed"]]
    observed_cnt = df.loc[df["observed"], "cnt"]
    sandy = (missing >= "2012-10-29") & (missing < "2012-10-31")
    return {
        "min_observed_cnt": int(observed_cnt.min()),
        "missing_share_00_06": float(np.mean(missing.hour < 6)),
        "missing_share_02_05": float(np.mean((missing.hour >= 2) & (missing.hour <= 5))),
        "missing_2012_10_29_30": int(sandy.sum()),
        "n_grid_hours": int(len(df)),
        "n_observed_hours": int(df["observed"].sum()),
        "n_missing_hours": int((~df["observed"]).sum()),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "n_rows_with_full_lags": int(df["cnt_lag_168"].notna().sum()),
    }


def modelling_frame(df: pd.DataFrame, feature_set: str = "forecasting") -> pd.DataFrame:
    """Rows usable for a given feature set (observed target + available history)."""
    if feature_set == "forecasting":
        required = ["cnt_lag_168", "cnt_roll24", "cnt_lag_24"]
    elif feature_set == "conditional":
        # Require the week lag too, so both tasks share the same honest baselines.
        required = ["temp", "hum", "windspeed", "weather", "cnt_lag_168"]
    else:
        raise ValueError(feature_set)
    keep = df[df["observed"] & df[required].notna().all(axis=1)].copy()
    return keep


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70,
                        val_frac: float = 0.15) -> ChronologicalSplit:
    """Split on *day* boundaries so one calendar day never crosses two sets."""
    start = df.index.min().normalize()
    n_days = (df.index.max().normalize() - start).days + 1
    train_end = start + pd.Timedelta(days=int(round(n_days * train_frac)))
    val_end = start + pd.Timedelta(days=int(round(n_days * (train_frac + val_frac))))
    train = df[df.index < train_end].copy()
    val = df[(df.index >= train_end) & (df.index < val_end)].copy()
    test = df[df.index >= val_end].copy()
    assert train.index.max() < val.index.min() <= val.index.max() < test.index.min()
    assert train.index.max().hour == 23 and val.index.max().hour == 23
    return ChronologicalSplit(train, val, test, train_end, val_end)


def rolling_origin_folds(df: pd.DataFrame, train_end: pd.Timestamp,
                          n_folds: int = 3, horizon_days: int = 28) -> list[BacktestFold]:
    """Non-overlapping expanding-window folds entirely inside the training era.

    These provide independent evidence without touching the validation or test periods.
    """
    folds = []
    for i in range(n_folds):
        t0 = train_end - pd.Timedelta(days=(n_folds - i) * horizon_days)
        t1 = t0 + pd.Timedelta(days=horizon_days)
        folds.append(BacktestFold(
            name=f"fold_{i + 1}",
            train=df[df.index < t0].copy(),
            horizon=df[(df.index >= t0) & (df.index < t1)].copy(),
            train_end=t0,
        ))
    return folds


def period_summary(split: ChronologicalSplit) -> pd.DataFrame:
    rows = []
    for name, d in [("Train", split.train), ("Validation", split.val), ("Test", split.test)]:
        rows.append({
            "Period": name,
            "Start": str(d.index.min()),
            "End": str(d.index.max()),
            "N hours": int(d["observed"].sum()),
            "Mean cnt": round(float(d["cnt"].mean()), 1),
            "Missing hours": int((~d["observed"]).sum()),
        })
    return pd.DataFrame(rows)
