"""Interpretable count models, deployable forecasting models and baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE_FORMULA = "cnt ~ C(hr) * C(workingday) + C(season) + C(yr) + holiday"

# Task A: demand MODELLING conditional on same-hour observed weather.
CONDITIONAL_FORMULA = (
    BASE_FORMULA
    + " + temp + I(temp ** 2) + hum + windspeed + C(weather)"
)
# Task A diagnostic: let the 2012 growth effect vary linearly with month.
# A C(yr):C(mnth) interaction would be rank-deficient here: the training set
# only observes yr=1 for Jan-May, so later-month cells have no support at all.
CONDITIONAL_INTERACTION_FORMULA = CONDITIONAL_FORMULA + " + C(yr):mnth"

# Task B: deployable FORECASTING. All regressors are known before hour t.
# C(mnth) is deliberately omitted: season is a deterministic function of
# month, so including both would make the design matrix singular.
FORECAST_FORMULA = BASE_FORMULA + " + log_lag168 + log_roll24"

TREE_CATEGORICAL = ["hr", "weekday", "workingday", "holiday", "season", "mnth", "yr"]
TREE_NUMERIC = ["cnt_lag_24", "cnt_lag_168", "cnt_roll24"]
TREE_FEATURES = TREE_CATEGORICAL + TREE_NUMERIC


def fit_glm(train: pd.DataFrame, formula: str, family: str = "poisson",
            alpha: float | None = None, dispersion_corrected: bool = True):
    """Fit a GLM with dispersion-corrected (quasi) standard errors.

    The Pearson dispersion of this dataset is ~13, so the default 'nonrobust'
    covariance badly understates uncertainty. ``scale='X2'`` rescales the
    standard errors; point predictions are unchanged.
    """
    if family == "poisson":
        fam = sm.families.Poisson()
    elif family == "nb":
        if alpha is None:
            raise ValueError("negative binomial requires alpha")
        fam = sm.families.NegativeBinomial(alpha=alpha)
    else:
        raise ValueError(f"unknown family: {family}")
    fit_kwargs = {"scale": "X2"} if dispersion_corrected else {}
    return smf.glm(formula=formula, data=train, family=fam).fit(**fit_kwargs)


def estimate_nb_alpha(poisson_fit, train: pd.DataFrame) -> float:
    """Method-of-moments plug-in for NB2 dispersion (in-sample residuals).

    Used only to define the variance function of the conformal scale, not to
    claim an optimised NB point predictor.
    """
    mu = np.asarray(poisson_fit.predict(train), dtype=float)
    y = train["cnt"].to_numpy(dtype=float)
    return max(float((((y - mu) ** 2 - y).sum()) / (mu ** 2).sum()), 1e-6)


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(frame), dtype=float)


def pearson_dispersion(fit) -> float:
    return float(fit.pearson_chi2 / fit.df_resid)


def quadratic_peak(fit, term: str = "temp", quad: str = "I(temp ** 2)") -> dict:
    """Correct reading of a polynomial term in a log-link GLM.

    With log(mu) = ... + b1*temp + b2*temp^2, exp(b1) is NOT a rate ratio:
    the marginal effect is b1 + 2*b2*temp and the response peaks at
    temp* = -b1 / (2*b2). We report both, plus a finite-change multiplier.
    """
    b1 = float(fit.params[term])
    b2 = float(fit.params[quad])
    peak = -b1 / (2 * b2)
    low, high = 0.5, 0.7
    finite_change = float(np.exp(b1 * (high - low) + b2 * (high ** 2 - low ** 2)))
    return {
        "linear_coef": b1,
        "quadratic_coef": b2,
        "peak_temp": peak,
        "marginal_at_0_5": b1 + 2 * b2 * 0.5,
        "marginal_at_0_8": b1 + 2 * b2 * 0.8,
        "multiplier_0_5_to_0_7": finite_change,
    }


def baselines(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, np.ndarray]:
    """Honest reference points, including three naive persistence forecasts.

    Task B is a one-step-ahead forecast, so persistence at lag 1 (previous
    hour) and lag 24 (same hour yesterday) are reported alongside the
    seasonal-naive week lag. If the previous hour is missing from the raw
    file (a handful of rows), the t-1 forecast falls back to the t-24 value.
    """
    global_mean = float(train["cnt"].mean())
    hourly = train.groupby(["hr", "workingday"])["cnt"].mean()
    pred_hourly = np.array([
        hourly.get((hr, wd), global_mean)
        for hr, wd in zip(target["hr"], target["workingday"])
    ], dtype=float)
    return {
        "Global mean": np.full(len(target), global_mean),
        "Hour x workingday mean": pred_hourly,
        "Naive (t-1h)": target["cnt_lag_1"].fillna(target["cnt_lag_24"]).to_numpy(dtype=float),
        "Naive (t-24h)": target["cnt_lag_24"].to_numpy(dtype=float),
        "Seasonal naive (t-168h)": target["cnt_lag_168"].to_numpy(dtype=float),
    }


def fit_tree(train: pd.DataFrame, kind: str = "xgb"):
    """Tree models use the same leakage-safe information as the GLM.

    The representation differs: trees receive raw lags (including the extra
    t-24h lag), month and weekday, while the GLM uses log-transformed lags and
    an explicit hour x working-day interaction. This is therefore not a
    perfectly matched feature-set experiment.
    """
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), TREE_CATEGORICAL),
        ("num", "passthrough", TREE_NUMERIC),
    ])
    if kind == "xgb":
        est = XGBRegressor(
            objective="count:poisson", n_estimators=400, max_depth=4,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", random_state=42, n_jobs=-1,
        )
    elif kind == "rf":
        est = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(train[TREE_FEATURES], train["cnt"])
    return pipe


def predict_tree(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(frame[TREE_FEATURES]), dtype=float)
