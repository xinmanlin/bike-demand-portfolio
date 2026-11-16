"""Split, rolling and adaptive conformal prediction for count time series.

All methods target marginal coverage 1-alpha. The online variants (rolling
and ACI) update the calibration pool using ONLY realised observations up to
t-1, which is legitimate in a live deployment; no future information is used.

Standard split conformal requires exchangeability of calibration scores.
Hourly demand violates it (strong drift and lag-1 autocorrelation), so we
report empirical coverage and diagnose the assumption rather than claiming
a guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    name: str
    lo: np.ndarray
    hi: np.ndarray
    covered: np.ndarray
    q_path: np.ndarray | None = None


def conformal_quantile(scores: np.ndarray, level: float = 0.90) -> float:
    """Finite-sample split-conformal quantile ceil((n+1)*level)/n."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    k = min(n, math.ceil((n + 1) * level))
    return float(s[k - 1])


def sigma_functions(alpha: float) -> dict:
    return {
        "Absolute": lambda mu: np.ones_like(mu),
        "Poisson-scaled": lambda mu: np.sqrt(np.maximum(mu, 1e-6)),
        "NB-scaled": lambda mu: np.sqrt(np.maximum(mu + alpha * mu ** 2, 1e-6)),
    }


def scores_for(y: np.ndarray, mu: np.ndarray, sigma) -> np.ndarray:
    return np.abs(y - mu) / sigma(mu)


def split_conformal(y_cal, mu_cal, y_eval, mu_eval, sigma,
                     level: float = 0.90, name: str = "Split") -> Interval:
    q = conformal_quantile(scores_for(y_cal, mu_cal, sigma), level)
    half = q * sigma(mu_eval)
    lo = np.maximum(0.0, mu_eval - half)
    hi = mu_eval + half
    return Interval(name, lo, hi, (y_eval >= lo) & (y_eval <= hi),
                    np.full(len(y_eval), q))


def rolling_conformal(y_cal, mu_cal, y_eval, mu_eval, sigma,
                       level: float = 0.90, window: int = 24 * 28,
                       name: str = "Rolling") -> Interval:
    """Re-estimate q every hour from the most recent `window` past scores."""
    pool = list(scores_for(y_cal, mu_cal, sigma))
    sig_eval = sigma(mu_eval)
    lo, hi, q_path = [], [], []
    for t in range(len(y_eval)):
        recent = np.asarray(pool[-window:], dtype=float)
        q = conformal_quantile(recent, level)
        l = max(0.0, float(mu_eval[t] - q * sig_eval[t]))
        h = float(mu_eval[t] + q * sig_eval[t])
        lo.append(l)
        hi.append(h)
        q_path.append(q)
        # Reality arrives at the end of hour t; only then does it enter the pool.
        pool.append(abs(float(y_eval[t]) - float(mu_eval[t])) / float(sig_eval[t]))
    lo, hi = np.asarray(lo), np.asarray(hi)
    return Interval(name, lo, hi, (y_eval >= lo) & (y_eval <= hi), np.asarray(q_path))


def aci_conformal(y_cal, mu_cal, y_eval, mu_eval, sigma,
                  level: float = 0.90, gamma: float = 0.01,
                  name: str = "ACI") -> Interval:
    """Adaptive Conformal Inference (Gibbs & Candes 2021), score form.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t); the conformal quantile
    is then recomputed at level 1 - alpha_{t+1} from all past scores.
    """
    alpha_nom = 1.0 - level
    alpha_t = alpha_nom
    pool = list(scores_for(y_cal, mu_cal, sigma))
    sig_eval = sigma(mu_eval)
    lo, hi, q_path = [], [], []
    for t in range(len(y_eval)):
        q = conformal_quantile(np.asarray(pool, dtype=float), 1.0 - alpha_t)
        l = max(0.0, float(mu_eval[t] - q * sig_eval[t]))
        h = float(mu_eval[t] + q * sig_eval[t])
        lo.append(l)
        hi.append(h)
        q_path.append(q)
        err = 0.0 if l <= y_eval[t] <= h else 1.0
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_nom - err), 0.001, 0.999))
        pool.append(abs(float(y_eval[t]) - float(mu_eval[t])) / float(sig_eval[t]))
    lo, hi = np.asarray(lo), np.asarray(hi)
    return Interval(name, lo, hi, (y_eval >= lo) & (y_eval <= hi), np.asarray(q_path))


def mondrian_conformal(y_cal, mu_cal, y_eval, mu_eval, sigma,
                        cal_groups, eval_groups, level: float = 0.90,
                        name: str = "Mondrian") -> Interval:
    """Group-conditional split conformal with coarse hour groups.

    Unlike the original per-hour calibration (about 100 calibration points
    per hour), four groups keep the quantile estimates stable.
    """
    cal_scores = scores_for(y_cal, mu_cal, sigma)
    q_by_group = {
        g: conformal_quantile(cal_scores[np.asarray(cal_groups) == g], level)
        for g in np.unique(cal_groups)
    }
    fallback = conformal_quantile(cal_scores, level)
    qs = np.array([q_by_group.get(g, fallback) for g in eval_groups], dtype=float)
    half = qs * sigma(mu_eval)
    lo = np.maximum(0.0, mu_eval - half)
    hi = mu_eval + half
    return Interval(name, lo, hi, (y_eval >= lo) & (y_eval <= hi), qs)
