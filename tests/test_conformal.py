import numpy as np

from src import conformal, evaluate


def test_conformal_quantile_finite_sample_correction():
    s = np.arange(1, 11, dtype=float)  # n=10, level=0.90 -> ceil(9.9)=10
    assert conformal.conformal_quantile(s, 0.90) == 10.0
    assert conformal.conformal_quantile(s, 0.50) == 6.0  # ceil(11 * 0.5) = 6


def test_split_conformal_covers_iid_data():
    rng = np.random.default_rng(0)
    mu_cal = np.full(2000, 100.0)
    mu_val = np.full(3000, 100.0)
    y_cal = rng.poisson(100, 2000).astype(float)
    y_val = rng.poisson(100, 3000).astype(float)
    sigma = conformal.sigma_functions(0.05)["Poisson-scaled"]
    res = conformal.split_conformal(y_cal, mu_cal, y_val, mu_val, sigma, 0.90)
    assert 0.87 <= res.covered.mean() <= 0.93


def test_winkler_penalises_misses_and_width():
    y = np.array([10.0])
    perfect = evaluate.winkler_score(y, np.array([0.0]), np.array([20.0]), 0.10)
    missed = evaluate.winkler_score(y, np.array([50.0]), np.array([60.0]), 0.10)
    wide = evaluate.winkler_score(y, np.array([0.0]), np.array([100.0]), 0.10)
    assert missed > perfect
    assert wide > perfect


def test_aci_adapts_after_a_miss():
    rng = np.random.default_rng(1)
    y_cal = rng.poisson(100, 800).astype(float)
    mu_cal = np.full(800, 100.0)
    y_eval = rng.poisson(100, 60).astype(float)
    mu_eval = np.full(60, 100.0)
    sigma = conformal.sigma_functions(0.05)["Poisson-scaled"]
    res = conformal.aci_conformal(y_cal, mu_cal, y_eval, mu_eval, sigma, 0.90, gamma=0.05)
    assert res.q_path is not None
    assert not np.allclose(res.q_path[0], res.q_path[-1]) or res.covered.all()
