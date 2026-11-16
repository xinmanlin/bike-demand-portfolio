"""One-command reproduction of the whole study.

    python -m src.run_analysis

Writes every table, figure and a machine-readable summary to results/.
Model choices are made on the validation set / rolling-origin backtest.
The test set is evaluated exactly once, at the end, and never used to
choose a model, a feature, or an interval method.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import conformal, data, evaluate, models  # noqa: E402

RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

ALPHA = 0.10
LEVEL = 1 - ALPHA
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(RESULTS / name, index=False)


def arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def main() -> None:
    summary: dict = {}
    df = data.load_hourly(ROOT / "data" / "raw" / "hour.csv")
    integrity = data.integrity_report(df)
    split = data.chronological_split(df)
    summary["integrity"] = integrity
    summary["split"] = {
        "train_end": str(split.train_end), "val_end": str(split.val_end),
    }
    period = data.period_summary(split)
    save(period, "01_data_and_split.csv")
    summary["periods"] = period.to_dict("records")
    print("Data grid", integrity["n_grid_hours"], "observed", integrity["n_observed_hours"],
          "missing", integrity["n_missing_hours"])
    print(period.to_string(index=False))

    # ---------------------------------------------------------------- Task A
    trA = data.modelling_frame(split.train, "conditional")
    vaA = data.modelling_frame(split.val, "conditional")
    teA = data.modelling_frame(split.test, "conditional")

    glm_A = models.fit_glm(trA, models.CONDITIONAL_FORMULA)
    disp_A = models.pearson_dispersion(glm_A)
    alpha_nb = models.estimate_nb_alpha(glm_A, trA)
    glm_A_nb = models.fit_glm(trA, models.CONDITIONAL_FORMULA, family="nb", alpha=alpha_nb)
    glm_A_int = models.fit_glm(trA, models.CONDITIONAL_INTERACTION_FORMULA)
    peak = models.quadratic_peak(glm_A)

    key_terms = ["temp", "I(temp ** 2)", "hum", "windspeed", "C(yr)[T.1]",
                 "C(weather)[T.2]", "C(weather)[T.3]", "holiday"]
    coef_tbl = pd.DataFrame({
        "Term": key_terms,
        "coef": [float(glm_A.params.get(t, np.nan)) for t in key_terms],
        "quasi SE": [float(glm_A.bse.get(t, np.nan)) for t in key_terms],
    })
    coef_tbl["quasi z"] = coef_tbl["coef"] / coef_tbl["quasi SE"]
    # exp(coef) is only meaningful for stand-alone terms. For the quadratic
    # temperature pair it is deliberately suppressed: quoting exp(4.05) as a
    # "rate ratio" while ignoring I(temp**2) is a common and serious misreading.
    poly = {"temp", "I(temp ** 2)"}
    coef_tbl["exp(coef) (single terms only)"] = [
        "n/a (joint quadratic pair)" if t in poly else round(float(np.exp(c)), 3)
        for t, c in zip(key_terms, coef_tbl["coef"])
    ]
    save(coef_tbl.round(4), "02_conditional_coefficients.csv")

    bA = models.baselines(trA, vaA)
    predA = {
        **bA,
        "Poisson GLM (conditional)": models.predict(glm_A, vaA),
        "NB GLM (conditional)": models.predict(glm_A_nb, vaA),
        "Poisson + yr:month": models.predict(glm_A_int, vaA),
    }
    mA = evaluate.point_metrics(arr(vaA["cnt"]), predA)
    # Systematic bias of the conditional (weather) model, for contrast with Task B
    predA_glm = predA["Poisson GLM (conditional)"]
    yA = arr(vaA["cnt"])
    biasA = {
        "mean_bias": float(np.mean(predA_glm - yA)),
        "median_actual_over_pred": float(np.median(yA / np.maximum(predA_glm, 1e-6))),
    }
    save(mA.round(2), "03_conditional_validation.csv")

    taskA = {
        "dispersion": disp_A,
        "nb_alpha": alpha_nb,
        "quadratic": peak,
        "coefficients": coef_tbl.to_dict("records"),
        "validation": mA.to_dict("records"),
        "interaction_aic": float(glm_A_int.aic),
        "baseline_aic": float(glm_A.aic),
        "bias": biasA,
    }
    summary["taskA"] = taskA
    print("\nTask A (conditional) dispersion", round(disp_A, 2),
          "temp peak at", round(peak["peak_temp"], 3))
    print(mA.round(2).to_string(index=False))

    # Correct reading of the polynomial temperature effect
    temp_grid = np.linspace(0.1, 0.95, 50)
    log_effect = peak["linear_coef"] * temp_grid + peak["quadratic_coef"] * temp_grid ** 2
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(temp_grid, np.exp(log_effect - log_effect.max()), color="#185FA5", lw=2)
    ax.axvline(peak["peak_temp"], color="#A32D2D", ls="--", lw=1,
               label=f"peak at temp = {peak['peak_temp']:.2f}")
    ax.set_xlabel("Normalised temperature")
    ax.set_ylabel("Relative demand (log-link, centred)")
    ax.set_title("Quadratic temperature effect (jointly interpreted)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGDIR / "temp_effect.png"); plt.close(fig)

    # ---------------------------------------------------------------- Task B
    trB = data.modelling_frame(split.train, "forecasting")
    vaB = data.modelling_frame(split.val, "forecasting")
    teB = data.modelling_frame(split.test, "forecasting")

    glm_B = models.fit_glm(trB, models.FORECAST_FORMULA)
    xgb = models.fit_tree(trB, "xgb")
    rf = models.fit_tree(trB, "rf")
    disp_B = models.pearson_dispersion(glm_B)

    bB = models.baselines(trB, vaB)
    predB = {
        **bB,
        "Poisson GLM (forecasting)": models.predict(glm_B, vaB),
        "XGBoost (lag features)": models.predict_tree(xgb, vaB),
        "Random forest (lag features)": models.predict_tree(rf, vaB),
    }
    mB = evaluate.point_metrics(arr(vaB["cnt"]), predB)
    save(mB.round(2), "04_forecast_validation.csv")

    # Rolling-origin backtest: independent evidence entirely inside training era
    folds = data.rolling_origin_folds(trB, split.train_end, n_folds=3, horizon_days=28)
    bt_rows = []
    for f in folds:
        ftr = data.modelling_frame(f.train, "forecasting")
        fho = data.modelling_frame(f.horizon, "forecasting")
        if len(ftr) < 500 or len(fho) < 100:
            continue
        gm = models.fit_glm(ftr, models.FORECAST_FORMULA)
        xm = models.fit_tree(ftr, "xgb")
        fpreds = {
            **models.baselines(ftr, fho),
            "Poisson GLM (forecasting)": models.predict(gm, fho),
            "XGBoost (lag features)": models.predict_tree(xm, fho),
        }
        fm = evaluate.point_metrics(arr(fho["cnt"]), fpreds)
        for _, r in fm.iterrows():
            bt_rows.append({"Fold": f.name, "Train until": str(f.train_end.date()),
                            "Model": r["Model"], "MAE": r["MAE"], "RMSE": r["RMSE"]})
    bt = pd.DataFrame(bt_rows)
    save(bt.round(2), "05_rolling_origin_backtest.csv")
    bt_summary = (bt.groupby("Model")["MAE"]
                  .agg(["mean", "std", "min", "max"]).round(2)
                  .sort_values("mean").reset_index())
    save(bt_summary, "06_backtest_summary.csv")

    # Test period: never used for selection (examined in an earlier project version)
    bT = models.baselines(trB, teB)
    predT = {
        **bT,
        "Poisson GLM (forecasting)": models.predict(glm_B, teB),
        "XGBoost (lag features)": models.predict_tree(xgb, teB),
        "Random forest (lag features)": models.predict_tree(rf, teB),
    }
    mT = evaluate.point_metrics(arr(teB["cnt"]), predT)
    save(mT.round(2), "07_frozen_test_metrics.csv")

    summary["taskB"] = {
        "dispersion": disp_B,
        "validation": mB.to_dict("records"),
        "backtest_folds": bt.to_dict("records"),
        "backtest_summary": bt_summary.to_dict("records"),
        "frozen_test": mT.to_dict("records"),
    }
    print("\nTask B (deployable forecasting) validation")
    print(mB.round(2).to_string(index=False))
    print("\nTest period")
    print(mT.round(2).to_string(index=False))

    # ------------------------------------------------------------ diagnostics
    yv = arr(vaB["cnt"])
    pv = models.predict(glm_B, vaB)
    diag = vaB[["hr", "temp", "cnt"]].copy()
    diag["pred"] = pv
    diag["ratio"] = diag["cnt"] / np.maximum(pv, 1e-6)
    diag["bias"] = pv - diag["cnt"]
    diag["temp_bin"] = pd.cut(diag["temp"], [0, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    bias_hour = diag.groupby("hr").agg(Median_ratio=("ratio", "median"),
                                      Mean_bias=("bias", "mean")).round(3)
    bias_temp = diag.groupby("temp_bin", observed=True).agg(
        Median_ratio=("ratio", "median"), Mean_bias=("bias", "mean"),
        N=("ratio", "size")).round(3)
    bias_hour.to_csv(RESULTS / "08_bias_by_hour.csv")
    bias_temp.to_csv(RESULTS / "09_bias_by_temperature.csv")
    daily_ratio = diag.groupby(diag.index.normalize())["ratio"].median()

    var_check = pd.DataFrame([
        {"Column": "yr", "Distinct values in validation": int(vaB["yr"].nunique()),
         "Share of rows at modal value": float(vaB["yr"].value_counts(normalize=True).iloc[0])},
        {"Column": "season", "Distinct values in validation": int(vaB["season"].nunique()),
         "Share of rows at modal value": float(vaB["season"].value_counts(normalize=True).iloc[0])},
        {"Column": "mnth", "Distinct values in validation": int(vaB["mnth"].nunique()),
         "Share of rows at modal value": float(vaB["mnth"].value_counts(normalize=True).iloc[0])},
    ])
    save(var_check.round(3), "10_validation_feature_variation.csv")

    # ------------------------------------------------------------- conformal
    cal_start = split.train.index.min() + pd.Timedelta(
        days=int(round((split.train_end - split.train.index.min()).days * 0.80))
    )
    fit_df = data.modelling_frame(trB[trB.index < cal_start], "forecasting")
    cal_df = data.modelling_frame(trB[trB.index >= cal_start], "forecasting")
    cal_model = models.fit_glm(fit_df, models.FORECAST_FORMULA)
    alpha_conf = models.estimate_nb_alpha(cal_model, fit_df)

    y_cal, mu_cal = arr(cal_df["cnt"]), models.predict(cal_model, cal_df)
    y_va, mu_va = arr(vaB["cnt"]), models.predict(cal_model, vaB)
    y_te, mu_te = arr(teB["cnt"]), models.predict(cal_model, teB)
    sigmas = conformal.sigma_functions(alpha_conf)
    s_abs, s_poi, s_nb = sigmas["Absolute"], sigmas["Poisson-scaled"], sigmas["NB-scaled"]

    cal_groups = evaluate.hour_group(arr(cal_df["hr"]))
    va_groups = evaluate.hour_group(arr(vaB["hr"]))
    te_groups = evaluate.hour_group(arr(teB["hr"]))
    va_hours, te_hours = arr(vaB["hr"]), arr(teB["hr"])

    def build(y_eval, mu_eval, groups, hours, tag):
        methods = [
            conformal.split_conformal(y_cal, mu_cal, y_eval, mu_eval, s_abs,
                                      LEVEL, "Split / absolute"),
            conformal.split_conformal(y_cal, mu_cal, y_eval, mu_eval, s_poi,
                                      LEVEL, "Split / Poisson-scaled"),
            conformal.split_conformal(y_cal, mu_cal, y_eval, mu_eval, s_nb,
                                      LEVEL, "Split / NB-scaled"),
            conformal.rolling_conformal(y_cal, mu_cal, y_eval, mu_eval, s_nb,
                                        LEVEL, name="Rolling / NB-scaled"),
            conformal.aci_conformal(y_cal, mu_cal, y_eval, mu_eval, s_nb,
                                    LEVEL, name="ACI / NB-scaled"),
            conformal.mondrian_conformal(y_cal, mu_cal, y_eval, mu_eval, s_nb,
                                         cal_groups, groups, LEVEL, "Mondrian / NB-scaled"),
        ]
        rows = []
        for m in methods:
            met = evaluate.interval_metrics(y_eval, m.lo, m.hi, ALPHA)
            peak = evaluate.peak_hour_mask(hours)
            rows.append({
                "Method": m.name, **met,
                "Peak-hour coverage": float(m.covered[peak].mean()),
                "Off-peak coverage": float(m.covered[~peak].mean()),
                "Worst-hour coverage": evaluate.worst_group_coverage(y_eval, m.lo, m.hi, hours),
            })
        tbl = pd.DataFrame(rows)
        save(tbl.round(4), f"11_conformal_{tag}.csv")
        by_hour = pd.DataFrame({"Hour": hours})
        for m in methods:
            by_hour[m.name] = m.covered
        by_hour = by_hour.groupby("Hour").mean().round(4)
        by_hour.to_csv(RESULTS / f"12_coverage_by_hour_{tag}.csv")
        return methods, tbl

    val_methods, val_tbl = build(y_va, mu_va, va_groups, va_hours, "validation")
    test_methods, test_tbl = build(y_te, mu_te, te_groups, te_hours, "test")

    # Selection rule fixed in advance: lowest validation Winkler score.
    chosen = str(val_tbl.sort_values("Winkler score").iloc[0]["Method"])

    acf_val = evaluate.residual_acf(y_cal, mu_cal, alpha_conf, nlags=24)
    acf_tbl = pd.DataFrame({"Lag": range(len(acf_val)), "ACF": acf_val})
    save(acf_tbl.round(4), "13_calibration_residual_acf.csv")

    summary["conformal"] = {
        "alpha_nb": alpha_conf,
        "calibration": {"start": str(cal_df.index.min()), "end": str(cal_df.index.max()),
                        "n": int(len(cal_df)), "mean_cnt": float(y_cal.mean())},
        "validation_mean_cnt": float(y_va.mean()),
        "test_mean_cnt": float(y_te.mean()),
        "acf_lag1": float(acf_val[1]), "acf_lag24": float(acf_val[24]),
        "chosen_method": chosen,
        "validation": val_tbl.to_dict("records"),
        "frozen_test": test_tbl.to_dict("records"),
    }
    print("\nConformal (validation)")
    print(val_tbl.round(3).to_string(index=False))
    print("\nChosen by validation Winkler:", chosen)
    print("ACF lag1", round(acf_val[1], 3), "lag24", round(acf_val[24], 3))

    summary["diagnostics"] = {
        "overall_mean_bias": float(diag["bias"].mean()),
        "median_actual_over_pred": float(diag["ratio"].median()),
        "bias_by_temperature": bias_temp.reset_index().astype(str).to_dict("records"),
        "validation_feature_variation": var_check.to_dict("records"),
    }

    # ----------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(8, 3.4))
    prof = df[df["observed"]].groupby(["hr", "workingday"])["cnt"].mean().unstack()
    ax.plot(prof.index, prof[1], marker="o", label="Working day", color="#185FA5")
    ax.plot(prof.index, prof[0], marker="o", label="Non-working day", color="#D85A30")
    ax.set_xticks(range(24)); ax.set_xlabel("Hour"); ax.set_ylabel("Mean hourly rentals")
    ax.set_title("Demand profile (full dataset)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGDIR / "demand_profile.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    wk = slice(0, 168)
    t = vaB.index[wk]
    ax.plot(t, yv[wk], color="#2C2C2A", lw=1.2, label="Actual")
    ax.plot(t, pv[wk], color="#185FA5", lw=1.2, label="Poisson GLM (forecasting)")
    ax.plot(t, bB["Seasonal naive (t-168h)"][wk], color="#BA7517", lw=1,
            alpha=0.85, label="Seasonal naive (t-168h)")
    ax.set_title("One week of deployable forecasts (validation)")
    ax.set_ylabel("Hourly rentals"); ax.legend(frameon=False)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIGDIR / "forecast_week.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for m in [val_methods[0], val_methods[2], val_methods[4]]:
        ax.plot(range(24),
                pd.Series(m.covered).groupby(va_hours).mean().values,
                marker="o", ms=3.5, label=m.name)
    ax.axhline(LEVEL, color="#A32D2D", ls="--", lw=1, label="Target 90%")
    ax.set_xticks(range(24)); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hour"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Conditional coverage by hour (validation)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "coverage_by_hour.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.scatter(mu_va, val_methods[0].hi - val_methods[0].lo, s=5, alpha=0.25,
               color="#378ADD", label="Split / absolute")
    ax.scatter(mu_va, val_methods[2].hi - val_methods[2].lo, s=5, alpha=0.25,
               color="#0F6E56", label="Split / NB-scaled")
    ax.set_xlabel("Predicted demand"); ax.set_ylabel("Interval width")
    ax.set_title("Variance-aware intervals widen at high demand")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "width_vs_prediction.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    one_week = vaB.index[wk]
    for ax, m in zip(axes, [val_methods[0], val_methods[4]]):
        ax.plot(one_week, yv[wk], color="#2C2C2A", lw=1.1, label="Actual")
        ax.fill_between(one_week, m.lo[wk], m.hi[wk], alpha=0.3, color="#185FA5",
                        label="90% interval")
        ax.set_title(m.name); ax.set_ylabel("Hourly rentals")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xlabel("Date")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIGDIR / "intervals_validation.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(daily_ratio.index, daily_ratio.values, color="#185FA5", lw=1)
    axes[0].axhline(1, color="#A32D2D", ls="--", lw=1)
    axes[0].set_title("Daily median actual / predicted")
    axes[0].set_ylabel("Ratio")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(range(1, 13), acf_val[1:13], color="#0F6E56")
    axes[1].axhline(0, color="#2C2C2A", lw=0.8)
    axes[1].set_title("Autocorrelation of calibration residuals")
    axes[1].set_xlabel("Lag (hours)")
    fig.tight_layout(); fig.savefig(FIGDIR / "bias_and_dependence.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    pivot = bt.pivot(index="Fold", columns="Model", values="MAE")
    # Baselines in neutral greys (light = weakest), models in distinct colours.
    bt_colours = {
        "Global mean": "#D3D1C7",
        "Hour x workingday mean": "#B4B2A9",
        "Naive (t-1h)": "#9C9A92",
        "Naive (t-24h)": "#888780",
        "Seasonal naive (t-168h)": "#5F5E5A",
        "XGBoost (lag features)": "#185FA5",
        "Poisson GLM (forecasting)": "#0F6E56",
    }
    order = [m for m in bt_colours if m in pivot.columns]
    order += [m for m in pivot.columns if m not in order]
    pivot = pivot[order]
    pivot.plot(kind="bar", ax=ax, width=0.8,
               color=[bt_colours.get(m, "#BA7517") for m in order])
    ax.set_ylabel("MAE"); ax.set_xlabel("")
    ax.set_title("Rolling-origin backtest (3 non-overlapping 28-day folds)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGDIR / "backtest.png"); plt.close(fig)

    summary["figures"] = sorted(p.name for p in FIGDIR.glob("*.png"))
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    print("\nAll results written to", RESULTS)


if __name__ == "__main__":
    main()
