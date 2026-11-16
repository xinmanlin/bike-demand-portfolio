# Hourly bike-sharing demand: interpretable count models and uncertainty under temporal shift

**Scope, stated up front.** This is a study of *hourly demand modelling and
uncertainty quantification*, not of weather forecasting. Task A models demand
conditional on weather observed **during the same hour** (an explanatory,
optimistic upper bound). Task B is a **deployable forecast** whose features are
all known before hour *t* -- calendar variables and lagged demand only. No
contemporaneous weather enters Task B.

## Design choices

1. **Transparent evaluation.** Model and interval choices are made on the
   validation period only; the test period is held out and not used for selection.
2. **A one-step-ahead forecasting task** (features known by the end of hour
   *t-1*), compared with three naive persistence baselines (t-1h, t-24h,
   t-168h) and checked on three non-overlapping 28-day backtest folds.
3. **Dispersion-aware inference.** Pearson dispersion is 11.56
   (severe over-dispersion), so standard errors are dispersion-corrected
   (quasi-Poisson) throughout. The quadratic temperature effect is interpreted
   jointly: demand peaks at normalised temperature 0.77, and moving
   from 0.5 to 0.7 multiplies expected demand by 1.19x.
4. **Uncertainty is the main object.** Six conformal variants are compared
   with the Winkler score (which penalises both misses and width), including
   rolling and adaptive (ACI) calibration designed for dependent data.
5. **Assumptions are tested, then reported honestly.** Calibration residuals
   have lag-1 autocorrelation 0.733 and mean demand moves
   229.4 -> 281.6 -> 236.9 across periods
   (a substantial distribution shift).
   The exchangeability assumption behind split conformal is therefore unlikely
   to hold; empirical coverage below is a measurement, **not a coverage guarantee**.

## Headline results (all reproducible)

| Result | Number |
|---|---|
| Poisson GLM test MAE (deployable features) | **41.79** |
| Global-mean baseline test MAE | 168.89 |
| Best naive baseline, seasonal t-168h, test MAE | 66.48 |
| Naive t-24h / t-1h test MAE | 77.77 / 81.26 |
| XGBoost test MAE (lag features) | 55.21 |
| Best interval method (lowest validation Winkler) | Split / Poisson-scaled |
| Its validation coverage / worst-hour coverage | 92.8% / 81.7% |
| Split-absolute peak-hour coverage (for contrast) | 78.0% |
| Test-period Winkler: pre-selected method vs ACI | 280.5 vs 273.5 |
| Missing hours in the raw file (mostly 02:00-05:00) | 165 of 17,544 grid hours |

**A note on interval selection.** Selection was fixed in advance as the lowest
validation Winkler score, which selects **Split / Poisson-scaled** (278.7),
with adaptive ACI essentially tied (279.1). On the
test period ACI in fact achieves the lowest Winkler score of all six methods
(273.5 vs 280.5 for the
pre-selected method), consistent with the idea that online calibration copes
better with drift. We report both plainly: two periods are not enough to claim
a significant win, and we did not re-select the method after seeing the test
numbers.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.run_analysis          # reproduces every table and figure
python report/build_docs.py          # regenerates this README and the report
python report/execute_notebook.py    # refreshes walkthrough outputs + HTML
python -m pytest -q                 # unit tests
```

`requirements.txt` pins the library versions used for the reported results.
Tree-model scores (random forest, XGBoost) can differ slightly across operating
systems because of floating-point differences; the GLM, baseline and interval
results are deterministic.

`data/raw/hour.csv` is the unmodified UCI file, included for one-command
reproduction. See [CITATION.md](CITATION.md) for provenance and references.

## Validation protocol

| Stage | Used for |
|---|---|
| Train (70%, day-aligned) | fitting |
| Rolling-origin backtest (3 x 28 days, inside train) | independent stability check |
| Validation (15%) | model / interval selection |
| Test (15%, held out) | final evaluation; not used for model or interval selection |

Because the calendar grid is incomplete (165 missing hours), lags
are built on a reindexed full hourly grid so `t-168h` is exactly one week ago.
The raw file contains no zero counts (minimum 1) and
59% of missing hours fall between 02:00 and 05:00, so most
missing hours probably represent hours with zero rentals rather than data loss.
36 of the 40 test-period gaps fall on 29-30 October 2012, which
coincides with Hurricane Sandy (not verified against operator records).

## Structure

```
data/raw/hour.csv        UCI source data
src/data.py              loading, integrity, leakage-safe features, splits
src/models.py            GLMs, dispersion correction, baselines, tree models
src/conformal.py         split / rolling / ACI / Mondrian conformal methods
src/evaluate.py          point metrics, Winkler score, coverage diagnostics
src/run_analysis.py      one-command reproduction of the whole study
report/                  technical report (HTML) and this generator
results/                 tables, figures, summary.json
notebooks/               results walkthrough notebook
tests/                   unit tests
```

## Limitations (please read)

- Task A uses same-hour observed weather; a production system needs weather
  **forecasts**, so those numbers are an optimistic upper bound.
- The exchangeability assumption is unlikely to hold (lag-1 autocorrelation
  0.73, strong drift). Coverage figures are empirical validation /
  test-period measurements, not guarantees.
  Rolling / ACI calibration mitigate but do not remove this.
- Online methods update calibration with realised counts up to *t-1*; a live
  deployment needs a reliable real-time count feed.
- One city, two years, one public teaching dataset. There is no claim of
  novelty in the data or of external validity to other systems.
- Intervals are symmetric truncations at zero; asymmetric count intervals
  would be a natural next step.
- Missing hours are excluded rather than imputed as zero; this slightly
  under-represents very low overnight demand.
- GLM and tree models share the same historical information but not an
  identical feature representation (trees also receive t-24h, month and
  weekday), so the comparison is not a perfectly matched feature-set experiment.

## Full write-up

Open [report/technical_report.html](report/technical_report.html) for the
technical report, or [notebooks/01_results_walkthrough.ipynb](notebooks/01_results_walkthrough.ipynb)
for a short walkthrough.
