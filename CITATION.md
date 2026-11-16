# Data provenance and references

## Dataset

UCI Machine Learning Repository, **Bike Sharing Dataset** (Washington, D.C.,
Capital Bikeshare, hourly counts 2011-01-01 to 2012-12-31).

- Fanaee-T, H., & Gama, J. (2014). *Event labeling combining ensemble detectors
  and background knowledge.* Progress in Artificial Intelligence, 2(2-3), 113-127.
- UCI repository page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- DOI: https://doi.org/10.24432/C5W894

`data/raw/hour.csv` is an unmodified copy of the `hour.csv` file distributed in
the UCI archive, included so the study reproduces with a single command.
The dataset contains **165 missing hours** on a 17,544-hour calendar grid;
`src/data.py` reindexes onto the full grid before computing lagged features.

## Methods cited in the report

- Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear models.
  *Journal of the Royal Statistical Society A*, 135(3), 370-384.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
  Chapman & Hall. (quasi-Poisson / dispersion-corrected standard errors)
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a
  Random World.* Springer. (split conformal prediction)
- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under
  distribution shift. *NeurIPS 34*.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: a gentle
  introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591.
- Winkler, R. L. (1972). A decision-theoretic approach to interval estimation.
  *Journal of the American Statistical Association*, 67(337), 187-191.
