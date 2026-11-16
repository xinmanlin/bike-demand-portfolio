# Data

`raw/hour.csv` is the unmodified `hour.csv` from the UCI Machine Learning
Repository Bike Sharing Dataset (DOI 10.24432/C5W894). It is included so the
study reproduces with one command; see `CITATION.md` for the full reference.

- SHA-256: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`
- Size: 1,156,736 bytes; 17,379 observed hourly rows; 17 columns
- Coverage: 2011-01-01 00:00 to 2012-12-31 23:00 (17,544-hour grid, 165 missing hours)

No derived or synthetic data is used anywhere in this project. All features in
`src/data.py` are computed from this single file.
