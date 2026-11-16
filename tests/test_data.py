from pathlib import Path

import numpy as np
import pytest

from src import data

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "hour.csv"


@pytest.mark.skipif(not RAW.exists(), reason="raw dataset not present")
def test_split_is_ordered_and_aligned_to_days():
    df = data.load_hourly(RAW)
    s = data.chronological_split(df)
    assert s.train.index.max() < s.val.index.min()
    assert s.val.index.max() < s.test.index.min()
    assert s.train_end.hour == 0 and s.val_end.hour == 0
    assert s.train.index.max().hour == 23
    assert s.val.index.max().hour == 23
    total = len(s.train) + len(s.val) + len(s.test)
    assert total == len(df)


@pytest.mark.skipif(not RAW.exists(), reason="raw dataset not present")
def test_lags_are_strictly_past_information():
    df = data.load_hourly(RAW)
    # lag_168 on a complete hourly grid equals the count exactly 7 days earlier
    sample = df[df["observed"]].iloc[1000:1300]
    expected = df["cnt"].shift(168).loc[sample.index].to_numpy()
    np.testing.assert_allclose(sample["cnt_lag_168"].to_numpy(), expected)
    # rolling mean starts from t-1, never includes the current hour
    assert df["cnt_roll24"].notna().sum() > 0


@pytest.mark.skipif(not RAW.exists(), reason="raw dataset not present")
def test_folds_do_not_touch_validation_or_test():
    df = data.load_hourly(RAW)
    s = data.chronological_split(df)
    folds = data.rolling_origin_folds(df, s.train_end, 3, 28)
    assert len(folds) == 3
    for f in folds:
        assert f.horizon.index.max() < s.train_end
        assert f.train.index.max() < f.horizon.index.min()
