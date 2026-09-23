import numpy as np
import pandas as pd
import pytest

from shoreshop3.timeseries import regularize


def test_interp_on_datetimes_flags_long_gap():
    t = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05", "2020-03-01", "2020-03-02"])
    x = [0.0, 2.0, 4.0, 10.0, 11.0]
    reg = regularize(t, x, "1D", max_gap="10D")
    assert reg.dt == 1.0
    assert np.issubdtype(reg.time.dtype, np.datetime64)
    assert reg.time[0] == np.datetime64("2020-01-01")
    assert reg.values[1] == pytest.approx(1.0)  # interpolated day
    assert not reg.gap[:5].any()  # short gaps are only "filled"
    assert reg.filled[1] and not reg.filled[2]
    inside = (reg.time > np.datetime64("2020-01-06")) & (reg.time < np.datetime64("2020-02-29"))
    assert reg.gap[inside].all()
    assert not np.isnan(reg.values).any()


def test_bin_method_averages_dense_samples():
    t = np.array([0.0, 0.2, 0.4, 1.0, 1.1, 5.0, 6.0])
    x = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 10.0])
    reg = regularize(t, x, 1.0, method="bin", max_gap=2.5)
    assert reg.values[0] == pytest.approx(2.0)  # mean of 1, 2, 3
    assert reg.values[1] == pytest.approx(5.0)
    assert reg.filled[2:5].all()
    assert reg.gap[2:5].all()  # 4-step span between bins 1 and 5 > 2.5
    assert not reg.gap[[0, 1, 5, 6]].any()


def test_duplicates_nans_and_unsorted():
    t = [3.0, 1.0, 1.0, 2.0, np.nan]
    x = [3.0, 0.0, 2.0, np.nan, 5.0]
    reg = regularize(t, x, 1.0)
    assert np.allclose(reg.time, [1.0, 2.0, 3.0])
    assert np.allclose(reg.values, [1.0, 2.0, 3.0])


def test_outside_range_is_flagged():
    reg = regularize([10.0, 20.0], [1.0, 2.0], 5.0, start=0.0, end=30.0)
    assert np.allclose(reg.time, [0, 5, 10, 15, 20, 25, 30])
    assert reg.gap.tolist() == [True, True, False, False, False, True, True]


def test_timezone_aware_input():
    t = pd.date_range("2021-01-01", periods=4, freq="2D", tz="America/New_York")
    reg = regularize(t, [1, 2, 3, 4], "1D")
    assert len(reg.values) == 7
