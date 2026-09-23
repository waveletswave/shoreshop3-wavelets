"""Checks of the wavelet toolkit on synthetic signals with known answers."""

import numpy as np
import pytest

from shoreshop3 import wavelets as wv


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def sine(n, period, dt=1.0, lag=0.0):
    t = np.arange(n) * dt
    return np.sin(2 * np.pi * (t - lag) / period)


def test_scales_and_coi_shape():
    res = wv.cwt(sine(500, 25), 1.0)
    assert res.coeffs.shape == (res.scales.size, 500)
    assert np.allclose(np.diff(np.log2(res.scales)), 1 / 12)
    assert res.periods[0] == pytest.approx(2 * wv.fourier_factor())
    assert np.allclose(res.coi, res.coi[::-1])  # symmetric
    assert res.coi.max() < res.periods.max()


def test_peak_period_recovered(rng):
    for period in (12.0, 40.0, 90.0):
        x = sine(2000, period) + 0.3 * rng.standard_normal(2000)
        gws, _ = wv.global_spectrum(wv.cwt(x, 1.0))
        res = wv.cwt(x, 1.0)
        assert res.periods[np.nanargmax(gws)] == pytest.approx(period, rel=0.06)


def test_units_do_not_matter(rng):
    x = rng.standard_normal(400)
    a, b = wv.cwt(x, 1.0), wv.cwt(x, 7.0)  # days vs weeks labelling
    assert np.allclose(a.periods * 7.0, b.periods)
    assert np.allclose(a.normalized_power(), b.normalized_power())


def test_full_reconstruction_and_variance(rng):
    t = np.arange(3000)
    x = 5 + 2 * np.sin(2 * np.pi * t / 60) + np.sin(2 * np.pi * t / 400) + 0.5 * rng.standard_normal(3000)
    res = wv.cwt(x, 1.0)
    xr = wv.reconstruct(res)
    inner = slice(300, -300)
    assert np.corrcoef(x[inner], xr[inner])[0, 1] > 0.99
    assert abs(xr.mean() - x.mean()) < 0.05
    # T&C eq. 14: wavelet variance ~ series variance (small loss below 2 dt)
    total = wv.band_variance_fraction(res, (0, np.inf))
    assert 0.9 < total < 1.05


def test_band_reconstruction_separates_components():
    n = 4000
    fast, slow = sine(n, 20), 2 * sine(n, 300)
    res = wv.cwt(fast + slow, 1.0)
    inner = slice(700, -700)
    rf = wv.reconstruct(res, (8, 50))
    rs = wv.reconstruct(res, (150, 600))
    assert np.corrcoef(rf[inner], fast[inner])[0, 1] > 0.99
    assert np.corrcoef(rs[inner], slow[inner])[0, 1] > 0.99
    assert rs[inner].std() == pytest.approx(slow[inner].std(), rel=0.08)
    # bands are additive
    lo, hi = wv.reconstruct(res, (0, 100)), wv.reconstruct(res, (100, np.inf))
    assert np.allclose(lo + hi, wv.reconstruct(res, (0, np.inf)))


def test_detrend_and_standardize_roundtrip(rng):
    t = np.arange(1500)
    x = 0.01 * t + np.sin(2 * np.pi * t / 50) + 0.2 * rng.standard_normal(1500)
    res = wv.cwt(x, 1.0, detrend=True, standardize=True)
    xr = wv.reconstruct(res)
    assert np.corrcoef(x[200:-200], xr[200:-200])[0, 1] > 0.99


def test_nan_rejected():
    x = sine(100, 10)
    x[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        wv.cwt(x, 1.0)


def test_gap_fraction_masks_short_periods_near_gaps():
    x = sine(2000, 30)
    inv = np.zeros(2000, bool)
    inv[980:1020] = True  # a 40-sample gap
    res = wv.cwt(x, 1.0, invalid=inv)
    frac = res.gap_fraction()
    assert frac.shape == res.coeffs.shape
    assert frac[0, 1000] > 0.95  # shortest period: all energy on the gap
    j_long = np.argmin(np.abs(res.periods - 400))
    assert frac[j_long, 1000] < 0.1  # a 40-day gap barely touches a 400-day wavelet
    valid = res.valid_mask()
    assert not valid[0, 1000] and valid[j_long, 1000]
    assert valid[0, 500]  # far from the gap and the edges
    assert not valid[:, 0].any()  # edges stay in the cone of influence
    no_gaps = wv.cwt(x, 1.0)
    assert np.array_equal(no_gaps.valid_mask(), no_gaps.periods[:, None] <= no_gaps.coi[None, :])


def test_power_floor_zeroes_coherence_without_variance(rng):
    n = 1500
    smooth = sine(n, 200)  # no short-period variance at all
    noisy = smooth + 0.3 * rng.standard_normal(n)
    coh = wv.wavelet_coherence(noisy, smooth, 1.0)
    short = coh.periods < 8
    assert coh.rsq[short].mean() < 0.05
    j = np.argmin(np.abs(coh.periods - 200))
    assert coh.rsq[j, coh.valid_mask()[j]].mean() > 0.95


def test_ar1_estimate(rng):
    x = wv.ar1_noise(20000, 0.7, rng)
    assert wv.ar1(x) == pytest.approx(0.7, abs=0.02)
    assert x.std() == pytest.approx(1.0, rel=0.05)


def test_power_significance_false_positive_rate(rng):
    alpha = 0.6
    rates = []
    for _ in range(5):
        x = wv.ar1_noise(3000, alpha, rng)
        res = wv.cwt(x, 1.0)
        sig = wv.power_significance(res, wv.ar1(x), 0.95)
        m = res.valid_mask()
        rates.append(((res.normalized_power() / sig[:, None]) > 1)[m].mean())
    assert 0.02 < np.mean(rates) < 0.09  # nominal 5 %


def test_z2_matches_torrence_compo():
    assert wv._z2(0.95) == pytest.approx(3.999, abs=0.002)


def test_coherence_self_and_phase_lag():
    n, period, lag = 3000, 60.0, 6.0
    x = sine(n, period)
    y = sine(n, period, lag=lag)  # y lags x
    noisy = x + 0.1 * np.random.default_rng(0).standard_normal(n)
    exact = wv.wavelet_coherence(noisy, noisy, 1.0, power_floor=0.0)
    assert np.allclose(exact.rsq, 1.0, atol=1e-9)
    assert wv.wavelet_coherence(noisy, noisy, 1.0).rsq.min() > 0.99  # default floor: tiny effect
    coh = wv.wavelet_coherence(x, y, 1.0)
    j = np.argmin(np.abs(coh.periods - period))
    m = coh.valid_mask()[j]
    phase = np.degrees(wv._circmean(coh.phase[j][m]))
    assert phase == pytest.approx(360 * lag / period, abs=2.0)  # + => x leads


def test_coherence_scale_invariant(rng):
    x = rng.standard_normal(800)
    y = x + rng.standard_normal(800)
    a = wv.wavelet_coherence(x, y, 1.0)
    b = wv.wavelet_coherence(10 * x + 3, 0.1 * y - 2, 1.0)
    assert np.allclose(a.rsq, b.rsq)


def test_coherence_significance_controls_false_positives(rng):
    n = 600
    sig = wv.coherence_significance(n, 1.0, 0.5, 0.5, n_surrogates=60, seed=1)
    assert sig.shape == wv.make_scales(n, 1.0).shape
    assert np.nanmin(sig) > 0.3 and np.nanmax(sig) <= 1.0
    fp = []
    for _ in range(6):
        x, y = wv.ar1_noise(n, 0.5, rng), wv.ar1_noise(n, 0.5, rng)
        coh = wv.wavelet_coherence(x, y, 1.0)
        m = coh.valid_mask() & np.isfinite(sig)[:, None]
        fp.append((coh.rsq >= sig[:, None])[m].mean())
    assert np.mean(fp) < 0.12  # nominal 5 %, small-sample tolerance
    # cached: second call is identical and instant
    again = wv.coherence_significance(n, 1.0, 0.5, 0.5, n_surrogates=60, seed=1)
    assert np.array_equal(sig, again, equal_nan=True)


def test_significance_disk_cache(tmp_path):
    a = wv.coherence_significance(300, 1.0, 0.3, 0.3, n_surrogates=10, seed=3, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("wtc_sig_*.npy"))) == 1
    wv._SIG_CACHE.clear()
    b = wv.coherence_significance(300, 1.0, 0.3, 0.3, n_surrogates=10, seed=3, cache_dir=tmp_path)
    assert np.array_equal(a, b, equal_nan=True)


def test_band_skill_identifies_strengths(rng):
    n = 6000
    seasonal, event = 10 * sine(n, 365.25), 3 * sine(n, 20)
    obs = seasonal + event + 0.5 * rng.standard_normal(n)
    model_a = seasonal + 0.2 * sine(n, 20)  # right seasonal cycle, misses events
    model_b = 10 * sine(n, 365.25, lag=30) + event  # events right, seasonal 30 days late
    bands = {"event": (10, 40), "seasonal": (250, 500)}
    a = wv.band_skill(obs, model_a, 1.0, bands).set_index("band")
    b = wv.band_skill(obs, model_b, 1.0, bands).set_index("band")
    assert a.loc["seasonal", "nse"] > 0.9 and a.loc["event", "nse"] < 0.2
    assert a.loc["event", "std_ratio"] < 0.2
    assert b.loc["event", "nse"] > 0.8
    assert b.loc["seasonal", "lag"] == pytest.approx(30, abs=4)  # model lags
    assert b.loc["seasonal", "phase_deg"] > 0
    assert a.loc["seasonal", "var_frac_obs"] > a.loc["event", "var_frac_obs"]


def test_band_skill_nan_when_band_outside_record():
    x = sine(300, 20)
    df = wv.band_skill(x, x, 1.0, {"too long": (5000, 9000)})
    assert df.loc[0, "n_scales"] == 0 and np.isnan(df.loc[0, "nse"])


def test_compare_models_tidy_output(rng):
    n = 1500
    obs = sine(n, 50) + 0.3 * rng.standard_normal(n)
    df = wv.compare_models(obs, {"A": sine(n, 50), "B": sine(n, 50, lag=10)}, 1.0,
                           {"b1": (30, 80)}, n_surrogates=20)
    assert list(df["model"]) == ["A", "B"]
    assert df["sig_frac"].notna().all()
    assert df.set_index("model").loc["B", "lag"] == pytest.approx(10, abs=2)


def test_matches_pycwt():
    """Cross-check against pycwt (same scales, padding and smoothing kernel)."""
    pycwt = pytest.importorskip("pycwt")
    from pycwt.helpers import rect

    rng = np.random.default_rng(0)
    n, dj = 700, 1 / 12
    x = sine(n, 40) + 0.4 * rng.standard_normal(n)
    y = sine(n, 40, lag=4) + 0.4 * rng.standard_normal(n)
    mine = wv.cwt(x, 1.0, dj=dj, pad="pow2")
    W, sj, *_ = pycwt.cwt(x - x.mean(), 1.0, dj, 2.0, mine.scales.size - 1, pycwt.Morlet(6))
    assert np.allclose(sj, mine.scales)
    assert np.abs(W - mine.coeffs).max() < 1e-8 * np.abs(W).max()
    # pycwt's scale boxcar is 2x Grinsted's; pass it explicitly to compare the rest
    kernel = rect(int(np.round(0.6 / dj * 2)), normalize=True)
    coh = wv.wavelet_coherence(x, y, 1.0, dj=dj, pad="pow2", scale_kernel=kernel, power_floor=0.0)
    WCT, *_ = pycwt.wct(x, y, 1.0, dj=dj, s0=2.0, J=mine.scales.size - 1, sig=False)
    assert np.abs(WCT - coh.rsq).max() < 1e-6


def test_grinsted_kernel():
    k = wv._scale_kernel(1 / 12)
    assert k.size == 9 and k.sum() == pytest.approx(1.0)
    assert k[0] == pytest.approx(0.6 / 8.2)


def test_scale_convolution_matches_conv2d(rng):
    from scipy.signal import convolve2d

    T = rng.standard_normal((40, 30)) + 1j * rng.standard_normal((40, 30))
    for kernel in (wv._scale_kernel(1 / 12), np.array([0.25, 0.5, 0.5, 0.25]), np.ones(1)):
        expected = convolve2d(T, kernel[:, None], mode="same")
        assert np.allclose(wv._convolve_scales(T, kernel), expected)
