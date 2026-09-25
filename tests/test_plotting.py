import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from shoreshop3 import plotting as sp  # noqa: E402
from shoreshop3 import wavelets as wv  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_figures_render_with_dates(tmp_path):
    sp.use_style()
    n = 800
    time = pd.date_range("2000-01-01", periods=n, freq="D").to_numpy()
    rng = np.random.default_rng(0)
    obs = np.sin(np.arange(n) / 20) + 0.3 * rng.standard_normal(n)
    mod = np.sin(np.arange(n) / 20 - 0.3)
    res = wv.cwt(obs, 1.0)
    coh = wv.wavelet_coherence(obs, mod, 1.0)
    sig = wv.coherence_significance(n, 1.0, wv.ar1(obs), wv.ar1(mod), n_surrogates=10)
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    sp.plot_series(time, obs, {"Model A": mod}, ax=axes[0])
    sp.plot_power(res, time=time, ax=axes[1], alpha=wv.ar1(obs))
    sp.plot_coherence(coh, time=time, ax=axes[2], sig=sig)
    sp.plot_global_spectra({"obs": res, "Model A": wv.cwt(mod, 1.0)}, ax=axes[3])
    fig.savefig(tmp_path / "wavelets.png")
    assert (tmp_path / "wavelets.png").stat().st_size > 10_000


def test_skill_heatmap_metrics(tmp_path):
    df = pd.DataFrame({
        "model": ["A", "A", "B", "B"], "band": ["x", "y", "x", "y"],
        "nse": [0.9, -0.5, np.nan, 0.2], "std_ratio": [1.0, 0.5, 2.0, 0.0],
        "lag": [1.0, -3.0, 0.0, 2.0], "mean_rsq": [0.8, 0.1, 0.5, 0.4],
    })
    df["phase_deg"] = [10.0, -40.0, np.nan, 5.0]
    for metric in ("nse", "std_ratio", "lag", "mean_rsq", "phase_deg"):
        fig, ax = plt.subplots()
        sp.plot_skill_heatmap(df, metric, ax=ax)
        fig.savefig(tmp_path / f"{metric}.png")
    fig, ax = plt.subplots()
    sp.plot_skill_heatmap(df, "phase_deg", text_metric="lag", fmt="{:.0f} d", ax=ax)
    texts = [t.get_text() for t in ax.texts]
    assert "1 d" in texts and "–" in texts


def test_negative_zero_is_not_printed():
    assert sp._fmt_value(-0.001, "{:.2f}") == "0.00"
    assert sp._fmt_value(-0.2, "{:.2f}") == "-0.20"
    assert sp._fmt_value(-12.3, lambda v: f"{v:.0f}") == "-12"


def test_heatmap_band_labels_and_callable_format():
    df = pd.DataFrame({"model": ["A", "A"], "band": ["x", "y"], "nse": [-15.2, 0.5]})
    fig, ax = plt.subplots()
    sp.plot_skill_heatmap(df, "nse", ax=ax, band_labels=["X\n50 % usable", "Y\n10 % usable"],
                          fmt=lambda v: f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["X\n50 % usable", "Y\n10 % usable"]
    assert {"-15", "0.50"} <= {t.get_text() for t in ax.texts}
    with pytest.raises(ValueError, match="one label per band"):
        sp.plot_skill_heatmap(df, "nse", ax=ax, band_labels=["only one"])


def test_power_in_physical_units_shares_one_scale(tmp_path):
    n = 512
    rng = np.random.default_rng(2)
    big = 10 * np.sin(np.arange(n) / 6) + rng.standard_normal(n)
    small = 0.1 * big
    wb, ws = wv.cwt(big, 1.0), wv.cwt(small, 1.0)
    levels = sp.power_levels([wb])
    assert np.allclose(np.diff(levels), 0.5)
    fig, axes = plt.subplots(1, 2)
    cf_b = sp.plot_power(wb, ax=axes[0], physical=True, levels=levels, colorbar=False)
    cf_s = sp.plot_power(ws, ax=axes[1], physical=True, levels=levels)  # colour bar in m², 2**k labels
    assert np.allclose(cf_b.levels, cf_s.levels)
    # same shape, 100x less power: log2 power shifted by log2(100) everywhere
    lp = lambda r: np.log2(r.normalized_power() * r.variance)  # noqa: E731
    assert np.allclose(lp(wb) - lp(ws), np.log2(100.0))
    labels = [t.get_text() for t in fig.axes[-1].get_xticklabels() + fig.axes[-1].get_yticklabels()]
    assert any(lab in {"1", "4", "16", "64", "256"} for lab in labels)
    fig.savefig(tmp_path / "physical.png")


def test_spectral_ratio_and_global_power_panels(tmp_path):
    sp.use_style()
    n = 600
    rng = np.random.default_rng(1)
    obs = np.sin(np.arange(n) / 8) + 0.5 * rng.standard_normal(n)
    gap = np.zeros(n, bool)
    gap[200:240] = True
    wo = wv.cwt(obs, 1.0, invalid=gap)
    wm = wv.cwt(0.5 * np.sin(np.arange(n) / 8), 1.0)
    per, po, pm = wv.paired_global_spectra(wo, wm)
    ratios = pd.DataFrame([pm / po, 2 * pm / po], index=["m1", "m2"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sp.plot_spectral_ratio(per, ratios, ax=axes[0], unresolved_below=8.0, row_colors={"m1": sp.CATEGORICAL[0]})
    assert any("not resolved" in t.get_text() for t in axes[0].texts)
    bands = {"short": (8.0, 32.0), "long": (32.0, 128.0)}
    sp.plot_global_power(wo, x=obs, ax=axes[1], alpha=wv.ar1(obs), bands=bands)
    shares = [t.get_text() for t in axes[1].texts]
    assert len(shares) == 2 and all(s.endswith("%") for s in shares)
    fig2, ax2 = plt.subplots()
    sp.plot_global_power(wo, ax=ax2, bands=bands)  # without the series: band limits only, no shares
    assert not ax2.texts
    plt.close(fig2)
    assert axes[1].yaxis_inverted()  # long periods at the bottom, like the power map
    fig.savefig(tmp_path / "panels.png")
    assert (tmp_path / "panels.png").stat().st_size > 5_000


def test_model_colors_limit():
    assert sp.model_colors(["a", "b"]) == {"a": sp.CATEGORICAL[0], "b": sp.CATEGORICAL[1]}
    with pytest.raises(ValueError, match="facet"):
        sp.model_colors([str(i) for i in range(9)])


def test_global_spectra_units_do_not_depend_on_standardising():
    x = 7.0 * np.random.default_rng(3).standard_normal(600)  # variance ~50
    fig, ax = plt.subplots()
    sp.plot_global_spectra({"obs": wv.cwt(x, 1.0), "std": wv.cwt(x, 1.0, standardize=True)}, ax=ax)
    a, b = ax.lines[0].get_ydata(), ax.lines[1].get_ydata()
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[ok], b[ok])
