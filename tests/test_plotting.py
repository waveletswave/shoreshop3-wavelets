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


def test_model_colors_limit():
    assert sp.model_colors(["a", "b"]) == {"a": sp.CATEGORICAL[0], "b": sp.CATEGORICAL[1]}
    with pytest.raises(ValueError, match="facet"):
        sp.model_colors([str(i) for i in range(9)])
