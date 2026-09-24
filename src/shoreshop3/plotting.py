"""Matplotlib figures for wavelet diagnostics.

Colour roles (validated palette):

* magnitude (wavelet power, coherence, fractions): one-hue blue ramp, light -> dark
* signed skill (NSE, correlation, amplitude ratio, lag): blue <-> red with a
  neutral grey midpoint, equal lightness steps per arm
* model identity: fixed-order categorical hues (max 8; facet beyond that)
* observations: ink

Wavelet maps follow the Torrence & Compo / Grinsted layout: period on a log2
axis increasing downward, the cone of influence washed out, significance as a
thin ink contour. Phase arrows: right = in phase, left = anti-phase,
down = observations lead (model lags) by 90 deg, up = model leads by 90 deg.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from .wavelets import CoherenceResult, CWTResult, global_spectrum, power_significance

__all__ = [
    "SURFACE", "INK", "CATEGORICAL", "SEQUENTIAL", "DIVERGING",
    "use_style", "model_colors", "plot_series", "plot_power", "power_levels", "log2_ticks",
    "plot_coherence",
    "plot_global_spectra", "plot_global_power", "plot_skill_heatmap", "plot_spectral_ratio",
    "legend_patches", "compact_pdf_images",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
#: fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list("shoreshop3_blue", _BLUE_RAMP)
#: blue (low) <-> red (high); arms matched in OKLCH lightness (0.43 / 0.72 / 0.95)
DIVERGING = LinearSegmentedColormap.from_list(
    "shoreshop3_diverging", ["#184f95", "#6da7ec", NEUTRAL, "#e7837c", "#902224"])


def use_style() -> None:
    """Quiet chart chrome: hairline axes, no top/right spines, frameless legends."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "font.family": "sans-serif", "figure.dpi": 110,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in PDF/EPS
    })


def model_colors(names: Sequence[str]) -> dict[str, str]:
    """Stable colour per model, in the order given (keep one canonical order).

    More than 8 models cannot be told apart by hue: use small multiples or
    highlight one model at a time instead.
    """
    names = list(names)
    if len(names) > len(CATEGORICAL):
        raise ValueError(f"{len(names)} models > {len(CATEGORICAL)} colours: facet the figure "
                         "(one panel per model) or highlight a subset")
    return {n: CATEGORICAL[i] for i, n in enumerate(names)}


# --- helpers ---------------------------------------------------------------------------

def _xvalues(n: int, dt: float, time) -> tuple[np.ndarray, bool]:
    if time is None:
        return np.arange(n) * dt, False
    t = np.asarray(time)
    if len(t) != n:
        raise ValueError("time must have one entry per sample")
    if np.issubdtype(t.dtype, np.datetime64) or isinstance(time, pd.DatetimeIndex):
        return mdates.date2num(pd.DatetimeIndex(t).to_pydatetime()), True
    return t.astype(float), False


def _fmt_period(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:g}"


def _tick_spec(ticks, lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Ticks as a list of values or a {value: label} mapping, clipped to [lo, hi]."""
    if ticks is None:
        ticks = 2.0 ** np.arange(np.ceil(np.log2(lo)), np.floor(np.log2(hi)) + 1)
    if isinstance(ticks, Mapping):
        pairs = [(float(v), str(lab)) for v, lab in ticks.items()]
    else:
        pairs = [(float(v), _fmt_period(float(v))) for v in ticks]
    pairs = [(v, lab) for v, lab in pairs if lo <= v <= hi]
    return [v for v, _ in pairs], [lab for _, lab in pairs]


def _period_axis(ax, lo: float, hi: float, ticks, label: str, lim=None) -> None:
    """Log2 period axis, long periods at the bottom; ``lim`` = (min, max) crops it."""
    if lim is not None:
        lo, hi = max(lo, lim[0]), min(hi, lim[1])
    ax.set_yscale("log", base=2)
    ax.set_ylim(hi, lo)  # long periods at the bottom
    values, labels = _tick_spec(ticks, lo, hi)
    ax.set_yticks(values)
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylabel(label)


def _coi_wash(ax, x: np.ndarray, coi: np.ndarray, lo: float, hi: float) -> None:
    """Wash out the cone of influence (record edges) below its smooth boundary."""
    c = np.clip(coi, lo, hi)
    ax.fill_between(x, c, hi, color=SURFACE, alpha=0.72, linewidth=0, zorder=3)
    ax.plot(x, c, color=INK_2, linewidth=0.8, zorder=4)


@contextmanager
def compact_pdf_images():
    """Keep only the cropped pixels of each rasterised layer while a PDF is written.

    matplotlib's mixed-mode renderer hands the PDF backend a *view* into a
    full-figure raster for every rasterised artist, and the backend keeps it
    until the file is closed, so a figure with many rasterised panels can
    hold gigabytes. Copying the cropped image when it is drawn keeps memory
    proportional to the pixels actually used. Use around ``fig.savefig(...pdf)``.
    """
    from matplotlib.backends import backend_pdf

    renderer = getattr(backend_pdf, "RendererPdf", None)
    original = getattr(renderer, "draw_image", None)
    if original is None:  # unknown matplotlib layout: save as usual
        yield
        return

    def draw_image(self, gc, x, y, im, *args, **kwargs):
        return original(self, gc, x, y, np.array(im, copy=True), *args, **kwargs)

    renderer.draw_image = draw_image
    try:
        yield
    finally:
        renderer.draw_image = original


def _rasterize(artist) -> None:
    """Rasterise a dense filled layer in vector output (PDF); no effect on PNG."""
    if hasattr(artist, "set_rasterized"):
        artist.set_rasterized(True)
    for coll in getattr(artist, "collections", []) or []:  # matplotlib < 3.8 contour sets
        coll.set_rasterized(True)


def _gap_wash(ax, x: np.ndarray, per: np.ndarray, gap_mask: np.ndarray) -> None:
    """Wash out cells dominated by data gaps and outline them."""
    if gap_mask.any():
        bad = gap_mask.astype(float)
        _rasterize(ax.contourf(x, per, bad, levels=[0.5, 1.5], colors=[SURFACE], alpha=0.72, zorder=3))
        ax.contour(x, per, bad, levels=[0.5], colors=[INK_2], linewidths=0.8, zorder=4)


def _fmt_value(v: float, fmt) -> str:
    """Format a cell value (``fmt``: format string or callable); never prints '-0.00'."""
    text = fmt(v) if callable(fmt) else fmt.format(v)
    return text[1:] if text.startswith("-") and not any(c in "123456789" for c in text) else text


def _finish_x(ax, is_date: bool) -> None:
    if is_date:
        ax.xaxis_date()
        loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _colorbar(ax, mappable, label: str):
    cb = ax.figure.colorbar(mappable, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label(label, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=AXIS, labelcolor=INK_2)
    return cb


# --- figures ----------------------------------------------------------------------------

def plot_series(time, obs, models: Mapping[str, Sequence[float]] | None = None, *, ax=None,
                colors: Mapping[str, str] | None = None, obs_label: str = "Observed",
                ylabel: str = "Shoreline position (m)", title: str | None = None):
    """Observed series (ink) with model series (categorical colours)."""
    ax = ax or plt.gca()
    models = models or {}
    colors = dict(colors or model_colors(list(models)))
    x = np.asarray(time)
    ax.plot(x, obs, color=INK, linewidth=1.0, label=obs_label, zorder=3)
    for name, series in models.items():
        ax.plot(x, series, color=colors[name], linewidth=1.5, label=name, zorder=2)
    ax.grid(axis="y")
    ax.set_ylabel(ylabel)
    if models:
        ax.legend(loc="upper left", ncol=min(len(models) + 1, 5))
    if title:
        ax.set_title(title)
    return ax


def power_levels(results: Sequence[CWTResult], *, step: float = 0.5, lo_pct: float = 2.0,
                 hi_pct: float = 99.5, period_range: tuple[float, float] | None = None) -> np.ndarray:
    """Shared contour levels for :func:`plot_power` with ``physical=True``.

    log2 of |W|^2 in the squared units of the series, spanning the ``lo_pct``
    to ``hi_pct`` percentiles of the valid cells (outside COI and gaps) of the
    given transforms, optionally only for periods within ``period_range``
    (units of ``res.periods``). Pass the same levels to every panel so the
    colours compare.
    """
    def _cells(r):
        mask = r.valid_mask()
        if period_range is not None:
            mask &= ((r.periods >= period_range[0]) & (r.periods <= period_range[1]))[:, None]
        return np.log2(np.maximum(r.normalized_power() * r.variance, 1e-12))[mask]

    vals = np.concatenate([_cells(r) for r in results])
    lo = np.floor(np.percentile(vals, lo_pct) / step) * step
    hi = np.ceil(np.percentile(vals, hi_pct) / step) * step
    return np.arange(lo, max(hi, lo + step) + step / 2, step)


def plot_power(res: CWTResult, *, time=None, ax=None, alpha: float | None = None,
               level: float = 0.95, period_scale: float = 1.0, period_label: str = "Period",
               period_ticks: Sequence[float] | None = None, title: str | None = None,
               colorbar: bool = True, vmin_log2: float = -4.0, vmax_log2: float | None = None,
               period_lim: tuple[float, float] | None = None, physical: bool = False,
               levels: Sequence[float] | None = None, units: str = "m²"):
    """Wavelet power on a log2 colour scale, with optional AR1 significance contour.

    By default power is relative to the series' own variance (white noise = 1).
    ``physical=True`` shows |W|^2 in ``units`` instead, so series with
    different variance can share one colour scale (pass the same ``levels``,
    e.g. from :func:`power_levels`). ``period_lim`` = (min, max) crops the
    period axis (units of ``periods * period_scale``).
    """
    ax = ax or plt.gca()
    x, is_date = _xvalues(res.n, res.dt, time)
    p = res.normalized_power()
    lp = np.log2(np.maximum(p * res.variance if physical else p, 1e-12))
    if levels is None:
        if physical:
            levels = power_levels([res])
        else:
            if vmax_log2 is None:
                vmax_log2 = max(float(np.ceil(np.percentile(lp, 99.5))), vmin_log2 + 1.0)
            levels = np.arange(vmin_log2, vmax_log2 + 0.25, 0.5)
    per = res.periods * period_scale
    cf = ax.contourf(x, per, lp, levels=levels, cmap=SEQUENTIAL, extend="both")
    _rasterize(cf)
    if alpha is not None:
        thr = power_significance(res, alpha, level)
        ax.contour(x, per, p / thr[:, None], levels=[1.0], colors=INK, linewidths=0.9)
    _coi_wash(ax, x, res.coi * period_scale, per.min(), per.max())
    _gap_wash(ax, x, per, res.gap_mask())
    _period_axis(ax, per.min(), per.max(), period_ticks, period_label, period_lim)
    _finish_x(ax, is_date)
    if colorbar:
        cb = _colorbar(ax, cf, f"Wavelet power ({units})" if physical else "log2(power / variance)")
        if physical:
            log2_ticks(cb, levels)
    if title:
        ax.set_title(title)
    return cf


def log2_ticks(cb, levels: Sequence[float], every: int = 2) -> None:
    """Label a log2 colour bar with the values themselves (1, 4, 16, ...)."""
    lo, hi = float(np.min(levels)), float(np.max(levels))
    ticks = [k for k in range(int(np.ceil(lo)), int(np.floor(hi)) + 1) if k % every == 0]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{2.0 ** k:g}" if k >= -2 else f"1/{2 ** -k:d}" for k in ticks])


def plot_coherence(coh: CoherenceResult, *, time=None, ax=None, sig: np.ndarray | None = None,
                   arrows: bool = True, arrow_min_rsq: float = 0.5,
                   arrow_grid: tuple[int, int] = (36, 14), period_scale: float = 1.0,
                   period_label: str = "Period", period_ticks: Sequence[float] | None = None,
                   title: str | None = None, colorbar: bool = True,
                   period_lim: tuple[float, float] | None = None):
    """Squared wavelet coherence with significance contour and phase arrows.

    ``period_lim`` = (min, max) crops the period axis; arrows stay inside it.
    """
    ax = ax or plt.gca()
    n = coh.rsq.shape[1]
    x, is_date = _xvalues(n, coh.dt, time)
    per = coh.periods * period_scale
    cf = ax.contourf(x, per, coh.rsq, levels=np.linspace(0, 1, 11), cmap=SEQUENTIAL)
    _rasterize(cf)
    if sig is not None:
        ratio = coh.rsq / np.where(np.isfinite(sig), sig, np.inf)[:, None]
        ax.contour(x, per, ratio, levels=[1.0], colors=INK, linewidths=0.9)
    rows = np.arange(per.size)
    if period_lim is not None:
        rows = rows[(per >= period_lim[0]) & (per <= period_lim[1])]
    if arrows and rows.size:
        ti = np.unique(np.linspace(0, n - 1, arrow_grid[0] + 2)[1:-1].round().astype(int))
        pj = rows[np.unique(np.linspace(0, rows.size - 1, arrow_grid[1] + 2)[1:-1].round().astype(int))]
        T, P = np.meshgrid(ti, pj)
        ok = (coh.rsq[P, T] >= arrow_min_rsq) & coh.valid_mask()[P, T]
        if sig is not None:
            ok &= coh.rsq[P, T] >= np.nan_to_num(sig, nan=np.inf)[P]
        ph = coh.phase[P, T][ok]
        ax.quiver(x[T][ok], per[P][ok], np.cos(ph), -np.sin(ph), angles="uv", pivot="mid",
                  color=INK, scale_units="width", scale=arrow_grid[0] * 1.35, width=0.0022,
                  headwidth=4.5, headlength=4.5, headaxislength=4, zorder=5)
    _coi_wash(ax, x, coh.coi * period_scale, per.min(), per.max())
    _gap_wash(ax, x, per, coh.gap_mask())
    _period_axis(ax, per.min(), per.max(), period_ticks, period_label, period_lim)
    _finish_x(ax, is_date)
    if colorbar:
        _colorbar(ax, cf, "Coherence R²")
    if title:
        ax.set_title(title)
    return cf


def plot_global_spectra(results: Mapping[str, CWTResult], *, obs_key: str = "obs", ax=None,
                        obs_label: str = "Observed",
                        colors: Mapping[str, str] | None = None, period_scale: float = 1.0,
                        period_label: str = "Period", units: str = "m²",
                        period_ticks: Sequence[float] | None = None, title: str | None = None):
    """Time-averaged wavelet power (outside the COI) in physical units.

    Shows at which periods each model carries too much or too little variance
    compared with the observations (``obs_key``, drawn in ink).
    """
    ax = ax or plt.gca()
    models = [k for k in results if k != obs_key]
    colors = dict(colors or model_colors(models))
    for name, res in results.items():
        gws, _ = global_spectrum(res)
        power = gws * res.variance  # normalised power -> squared units, standardised or not
        is_obs = name == obs_key
        ax.plot(res.periods * period_scale, power, color=INK if is_obs else colors[name],
                linewidth=2.0 if is_obs else 1.5, label=obs_label if is_obs else name,
                zorder=3 if is_obs else 2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    per = next(iter(results.values())).periods * period_scale
    values, labels = _tick_spec(period_ticks, per.min(), per.max())
    ax.set_xticks(values)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.grid(which="major")
    ax.set_xlabel(period_label)
    ax.set_ylabel(f"Mean wavelet power ({units})")
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    return ax


def plot_global_power(res: CWTResult, *, ax=None, alpha: float | None = None, level: float = 0.95,
                      period_scale: float = 1.0, period_ticks=None, period_label: str = "",
                      units: str = "m²", bands: Mapping[str, tuple[float, float]] | None = None,
                      title: str | None = None, period_lim: tuple[float, float] | None = None):
    """Global wavelet spectrum on a vertical period axis, to sit beside :func:`plot_power`.

    Time-averaged power outside the COI and gaps in physical units (ink), the
    AR1 ``level`` significance line (dashed) and, per band (limits in the
    units of ``res.periods``), the band's share of the resolved variance
    (:func:`shoreshop3.wavelets.band_variance_fraction`, same trustworthy cells).
    """
    from matplotlib.transforms import blended_transform_factory

    from .wavelets import band_variance_fraction, global_significance

    ax = ax or plt.gca()
    gws, count = global_spectrum(res)
    per = res.periods * period_scale
    power = gws * res.variance
    ax.plot(power, per, color=INK, linewidth=1.5, label="time mean", zorder=3)
    if alpha is not None:
        thr = global_significance(res, alpha, level, n_used=count) * res.variance
        ax.plot(thr, per, color=INK_2, linewidth=1.0, linestyle=(0, (4, 2)),
                label=f"{level:.0%} red noise", zorder=2)
    ax.set_xscale("log")
    shown = np.isfinite(power) & (power > 0)
    if period_lim is not None:
        shown &= (per >= period_lim[0]) & (per <= period_lim[1])
    if shown.any():
        # room on the left for the legend and on the right for the band labels
        ax.set_xlim(power[shown].min() * 0.01, power[shown].max() * 30)
    _period_axis(ax, per.min(), per.max(), period_ticks, period_label, period_lim)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.grid(axis="x")
    ax.set_xlabel(f"Power ({units})")
    if bands:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        edges = sorted({v for b in bands.values() for v in b})
        for v in edges:
            if per.min() <= v * period_scale <= per.max():
                ax.axhline(v * period_scale, color=AXIS, linewidth=0.8, zorder=1)
        for lo, hi in bands.values():
            mid = np.sqrt(lo * hi) * period_scale
            if per.min() <= mid <= per.max():
                share = band_variance_fraction(res, (lo, hi))
                ax.text(0.97, mid, f"{share:.0%}" if np.isfinite(share) else "–", transform=trans,
                        ha="right", va="center", fontsize=9, color=INK_2)
    ax.legend(loc="lower left", fontsize=8, handlelength=1.8)
    if title:
        ax.set_title(title)
    return ax


_METRICS = {
    "nse": dict(cmap=DIVERGING.reversed(), norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
                label="NSE (1 = perfect)"),
    "corr": dict(cmap=DIVERGING.reversed(), norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
                 label="Correlation"),
    "std_ratio": dict(cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
                      label="Model / observed amplitude", transform=np.log2,
                      ticks=([-2, -1, 0, 1, 2], ["×0.25", "×0.5", "×1", "×2", "×4"])),
    "mean_rsq": dict(cmap=SEQUENTIAL, norm=Normalize(0.0, 1.0), label="Coherence R²"),
    "sig_frac": dict(cmap=SEQUENTIAL, norm=Normalize(0.0, 1.0), label="Coherent share"),
    "var_frac_obs": dict(cmap=SEQUENTIAL, norm=Normalize(0.0, 1.0), label="Share of obs. variance"),
    "valid_frac": dict(cmap=SEQUENTIAL, norm=Normalize(0.0, 1.0), label="Share evaluable"),
    "phase_deg": dict(cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-90.0, vcenter=0.0, vmax=90.0),
                      label="Phase (deg, + = model late)"),
}


def _row_markers(ax, rows: Sequence[str], row_colors: Mapping[str, str]) -> None:
    """Small colour squares left of the row labels (identity sits beside the text)."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, r in enumerate(rows):
        if r in row_colors:
            ax.scatter([-0.012], [i + 0.5], marker="s", s=28, color=row_colors[r], transform=trans,
                       clip_on=False, zorder=5, linewidths=0)
    ax.tick_params(axis="y", pad=10)


def legend_patches(colors: Mapping[str, str]):
    """Legend handles (squares) for a {label: colour} mapping."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker="s", linestyle="", markersize=7, color=c, label=k)
            for k, c in colors.items()]


def _text_color(rgb) -> str:
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.30 else INK


def plot_skill_heatmap(df: pd.DataFrame, metric: str = "nse", *, ax=None,
                       models: Sequence[str] | None = None, bands: Sequence[str] | None = None,
                       annotate: bool = True, fmt="{:.2f}", text_metric: str | None = None,
                       title: str | None = None, colorbar: bool = True,
                       row_colors: Mapping[str, str] | None = None, norm=None, cmap=None,
                       label: str | None = None, band_labels: Sequence[str] | None = None):
    """Models x period bands, coloured by a :func:`band_skill` metric.

    ``df`` is the output of :func:`shoreshop3.wavelets.compare_models`. For
    ``lag`` a symmetric diverging scale is built from the data; ``phase_deg``
    is comparable across bands. ``text_metric`` writes another column in the
    cells (e.g. colour by ``phase_deg``, print ``lag`` in days). ``fmt`` is a
    format string or a callable; ``band_labels`` replaces the column labels.
    """
    ax = ax or plt.gca()
    models = list(models or dict.fromkeys(df["model"]))
    bands = list(bands or dict.fromkeys(df["band"]))
    table = df.pivot(index="model", columns="band", values=metric).reindex(index=models, columns=bands)
    values = table.to_numpy(dtype=float)
    labels = values if text_metric is None else df.pivot(
        index="model", columns="band", values=text_metric).reindex(index=models, columns=bands).to_numpy(float)
    style = dict(_METRICS.get(metric, {}))
    if metric == "lag" or not style:
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        style = dict(cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), label=metric)
    if norm is not None:
        style["norm"] = norm
    if cmap is not None:
        style["cmap"] = cmap
    if label is not None:
        style["label"] = label
    with np.errstate(divide="ignore", invalid="ignore"):
        shown = style["transform"](values) if "transform" in style else values
    cmap = style["cmap"].with_extremes(bad=GRID)
    mesh = ax.pcolormesh(np.ma.masked_invalid(shown), cmap=cmap, norm=style["norm"],
                         edgecolors=SURFACE, linewidth=2)
    if band_labels is not None and len(band_labels) != len(bands):
        raise ValueError("band_labels needs one label per band")
    ax.set_xticks(np.arange(len(bands)) + 0.5)
    ax.set_xticklabels(list(band_labels or bands), rotation=0, fontsize=9)
    ax.set_yticks(np.arange(len(models)) + 0.5)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_ylim(len(models), 0)
    ax.tick_params(length=0)
    if row_colors:
        _row_markers(ax, models, row_colors)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(len(models)):
            for j in range(len(bands)):
                v = labels[i, j]
                if np.isfinite(v) and np.isfinite(values[i, j]):
                    rgb = cmap(style["norm"](shown[i, j]))
                    ax.text(j + 0.5, i + 0.5, _fmt_value(v, fmt), ha="center", va="center",
                            fontsize=9, color=_text_color(rgb))
                else:
                    ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", fontsize=9, color=INK_2)
    if colorbar:
        cb = _colorbar(ax, mesh, style["label"])
        if "ticks" in style and norm is None:
            cb.set_ticks(style["ticks"][0])
            cb.set_ticklabels(style["ticks"][1])
    if title:
        ax.set_title(title)
    return mesh


def plot_spectral_ratio(periods: np.ndarray, ratios: pd.DataFrame, *, ax=None,
                        period_scale: float = 1.0, period_ticks=None,
                        period_label: str = "Period", row_colors: Mapping[str, str] | None = None,
                        title: str | None = None, colorbar: bool = True, lim_log2: float = 2.0,
                        unresolved_below: float | None = None,
                        unresolved_label: str = "not resolved\nby the data",
                        period_lim: tuple[float, float] | None = None):
    """Models x period heatmap of model / observed time-averaged wavelet power.

    ``ratios``: DataFrame indexed by model, one column per period (same order
    as ``periods``). Blue = too little variance at that period, red = too much.
    Periods below ``unresolved_below`` (same units as ``periods * period_scale``)
    are washed out and labelled, e.g. periods shorter than the survey spacing.
    ``period_lim`` = (min, max) crops the period axis.
    """
    ax = ax or plt.gca()
    per = np.asarray(periods, float) * period_scale
    edges = np.sqrt(per[:-1] * per[1:])
    edges = np.r_[per[0] ** 2 / edges[0], edges, per[-1] ** 2 / edges[-1]]
    r = ratios.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log2(np.where(r == 0, 2.0 ** (-lim_log2 - 1), r))  # a constant model: far too little
    cmap = DIVERGING.with_extremes(bad=GRID)
    norm = TwoSlopeNorm(vmin=-lim_log2, vcenter=0.0, vmax=lim_log2)
    rows = list(ratios.index)
    mesh = ax.pcolormesh(edges, np.arange(len(rows) + 1), np.ma.masked_invalid(z), cmap=cmap,
                         norm=norm, shading="flat")
    ax.set_xscale("log", base=2)
    x0, x1 = edges[0], edges[-1]
    if period_lim is not None:
        x0, x1 = max(x0, period_lim[0]), min(x1, period_lim[1])
    ax.set_xlim(x0, x1)
    values, labels = _tick_spec(period_ticks, x0, x1)
    ax.set_xticks(values)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel(period_label)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_ylim(len(rows), 0)
    ax.tick_params(axis="y", length=0)
    for sp_ in ("left", "right", "top"):
        ax.spines[sp_].set_visible(False)
    if row_colors:
        _row_markers(ax, rows, row_colors)
    if unresolved_below is not None and unresolved_below > x0:
        hi = min(unresolved_below, x1)
        ax.axvspan(x0, hi, color=SURFACE, alpha=0.82, linewidth=0, zorder=3)
        ax.axvline(hi, color=INK_2, linewidth=0.8, zorder=4)
        ax.text(np.sqrt(x0 * hi), len(rows) / 2, unresolved_label, rotation=90, ha="center",
                va="center", fontsize=9, color=INK_2, zorder=5)
    if colorbar:
        cb = _colorbar(ax, mesh, "Model / observed variance")
        ticks = np.arange(-lim_log2, lim_log2 + 0.5, 1.0)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"×{2.0 ** t:g}" for t in ticks])
    if title:
        ax.set_title(title)
    return mesh
