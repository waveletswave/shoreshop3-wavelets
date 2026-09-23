#!/usr/bin/env python3
"""Wavelet comparison of the ShoreShop3 Duck single-profile hindcasts with FRF surveys.

Brad's first request: 1980-2023 results of single-profile models at the Duck
profiles yFRF = 1 and 1006. For every ``mipDuck_1980-2023*.csv`` under data/raw:

1. FRF surveys go on a weekly grid over the window where every model and the
   surveys exist (default 1988-03 to 2019-11); survey gaps longer than 60 days
   are masked. Each model is read on the survey days and interpolated the same
   way (``--sampling surveys``), so both carry the same sampling filter.
2. Observed wavelet power, model/observed variance by period, wavelet
   coherence with red-noise significance, and skill per period band.
3. Tables and figures go to outputs/duck_1980-2023/ (see README.md there).

Usage:  python scripts/duck_wavelets.py            (about 5-10 minutes)
        python scripts/duck_wavelets.py --n-surrogates 100 --no-maps   (quick look)
        python scripts/duck_wavelets.py --profiles 1006 --end 2017-06-10 \\
            --out outputs/duck_1980-2023_1006_pre2017      (before the June 2017 step)
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # usable before `pip install -e .`

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shoreshop3 import duck  # noqa: E402
from shoreshop3 import plotting as sp  # noqa: E402
from shoreshop3 import wavelets as wv  # noqa: E402
from shoreshop3.paths import INTERIM, OUTPUTS, PROCESSED, RAW  # noqa: E402

YEAR = 365.25
BANDS = {  # periods in days; the surveys (every ~2-6 weeks) do not resolve < ~45 days
    "Events\n1.5-4 mo": (45, 120),
    "Sub-annual\n4-8 mo": (120, 240),
    "Annual\n8-18 mo": (240, 540),
    "Interannual\n1.5-4 yr": (540, 1460),
    "Multi-year\n4-8 yr": (1460, 2920),
}
FAMILY_ORDER = ["Equilibrium", "Equilibrium + DA", "Hybrid", "Process-based", "One-line",
                "Deep learning", "Unclassified"]
PERIOD_TICKS = {30 / YEAR: "1 mo", 91.3 / YEAR: "3 mo", 0.5: "6 mo", 1.0: "1 yr", 2.0: "2 yr",
                4.0: "4 yr", 8.0: "8 yr"}
PROFILE_NAME = {"1": "Profile 1 (≈500 m south of the FRF pier)",
                "1006": "Profile 1006 (≈500 m north of the FRF pier)"}
JUMP_M = 20.0  # survey-to-survey change flagged in the data checks
BAND_LO = min(lo for lo, _ in BANDS.values())
BAND_HI = max(hi for _, hi in BANDS.values())
MAP_PERIODS = (0.9 * BAND_LO / YEAR, 1.5 * BAND_HI / YEAR)  # period range drawn in the maps (years)


def family_colors(families) -> dict[str, str]:
    colors, i = {}, 0
    for fam in FAMILY_ORDER + sorted(set(families) - set(FAMILY_ORDER)):
        if fam not in set(families):
            continue
        if fam == "Unclassified":
            colors[fam] = sp.MUTED
        else:
            colors[fam] = sp.CATEGORICAL[i % len(sp.CATEGORICAL)]
            i += 1
    return colors


def ordered_models(meta: pd.DataFrame) -> list[str]:
    """Models grouped by family (fixed order), alphabetical within a family."""
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    rows = zip(meta["family"], meta["model"])
    return [m for _, m in sorted(rows, key=lambda fm: (rank.get(fm[0], len(rank)), fm[1].lower()))]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def one_line(band: str) -> str:
    return band.replace("\n", " ")


def fmt_nse(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 9.995 else f"{v:.2f}"


def days(td: str) -> str:
    return f"{pd.Timedelta(td) / pd.Timedelta('1D'):.0f} days"


# --- data checks -------------------------------------------------------------------------------

def survey_checks(case: duck.DuckCase, max_gap_days: float) -> dict:
    """Sampling statistics, isolated spikes and persistent steps in the surveys of the window."""
    t0, t1 = pd.Timestamp(case.time[0]), pd.Timestamp(case.time[-1])
    s = case.obs_raw[(case.obs_raw.index >= t0) & (case.obs_raw.index <= t1)]
    x, t = s.to_numpy(), s.index
    dd = np.diff(t.to_numpy()).astype("timedelta64[s]").astype(float) / 86400.0
    spikes, steps = [], []
    for i in range(1, x.size):
        d_prev = x[i] - x[i - 1]
        if abs(d_prev) <= JUMP_M:
            continue
        if i + 1 < x.size:
            d_next = x[i + 1] - x[i]
            if abs(d_next) > JUMP_M and np.sign(d_next) != np.sign(d_prev) and abs(x[i + 1] - x[i - 1]) < JUMP_M / 2:
                spikes.append((t[i].date(), float(d_prev)))
                continue
        before, after = x[max(0, i - 5):i], x[i:i + 5]
        if after.size >= 3 and abs(np.median(after) - np.median(before)) > 0.75 * JUMP_M:
            steps.append((t[i - 1].date(), t[i].date(), float(np.median(after) - np.median(before))))
    return dict(n=int(x.size), median_spacing=float(np.median(dd)), n_long_gaps=int((dd > max_gap_days).sum()),
                longest_gap=float(dd.max()), spikes=spikes, steps=steps, first=t[0].date(), last=t[-1].date())


def duplicate_models(case: duck.DuckCase, order: list[str], r_min: float = 0.99) -> tuple[list, list]:
    """Pairs of models that are identical (or nearly: r >= r_min) on the analysis grid."""
    same, close = [], []
    for a, b in itertools.combinations(order, 2):
        xa = case.models[a] - case.models[a].mean()
        xb = case.models[b] - case.models[b].mean()
        if np.allclose(xa, xb, atol=1e-6):
            same.append((a, b))
        elif xa.std() > 0 and xb.std() > 0 and np.corrcoef(xa, xb)[0, 1] >= r_min:
            close.append((a, b, float(np.corrcoef(xa, xb)[0, 1])))
    return same, close


def trends(case: duck.DuckCase, order: list[str], meta: pd.DataFrame) -> pd.DataFrame:
    """Least-squares linear trend (m/yr) over the window, outside survey gaps.

    Periods longer than the multi-year band cannot be resolved by wavelets in a
    ~30-year record, so the trend is reported separately. Also the spread left
    after removing it (detrended std), which flags models that are almost a
    straight line.
    """
    t_yr = (pd.DatetimeIndex(case.time) - pd.Timestamp(case.time[0])) / pd.Timedelta(days=YEAR)
    t_yr = np.asarray(t_yr, float)
    ok = ~case.gap

    def fit(x):
        b = np.polyfit(t_yr[ok], x[ok], 1)
        return float(b[0]), float(np.std(x[ok] - np.polyval(b, t_yr[ok])))

    obs_trend, obs_std = fit(case.obs)
    fam = dict(zip(meta["model"], meta["family"]))
    rows = []
    for m in order:
        tr, sd = fit(case.models[m])
        rows.append(dict(profile=case.profile, model=m, family=fam.get(m, ""), trend_m_per_yr=tr,
                         obs_trend_m_per_yr=obs_trend, trend_error=tr - obs_trend, detrended_std=sd,
                         obs_detrended_std=obs_std))
    return pd.DataFrame(rows)


# --- figures ---------------------------------------------------------------------------------

def fig_observations(cases: dict, args, out: Path) -> Path:
    n = len(cases)
    fig = plt.figure(figsize=(13.5, 5.6 * n))
    outer = fig.add_gridspec(n, 1, hspace=0.3, left=0.07, right=0.985, top=0.95, bottom=0.05)
    for k, (p, c) in enumerate(cases.items()):
        case, wo, chk = c["case"], c["wo"], c["checks"]
        gs = outer[k].subgridspec(2, 2, height_ratios=[1, 2], width_ratios=[6, 1.35], hspace=0.14,
                                  wspace=0.035)
        ax_ts, ax_map = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        ax_info, ax_gws = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        t = pd.DatetimeIndex(case.time)
        edges = np.flatnonzero(np.diff(np.r_[0, case.gap.astype(int), 0]))
        for a, z in zip(edges[::2], edges[1::2]):
            ax_ts.axvspan(t[a], t[min(z, len(t) - 1)], color=sp.GRID, alpha=0.9, linewidth=0)
        ax_ts.plot(t, case.obs, color=sp.INK, linewidth=0.8, alpha=0.45)
        raw = case.obs_raw[(case.obs_raw.index >= t[0]) & (case.obs_raw.index <= t[-1])]
        ax_ts.plot(raw.index, raw.to_numpy(), "o", markersize=2.2, color=sp.INK)
        ax_ts.set_ylabel("Shoreline xFRF (m)")
        ax_ts.grid(axis="y")
        ax_ts.set_xlim(t[0], t[-1])
        ax_ts.tick_params(labelbottom=False)
        ax_ts.set_title(f"{PROFILE_NAME.get(p, p)}: FRF surveys (dots), grey = gaps > {days(args.max_gap)}")
        cf = sp.plot_power(wo, time=case.time, ax=ax_map, alpha=wv.ar1(case.obs), period_scale=1 / YEAR,
                           period_ticks=PERIOD_TICKS, period_label="Period", colorbar=False,
                           period_lim=MAP_PERIODS,
                           title="Observed wavelet power (black contour: 95 % vs red noise; "
                                 "pale: cone of influence and survey gaps)")
        ax_map.set_xlim(mpl_dates(t[0]), mpl_dates(t[-1]))
        sp.plot_global_power(wo, ax=ax_gws, alpha=wv.ar1(case.obs), period_scale=1 / YEAR,
                             period_ticks=PERIOD_TICKS, bands=BANDS, period_lim=MAP_PERIODS)
        ax_gws.set_title("Time mean · % of variance", fontsize=9.5)
        ax_gws.tick_params(labelleft=False)
        ax_info.axis("off")
        ax_info.text(0.02, 1.0, f"{chk['n']} surveys\nmedian spacing {chk['median_spacing']:.0f} days\n"
                     f"{chk['n_long_gaps']} gaps > {days(args.max_gap)}", transform=ax_info.transAxes,
                     va="top", ha="left", fontsize=9, color=sp.INK_2, linespacing=1.5)
        cax = ax_info.inset_axes((0.04, 0.2, 0.9, 0.09))
        ticks = [v for v in range(-4, 13, 4) if cf.levels.min() <= v <= cf.levels.max()]
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
        cb.set_label("log2(power / variance)", color=sp.INK_2, fontsize=8, labelpad=2)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=7, color=sp.AXIS, labelcolor=sp.INK_2, length=2)
        cb.outline.set_visible(False)
    path = out / "fig1_observed_wavelet_power.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mpl_dates(ts) -> float:
    import matplotlib.dates as mdates

    return mdates.date2num(pd.Timestamp(ts).to_pydatetime())


def _panels(n_panels: int, n_rows: int, width_per: float = 6.6):
    fig, axes = plt.subplots(1, n_panels, figsize=(max(width_per * n_panels, 8.5), 1.9 + 0.27 * n_rows),
                             squeeze=False)
    return fig, list(axes[0])


def _finish_panels(fig, colors: dict, title: str) -> None:
    """Suptitle (wrapped to the figure width) and family legend below the panels."""
    width = fig.get_size_inches()[0]
    fig.suptitle(textwrap.fill(title, width=int(width / 0.1)), x=0.01, ha="left", fontsize=12, color=sp.INK)
    ncol = len(colors) if width > 11 else 4
    nrow = int(np.ceil(len(colors) / ncol))
    fig.legend(handles=sp.legend_patches(colors), loc="lower center", ncol=ncol, frameon=False)
    height = fig.get_size_inches()[1]
    lines = title.count("\n") + int(np.ceil(len(title) / int(width / 0.1)))
    fig.tight_layout(rect=(0, (0.1 + 0.22 * nrow) / height, 1, 1 - (0.05 + 0.2 * lines) / height))


def fig_spectral_ratio(cases: dict, colors: dict, out: Path) -> Path:
    n_rows = max(len(c["order"]) for c in cases.values())
    fig, axes = _panels(len(cases), n_rows, 6.75)
    lo_band = min(lo for lo, _ in BANDS.values())
    for ax, (p, c) in zip(axes, cases.items()):
        per, ratios = c["ratio"]
        row_colors = {m: colors[f] for m, f in zip(c["meta"]["model"], c["meta"]["family"])}
        sp.plot_spectral_ratio(per, ratios.loc[c["order"]], ax=ax, period_scale=1 / YEAR,
                               period_ticks=PERIOD_TICKS, row_colors=row_colors,
                               colorbar=ax is axes[-1], title=f"{PROFILE_NAME.get(p, p)}",
                               unresolved_below=lo_band / YEAR, period_lim=(0, BAND_HI / YEAR * 1.001),
                               unresolved_label="below the\nsurvey spacing")
        for lo, hi in BANDS.values():
            ax.axvline(hi / YEAR, color=sp.SURFACE, linewidth=1.2)
    _finish_panels(fig, colors, "Variance at each period: model / observed (same times, outside the cone of "
                                "influence and survey gaps); white lines = band limits")
    path = out / "fig2_variance_ratio_by_period.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def band_labels(skill_p: pd.DataFrame) -> list[str]:
    usable = skill_p.groupby("band", sort=False)["valid_frac"].first()
    return [f"{b}\n{usable[b]:.0%} usable" if b in usable and np.isfinite(usable[b]) else b for b in BANDS]


def fig_band_heatmap(skill: pd.DataFrame, cases: dict, colors: dict, metric: str, out: Path,
                     name: str, title: str, fmt="{:.2f}", text_metric: str | None = None) -> Path:
    n_rows = max(len(c["order"]) for c in cases.values())
    fig, axes = _panels(len(cases), n_rows, 6.5)
    for ax, (p, c) in zip(axes, cases.items()):
        df = skill[skill["profile"] == p]
        row_colors = {m: colors[f] for m, f in zip(c["meta"]["model"], c["meta"]["family"])}
        sp.plot_skill_heatmap(df, metric, ax=ax, models=c["order"], bands=list(BANDS), fmt=fmt,
                              text_metric=text_metric, row_colors=row_colors,
                              band_labels=band_labels(df), colorbar=ax is axes[-1],
                              title=PROFILE_NAME.get(p, p))
    _finish_panels(fig, colors, title)
    path = out / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_coherence_atlas(p: str, c: dict, colors: dict, out: Path, ncols: int = 5) -> Path:
    order = c["order"]
    nrows = int(np.ceil(len(order) / ncols))
    height = 2.1 * nrows + 1.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, height), sharex=True,
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.8 / height, bottom=0.75 / height,
                        hspace=0.34, wspace=0.06)
    fam = dict(zip(c["meta"]["model"], c["meta"]["family"]))
    mappable = None
    for ax, m in zip(axes.flat, order):
        coh, sig = c["coh"][m]
        mappable = sp.plot_coherence(coh, time=c["case"].time, ax=ax, sig=sig, arrows=False,
                                     period_scale=1 / YEAR, period_ticks=PERIOD_TICKS,
                                     period_label="", colorbar=False, period_lim=MAP_PERIODS)
        ax.set_title(m, fontsize=9, x=0.06)
        ax.scatter([0.015], [1.075], marker="s", s=22, color=colors[fam[m]], transform=ax.transAxes,
                   clip_on=False, linewidths=0)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    fig.suptitle(f"{PROFILE_NAME.get(p, p)}: wavelet coherence with the FRF surveys "
                 "(black contour: 95 % significance; pale: cone of influence and survey gaps)",
                 x=0.01, ha="left", fontsize=12, color=sp.INK)
    cax = fig.add_axes((0.36, 0.3 / height, 0.3, 0.1 / height))
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cax.text(-0.02, 0.5, "Coherence R²", transform=cax.transAxes, ha="right", va="center",
             fontsize=10, color=sp.INK_2)
    cb.outline.set_visible(False)
    fig.legend(handles=sp.legend_patches(colors), loc="lower right", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.99, 0.0))
    path = out / f"fig7_coherence_atlas_profile{p}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_power_atlas(p: str, c: dict, colors: dict, out: Path, ncols: int = 5) -> Path:
    """Each model's own wavelet power (m²) next to the surveys', on one colour scale."""
    panels = ["obs"] + c["order"]
    nrows = int(np.ceil(len(panels) / ncols))
    height = 2.1 * nrows + 1.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, height), sharex=True,
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.8 / height, bottom=0.75 / height,
                        hspace=0.34, wspace=0.06)
    fam = dict(zip(c["meta"]["model"], c["meta"]["family"]))
    levels = sp.power_levels([c["wo"]], period_range=(MAP_PERIODS[0] * YEAR, MAP_PERIODS[1] * YEAR))
    mappable = None
    for ax, key in zip(axes.flat, panels):
        res = c["wo"] if key == "obs" else c["wms"][key]
        mappable = sp.plot_power(res, time=c["case"].time, ax=ax, physical=True, levels=levels,
                                 period_scale=1 / YEAR, period_ticks=PERIOD_TICKS, period_label="",
                                 colorbar=False, period_lim=MAP_PERIODS)
        if key == "obs":
            ax.set_title("FRF surveys (observed)", fontsize=9, x=0.0, fontweight="bold")
        else:
            ax.set_title(key, fontsize=9, x=0.06)
            ax.scatter([0.015], [1.075], marker="s", s=22, color=colors[fam[key]], transform=ax.transAxes,
                       clip_on=False, linewidths=0)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{PROFILE_NAME.get(p, p)}: wavelet power of the surveys (first panel) and of each model, "
                 "same colour scale (pale: cone of influence and survey gaps)",
                 x=0.01, ha="left", fontsize=12, color=sp.INK)
    cax = fig.add_axes((0.36, 0.3 / height, 0.3, 0.1 / height))
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    sp.log2_ticks(cb, levels)
    cb.ax.tick_params(labelsize=8)
    cax.text(-0.02, 0.5, "Wavelet power (m²)", transform=cax.transAxes, ha="right", va="center",
             fontsize=10, color=sp.INK_2)
    cb.outline.set_visible(False)
    fig.legend(handles=sp.legend_patches(colors), loc="lower right", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.99, 0.0))
    path = out / f"fig8_power_atlas_profile{p}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_single_coherence(p: str, m: str, c: dict, out: Path) -> Path:
    coh, sig = c["coh"][m]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sp.plot_coherence(coh, time=c["case"].time, ax=ax, sig=sig, period_scale=1 / YEAR,
                      period_ticks=PERIOD_TICKS, period_label="Period", period_lim=MAP_PERIODS,
                      title=f"{PROFILE_NAME.get(p, p)}: FRF surveys vs {m} "
                            "(arrows: right = in phase, down = model lags)")
    fig.tight_layout()
    path = out / "wtc" / f"profile{p}_{slug(m)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- summary -------------------------------------------------------------------------------------

def family_table(skill: pd.DataFrame) -> pd.DataFrame:
    """Median NSE and share of significant coherence per family, band and profile."""
    g = skill.groupby(["profile", "family", "band"], sort=False)
    out = g.agg(n_models=("model", "nunique"), median_nse=("nse", "median"),
                median_sig_frac=("sig_frac", "median"), median_std_ratio=("std_ratio", "median"))
    return out.reset_index()


def _md_family(fam: pd.DataFrame, p: str, col: str, fmt) -> list[str]:
    df = fam[fam["profile"] == p]
    fams = [f for f in FAMILY_ORDER if f in set(df["family"])] + sorted(set(df["family"]) - set(FAMILY_ORDER))
    head = "| Family (n) | " + " | ".join(one_line(b) for b in BANDS) + " |"
    rows = [head, "|---" * (len(BANDS) + 1) + "|"]
    for f in fams:
        d = df[df["family"] == f].set_index("band")
        n = int(d["n_models"].iloc[0])
        cells = [fmt(d.loc[b, col]) if b in d.index and np.isfinite(d.loc[b, col]) else "–" for b in BANDS]
        rows.append(f"| {f} ({n}) | " + " | ".join(cells) + " |")
    return rows


def write_summary(skill: pd.DataFrame, fam: pd.DataFrame, cases: dict, args, out: Path,
                  figures: list[Path], teams_without: list[str]) -> Path:
    c0 = next(iter(cases.values()))
    w0, w1 = pd.Timestamp(c0["case"].time[0]).date(), pd.Timestamp(c0["case"].time[-1]).date()
    n_models = len(c0["order"])
    lines = ["# Duck single-profile hindcasts (1980-2023 submissions) vs FRF surveys: wavelet comparison", ""]
    if args.note:
        lines += [f"> {args.note}", ""]
    lines += [f"* Window {w0} to {w1}, grid step {days(args.step)}; survey gaps > {days(args.max_gap)} masked. "
             "The window is the stretch every model and the surveys cover.",
             f"* {n_models} models from {c0['meta']['team'].nunique()} teams. Model families are provisional "
             "(edit `config/duck_models.csv`).",
             "* Models were read on the survey days and interpolated like the surveys, so both carry the same "
             "sampling filter." if args.sampling == "surveys" else
             f"* Models were averaged over {args.step} (centred) and compared with the interpolated surveys.",
             f"* Coherence significance: {args.n_surrogates} pairs of red-noise (AR1) surrogates; "
             "by chance about 5 % of a band is 'significant'.",
             "* Offsets between submissions do not matter: every series is compared as an anomaly.",
             "* 'Usable' = share of the record where the whole band lies outside the cone of influence and "
             "the survey gaps. NSE, amplitude and correlation use only those times.", ""]

    lines += ["## Figures", "",
              "1. `fig1_observed_wavelet_power.png`: the surveys, their wavelet power, and how the observed "
              "variance splits across the bands.",
              "2. `fig2_variance_ratio_by_period.png`: model / observed variance at each period "
              "(red = too much, blue = too little).",
              "3. `fig3_coherence_by_band.png`: share of each band where the model co-varies significantly "
              "with the surveys (timing, amplitude-blind).",
              "4. `fig4_amplitude_by_band.png`: model / observed standard deviation of the band signal.",
              "5. `fig5_nse_by_band.png`: Nash-Sutcliffe efficiency of the band signal (1 = perfect, "
              "0 = no better than the mean).",
              "6. `fig6_timing_by_band.png`: phase where coherent; text = lag in days (+ = model late).",
              "7. `fig7_coherence_atlas_profile*.png`: coherence maps, one panel per model"
              + ("." if args.no_maps else "; `wtc/` has one larger map per model with phase arrows."),
              "8. `fig8_power_atlas_profile*.png`: each model's own wavelet power in m², on the same colour "
              "scale as the surveys (first panel): where in time and period each model carries variance.", ""]

    lines += ["## Best models per band", ""]
    for p, c in cases.items():
        df = skill[skill["profile"] == p]
        lines += [f"### {PROFILE_NAME.get(p, p)}", "",
                  "| Band | obs. variance | usable | best NSE | most coherent (share significant) |",
                  "|---|---|---|---|---|"]
        for band in BANDS:
            b = df[df["band"] == band]
            top_nse = b.dropna(subset=["nse"]).nlargest(3, "nse")
            top_coh = b.dropna(subset=["sig_frac"]).nlargest(3, "sig_frac")
            nse = ", ".join(f"{r.model} ({fmt_nse(r.nse)})" for r in top_nse.itertuples()) or "–"
            coh = ", ".join(f"{r.model} ({r.sig_frac:.0%})" for r in top_coh.itertuples()) or "–"
            lines.append(f"| {one_line(band)} | {b['var_frac_obs'].iloc[0]:.0%} | {b['valid_frac'].iloc[0]:.0%} "
                         f"| {nse} | {coh} |")
        lines += ["", "Median NSE per family:", ""] + _md_family(fam, p, "median_nse", fmt_nse)
        lines += ["", "Median share of the band with significant coherence per family:", ""] + \
            _md_family(fam, p, "median_sig_frac", lambda v: f"{v:.0%}")
        lines.append("")

    lines += ["## Long-term trend (longer than the wavelet bands)", "",
              "Least-squares trend over the window, outside survey gaps (m/yr, + = seaward).", "",
              "| Profile | observed | models: median (range) | closest to observed |", "|---|---|---|---|"]
    for p, c in cases.items():
        tr = c["trend"]
        close = tr.assign(err=tr["trend_error"].abs()).nsmallest(3, "err")
        lines.append(f"| {p} | {tr['obs_trend_m_per_yr'].iloc[0]:+.2f} | {tr['trend_m_per_yr'].median():+.2f} "
                     f"({tr['trend_m_per_yr'].min():+.2f} to {tr['trend_m_per_yr'].max():+.2f}) | "
                     + ", ".join(f"{r.model} ({r.trend_m_per_yr:+.2f})" for r in close.itertuples()) + " |")
    lines.append("")

    lines += ["## Data checks", ""]
    for p, c in cases.items():
        chk, case = c["checks"], c["case"]
        lines.append(f"* Profile {p}: {chk['n']} surveys in the window ({chk['first']} to {chk['last']}), "
                     f"median spacing {chk['median_spacing']:.0f} days, {chk['n_long_gaps']} gaps longer than "
                     f"{days(args.max_gap)} (longest {chk['longest_gap']:.0f} days).")
        for d, v in chk["spikes"]:
            lines.append(f"  * Isolated survey {v:+.0f} m off its neighbours on {d} (possible outlier; kept).")
        for d0, d1, v in chk["steps"]:
            lines.append(f"  * Lasting step of {v:+.0f} m between the surveys of {d0} and {d1} "
                         f"(e.g. a nourishment). To leave it out: `--profiles {p} --end {d0}`.")
        tr = c["trend"]
        for r in tr[tr["detrended_std"] < 0.15 * tr["obs_detrended_std"]].itertuples():
            lines.append(f"  * {r.model} is almost a straight line here (spread around its trend "
                         f"{r.detrended_std:.1f} m vs {r.obs_detrended_std:.1f} m observed).")
        sk = skill[skill["profile"] == p]
        df = sk.drop_duplicates("band").set_index("band")
        worst = sk.groupby("band")["nse"].min()
        weak = [one_line(b) for b in BANDS if b in df.index and
                (df.loc[b, "valid_frac"] < 0.3 or (df.loc[b, "var_frac_obs"] < 0.05 and worst.get(b, 0) < -3))]
        if weak:
            lines.append(f"  * {', '.join(weak)}: few usable years and little observed variance, so NSE swings to "
                         "large negative values there; lean on coherence (fig3).")
        if case.skipped:
            lines.append("  * Skipped: " + "; ".join(f"{k} ({v})" for k, v in case.skipped.items()))
        same, close = c["dups"]
        for a, b in same:
            lines.append(f"  * {a} and {b} are identical in this window (counted twice in the figures).")
        for a, b, r in close:
            lines.append(f"  * {a} and {b} are nearly identical here (r = {r:.3f}).")
    if teams_without:
        lines.append(f"* Team folders without a `mipDuck_1980-2023*.csv` file: {', '.join(teams_without)}.")
    last_survey = max(c["case"].obs_raw.index.max() for c in cases.values()).date()
    lines += ["", "## Caveats", "",
              f"* The FRF surveys provided to the teams end on {last_survey}, so every score here is in-sample: "
              "the teams could calibrate on these surveys. The scores show how well each model reproduces "
              "the record, not blind skill. Models in the 'Equilibrium + DA' family assimilate the surveys.",
              "* Bands with a low usable share (long periods) rest on a few years of data; treat their NSE "
              "with care.",
              "* A lasting step in the surveys (e.g. a nourishment) shows up as interannual and multi-year "
              "variance; see the data checks above.",
              "* Families are provisional labels for grouping only.", ""]
    lines += ["## Files", ""] + [f"* `{f.relative_to(out)}`" for f in figures] + \
             ["* `skill_by_band.csv`: every metric for every model, band and profile "
              "(sig_frac = share significant, mean_rsq = mean coherence, lag in days)",
              "* `family_summary.csv`: medians per family",
              "* `trend_by_model.csv`: linear trend and detrended spread of every model",
              "* `variance_ratio_by_period_profile*.csv`: model / observed variance per period (years)",
              "* `run_info.json`: settings of this run", ""]
    path = out / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "duck_1980-2023")
    ap.add_argument("--start", default="1988-03-01")
    ap.add_argument("--end", default="2019-11-30")
    ap.add_argument("--step", default="7D")
    ap.add_argument("--max-gap", default="60D")
    ap.add_argument("--sampling", choices=duck.SAMPLING, default="surveys",
                    help="surveys: read models on the survey days (default); mean: weekly means")
    ap.add_argument("--n-surrogates", type=int, default=300)
    ap.add_argument("--profiles", nargs="+", default=list(duck.PROFILES))
    ap.add_argument("--no-maps", action="store_true", help="skip one-map-per-model figures")
    ap.add_argument("--note", default="", help="one line printed at the top of the README")
    args = ap.parse_args()

    t_start = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_maps:
        (out / "wtc").mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sp.use_style()
    subs = duck.find_submissions(args.raw)
    if subs.empty:
        sys.exit(f"No mipDuck_1980-2023*.csv under {args.raw}: download the Duck files first")
    print(f"{len(subs)} submissions from {subs['team'].nunique()} teams")
    colors = family_colors(subs["family"])
    teams_without = sorted(set(duck.team_folders(args.raw)) - set(subs["folder"]), key=str.lower)
    max_gap_days = pd.Timedelta(args.max_gap) / pd.Timedelta("1D")

    cases, skills = {}, []
    for p in args.profiles:
        case = duck.build_case(args.raw, p, args.start, args.end, step=args.step, max_gap=args.max_gap,
                               sampling=args.sampling, submissions=subs)
        order = ordered_models(case.meta)
        aligned = pd.DataFrame({"obs": case.obs, "obs_gap": case.gap, **case.models},
                               index=pd.DatetimeIndex(case.time, name="time"))
        aligned.to_csv(PROCESSED / f"duck_profile{p}_{args.start}_{args.end}_{args.step}_{args.sampling}.csv")
        print(f"profile {p}: {len(order)} models, {case.obs.size} grid points, "
              f"{int(case.gap.sum())} in survey gaps; skipped: {case.skipped or 'none'}")

        skill = wv.compare_models(case.obs, {m: case.models[m] for m in order}, case.dt_days, BANDS,
                                  invalid_obs=case.gap, n_surrogates=args.n_surrogates,
                                  cache_dir=INTERIM / "wtc_sig", alpha_decimals=2)
        info = case.meta.set_index("model")[["team", "family", "file"]]
        skill = skill.join(info, on="model")
        skill.insert(0, "profile", p)
        skills.append(skill)

        wo = wv.cwt(case.obs, case.dt_days, invalid=case.gap)
        a_obs = wv.ar1(case.obs)
        ratio, coh, wms = {}, {}, {}
        inv_mod = case.gap if case.sampling == "surveys" else None
        for m in order:
            wm = wms[m] = wv.cwt(case.models[m], case.dt_days, invalid=inv_mod)
            per, p_obs, p_mod = wv.paired_global_spectra(wo, wm)
            ratio[m] = p_mod / p_obs
            sig = wv.coherence_significance(case.obs.size, case.dt_days, a_obs, wv.ar1(case.models[m]),
                                            n_surrogates=args.n_surrogates,
                                            cache_dir=INTERIM / "wtc_sig", alpha_decimals=2)
            coh[m] = (wv.wavelet_coherence(case.obs, case.models[m], case.dt_days, invalid_x=case.gap), sig)
        ratio_df = pd.DataFrame(ratio).T
        ratio_df.columns = np.round(per / YEAR, 4)
        ratio_df.rename_axis(index="model", columns="period_years").to_csv(
            out / f"variance_ratio_by_period_profile{p}.csv")
        cases[p] = dict(case=case, wo=wo, wms=wms, order=order, meta=case.meta,
                        ratio=(per, pd.DataFrame(ratio).T), coh=coh, checks=survey_checks(case, max_gap_days), dups=duplicate_models(case, order),
                        trend=trends(case, order, case.meta))
        print(f"  analysed in {time.time() - t_start:.0f} s")

    skill_all = pd.concat(skills, ignore_index=True)
    fam = family_table(skill_all)
    skill_all.assign(band=skill_all["band"].map(one_line)).to_csv(out / "skill_by_band.csv", index=False)
    fam.assign(band=fam["band"].map(one_line)).to_csv(out / "family_summary.csv", index=False)
    pd.concat([c["trend"] for c in cases.values()], ignore_index=True).to_csv(
        out / "trend_by_model.csv", index=False, float_format="%.4g")

    figures = [
        fig_observations(cases, args, out),
        fig_spectral_ratio(cases, colors, out),
        fig_band_heatmap(skill_all, cases, colors, "sig_frac", out, "fig3_coherence_by_band.png",
                         "Co-variation: share of each band where model and surveys are significantly coherent "
                         "(95 %, red-noise test; 5 % = chance)", fmt="{:.0%}"),
        fig_band_heatmap(skill_all, cases, colors, "std_ratio", out, "fig4_amplitude_by_band.png",
                         "Amplitude: model / observed standard deviation of the band signal (1 = right size)"),
        fig_band_heatmap(skill_all, cases, colors, "nse", out, "fig5_nse_by_band.png",
                         "Overall skill: Nash-Sutcliffe efficiency of the band signal (1 = perfect, "
                         "0 = no better than the mean)", fmt=fmt_nse),
        fig_band_heatmap(skill_all, cases, colors, "phase_deg", out, "fig6_timing_by_band.png",
                         "Timing error where coherent: colour = phase, text = lag in days (+ = model late)",
                         fmt="{:.0f} d", text_metric="lag"),
    ]
    for p, c in cases.items():
        figures.append(fig_coherence_atlas(p, c, colors, out))
    for p, c in cases.items():
        figures.append(fig_power_atlas(p, c, colors, out))
    for p, c in cases.items():
        if not args.no_maps:
            for m in c["order"]:
                fig_single_coherence(p, m, c, out)
    summary = write_summary(skill_all, fam, cases, args, out, figures, teams_without)
    def _rel(v):
        try:
            return str(Path(v).resolve().relative_to(ROOT))
        except ValueError:
            return Path(v).name
    info = {k: (_rel(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    info.update(bands_days={one_line(k): v for k, v in BANDS.items()}, n_models=len(cases[args.profiles[0]]["order"]),
                runtime_s=round(time.time() - t_start), created=time.strftime("%Y-%m-%d %H:%M"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t_start:.0f} s. See {summary}")


if __name__ == "__main__":
    main()
