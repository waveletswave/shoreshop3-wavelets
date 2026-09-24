#!/usr/bin/env python3
"""Wavelet comparison of the ShoreShop3 Duck single-profile hindcasts with FRF surveys.

First analysis: the 1980-2023 single-profile hindcasts at the Duck profiles
yFRF = 1 and 1006. For every ``mipDuck_1980-2023*.csv`` under data/raw:

1. FRF surveys go on a weekly grid over the whole survey record (from October
   1980; ``--window surveys``). Models that do not cover it are listed with the
   reason and can be compared over a shorter common window (``--window
   common``). Survey gaps longer than 60 days are masked. Each model is read on
   the survey days and interpolated the same way (``--sampling surveys``), so
   both series carry the same sampling filter.
2. Observed wavelet power, model/observed variance by period, wavelet
   coherence with red-noise significance, and skill per period band.
3. Tables and figures (PNG and PDF) go to outputs/duck_1980-2023/ (see
   README.md there).

Usage:  python scripts/duck_wavelets.py            (about 5 minutes the first time)
        python scripts/duck_wavelets.py --window common --out outputs/duck_1980-2023_common
            (every run that covers a shorter common window, including those that start later)
        python scripts/duck_wavelets.py --profiles 1006 --end 2017-06-13 --no-maps \\
            --out outputs/duck_1980-2023_1006_pre2017      (before the June 2017 step)
        python scripts/duck_wavelets.py --n-surrogates 100 --no-maps   (quick look)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # usable before `pip install -e .`

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shoreshop3 import duck  # noqa: E402
from shoreshop3 import plotting as sp  # noqa: E402
from shoreshop3 import wavelets as wv  # noqa: E402
from shoreshop3.paths import INTERIM, OUTPUTS, PROCESSED, RAW  # noqa: E402

YEAR = 365.25
BANDS = {  # periods in days; surveys every ~2-6 weeks do not resolve shorter periods
    "Monthly\n1.5-4 mo": (45, 120),  # month-to-month changes the surveys can see, not single storms
    "Sub-annual\n4-8 mo": (120, 240),
    "Annual\n8-18 mo": (240, 540),
    "Interannual\n1.5-4 yr": (540, 1460),
    "Multi-year\n4-8 yr": (1460, 2920),
}
MIN_CYCLES = 3.0  # bands with fewer usable cycles are exploratory: shown, not ranked
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
ATLAS_DPI = 220  # atlases are 16 x 16 in: 220 dpi is ~3600 px wide
EXPLORATORY_NOTE = f"* fewer than {MIN_CYCLES:.0f} usable cycles: exploratory, not ranked"
EXPLORATORY_MD = EXPLORATORY_NOTE.replace("*", "\\*", 1)  # the same note in Markdown


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


def cycles(n: float) -> str:
    return f"{n:.0f}" if n >= 10 else f"{n:.1f}"


def month(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m") if pd.notna(ts) else "–"


def mpl_dates(ts) -> float:
    return mdates.date2num(pd.Timestamp(ts).to_pydatetime())


def save(fig, path: Path, args, dpi: int | None = None, pdf: bool = True) -> Path:
    """PNG at ``dpi`` (default --dpi) and, unless --no-pdf, a PDF next to it."""
    dpi = dpi or args.dpi
    fig.savefig(path, dpi=dpi)
    if pdf and not args.no_pdf:
        with sp.compact_pdf_images():  # dense layers are rasterised; keep memory small
            fig.savefig(path.with_suffix(".pdf"), dpi=min(max(dpi, 200), 300))  # dpi of those layers
    plt.close(fig)
    return path


# --- data checks -------------------------------------------------------------------------------

def survey_checks(case: duck.DuckCase, max_gap_days: float) -> dict:
    """Sampling statistics, isolated spikes and persistent steps in the surveys used.

    The surveys used are those inside the grid plus the enclosing survey on
    each side (they set the first and last grid values).
    """
    s = case.obs_raw.loc[duck.needed_days(case.obs_raw.index, case.time[0], case.time[-1])]
    x, t = s.to_numpy(), s.index
    dd = np.diff(t.to_numpy()).astype("timedelta64[s]").astype(float) / 86400.0
    spacing = pd.Series(dd, index=t[1:])
    blocks = spacing.groupby((spacing.index.year // 5) * 5).median()  # median spacing per 5-year block
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
    edge_gaps = []  # a long gap right at the start or end of the record
    if dd.size and dd[0] > max_gap_days:
        edge_gaps.append(("start", t[0].date(), t[1].date(), float(dd[0])))
    if dd.size > 1 and dd[-1] > max_gap_days:
        edge_gaps.append(("end", t[-2].date(), t[-1].date(), float(dd[-1])))
    return dict(n=int(x.size), median_spacing=float(np.median(dd)), n_long_gaps=int((dd > max_gap_days).sum()),
                longest_gap=float(dd.max()), spikes=spikes, steps=steps, first=t[0].date(), last=t[-1].date(),
                block_min=float(blocks.min()), block_max=float(blocks.max()), sparsest=int(blocks.idxmax()),
                edge_gaps=edge_gaps)


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
    team = dict(zip(meta["model"], meta["team"]))
    rows = []
    for m in order:
        tr, sd = fit(case.models[m])
        rows.append(dict(profile=case.profile, model=m, team=team.get(m, ""), family=fam.get(m, ""),
                         trend_m_per_yr=tr, obs_trend_m_per_yr=obs_trend, trend_error=tr - obs_trend,
                         detrended_std=sd, obs_detrended_std=obs_std))
    return pd.DataFrame(rows)


def add_support_dates(skill: pd.DataFrame, case: duck.DuckCase) -> pd.DataFrame:
    """Dates and years behind each band's scores, and the exploratory flag."""
    t = pd.DatetimeIndex(case.time)

    def _date(i):
        return t[int(i)] if pd.notna(i) and int(i) >= 0 else pd.NaT

    out = skill.copy()
    out["valid_start"] = [_date(i) for i in out["valid_first"]]
    out["valid_end"] = [_date(i) for i in out["valid_last"]]
    out["valid_years"] = out["valid_duration"] / YEAR
    out["exploratory"] = ~(out["n_cycles"] >= MIN_CYCLES)
    return out


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
                           title="Observed wavelet power (black contour: nominal 95 % level, AR(1) red noise; "
                                 "pale: cone of influence and survey gaps)")
        ax_map.set_xlim(mpl_dates(t[0]), mpl_dates(t[-1]))
        sp.plot_global_power(wo, ax=ax_gws, alpha=wv.ar1(case.obs), period_scale=1 / YEAR,
                             period_ticks=PERIOD_TICKS, bands=BANDS, period_lim=MAP_PERIODS)
        ax_gws.set_title("Time mean · % per band", fontsize=9.5)
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
    return save(fig, out / "fig1_observed_wavelet_power.png", args)


def _panels(n_panels: int, n_rows: int, width_per: float = 6.6):
    fig, axes = plt.subplots(1, n_panels, figsize=(max(width_per * n_panels, 8.5), 1.9 + 0.27 * n_rows),
                             squeeze=False)
    return fig, list(axes[0])


def _finish_panels(fig, colors: dict, title: str, footnote: str | None = None) -> None:
    """Suptitle (wrapped to the figure width), family legend and an optional footnote."""
    width = fig.get_size_inches()[0]
    chars = int(width / 0.1)
    fig.suptitle(textwrap.fill(title, width=chars), x=0.01, ha="left", fontsize=12, color=sp.INK)
    ncol = len(colors) if width > 11 else 4
    nrow = int(np.ceil(len(colors) / ncol))
    fig.legend(handles=sp.legend_patches(colors), loc="lower center", ncol=ncol, frameon=False)
    height = fig.get_size_inches()[1]
    bottom = (0.1 + 0.22 * nrow) / height
    if footnote:
        fig.text(0.01, bottom, footnote, ha="left", va="bottom", fontsize=8.5, color=sp.INK_2)
        bottom += 0.25 / height
    lines = title.count("\n") + int(np.ceil(len(title) / chars))
    fig.tight_layout(rect=(0, bottom, 1, 1 - (0.05 + 0.2 * lines) / height))


def fig_spectral_ratio(cases: dict, colors: dict, args, out: Path) -> Path:
    n_rows = max(len(c["order"]) for c in cases.values())
    fig, axes = _panels(len(cases), n_rows, 6.75)
    for ax, (p, c) in zip(axes, cases.items()):
        per, ratios = c["ratio"]
        row_colors = {m: colors[f] for m, f in zip(c["meta"]["model"], c["meta"]["family"])}
        sp.plot_spectral_ratio(per, ratios.loc[c["order"]], ax=ax, period_scale=1 / YEAR,
                               period_ticks=PERIOD_TICKS, row_colors=row_colors,
                               colorbar=ax is axes[-1], title=f"{PROFILE_NAME.get(p, p)}",
                               unresolved_below=BAND_LO / YEAR, period_lim=(0, BAND_HI / YEAR * 1.001),
                               unresolved_label="below the\nsurvey spacing")
        for lo, hi in BANDS.values():
            ax.axvline(hi / YEAR, color=sp.SURFACE, linewidth=1.2)
    _finish_panels(fig, colors, "Variance at each period: model / observed (same times, outside the cone of "
                                "influence and survey gaps); white lines = band limits")
    return save(fig, out / "fig2_variance_ratio_by_period.png", args)


SUPPORT = {"time": ("valid_frac", "of time"), "cells": ("valid_cell_frac", "of cells")}


def band_labels(skill_p: pd.DataFrame, support: str = "time") -> list[str]:
    """Band names with the support of the metric shown: share of the time steps at
    which the whole band is usable (NSE, amplitude) or of the usable time-period
    cells (coherence, phase)."""
    col, text = SUPPORT[support]
    first = skill_p.drop_duplicates("band").set_index("band")
    labels = []
    for b in BANDS:
        if b not in first.index or not np.isfinite(first.loc[b, col]):
            labels.append(b)
            continue
        star = "*" if first.loc[b, "exploratory"] else ""
        labels.append(f"{b}\n{first.loc[b, col]:.0%} {text}{star}")
    return labels


def fig_band_heatmap(skill: pd.DataFrame, cases: dict, colors: dict, metric: str, args, out: Path,
                     name: str, title: str, fmt="{:.2f}", text_metric: str | None = None,
                     support: str = "time") -> Path:
    n_rows = max(len(c["order"]) for c in cases.values())
    fig, axes = _panels(len(cases), n_rows, 7.0)
    for ax, (p, c) in zip(axes, cases.items()):
        df = skill[skill["profile"] == p]
        row_colors = {m: colors[f] for m, f in zip(c["meta"]["model"], c["meta"]["family"])}
        sp.plot_skill_heatmap(df, metric, ax=ax, models=c["order"], bands=list(BANDS), fmt=fmt,
                              text_metric=text_metric, row_colors=row_colors,
                              band_labels=band_labels(df, support), colorbar=ax is axes[-1],
                              title=PROFILE_NAME.get(p, p))
    footnote = EXPLORATORY_NOTE if skill["exploratory"].any() else None
    _finish_panels(fig, colors, title, footnote)
    return save(fig, out / name, args)


def _atlas_axes(n_panels: int, ncols: int = 5):
    nrows = int(np.ceil(n_panels / ncols))
    height = 2.1 * nrows + 1.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, height), sharex=True,
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.8 / height, bottom=0.75 / height,
                        hspace=0.34, wspace=0.06)
    return fig, axes, height


def _panel_title(ax, name: str, color: str | None) -> None:
    if color is None:
        ax.set_title(name, fontsize=9, x=0.0, fontweight="bold")
        return
    ax.set_title(name, fontsize=9, x=0.06)
    ax.scatter([0.015], [1.075], marker="s", s=22, color=color, transform=ax.transAxes,
               clip_on=False, linewidths=0)


def _empty_panel(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=9, color=sp.INK_2)
    ax.set_facecolor(sp.NEUTRAL)


def fig_coherence_atlas(p: str, c: dict, colors: dict, args, out: Path) -> Path:
    order = c["order"]
    fig, axes, height = _atlas_axes(len(order))
    fam = dict(zip(c["meta"]["model"], c["meta"]["family"]))
    mappable = None
    for ax, m in zip(axes.flat, order):
        if c["coh"][m] is None:
            _empty_panel(ax, "constant series:\ncoherence undefined")
        else:
            coh, sig = c["coh"][m]
            mappable = sp.plot_coherence(coh, time=c["case"].time, ax=ax, sig=sig, arrows=False,
                                         period_scale=1 / YEAR, period_ticks=PERIOD_TICKS,
                                         period_label="", colorbar=False, period_lim=MAP_PERIODS)
        _panel_title(ax, m, colors[fam[m]])
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    fig.suptitle(f"{PROFILE_NAME.get(p, p)}: wavelet coherence with the FRF surveys "
                 "(black contour: nominal 95 % level; pale: cone of influence and survey gaps)",
                 x=0.01, ha="left", fontsize=12, color=sp.INK)
    if mappable is not None:
        cax = fig.add_axes((0.36, 0.3 / height, 0.3, 0.1 / height))
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cax.text(-0.02, 0.5, "Coherence R²", transform=cax.transAxes, ha="right", va="center",
                 fontsize=10, color=sp.INK_2)
        cb.outline.set_visible(False)
    fig.legend(handles=sp.legend_patches(colors), loc="lower right", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.99, 0.0))
    return save(fig, out / f"fig7_coherence_atlas_profile{p}.png", args, dpi=min(args.dpi, ATLAS_DPI))


def fig_power_atlas(p: str, c: dict, colors: dict, args, out: Path) -> Path:
    """Each model's own wavelet power (m²) next to the surveys', on one colour scale."""
    panels = ["obs"] + c["order"]
    fig, axes, height = _atlas_axes(len(panels))
    fam = dict(zip(c["meta"]["model"], c["meta"]["family"]))
    levels = sp.power_levels([c["wo"]], period_range=(MAP_PERIODS[0] * YEAR, MAP_PERIODS[1] * YEAR))
    mappable = None
    for ax, key in zip(axes.flat, panels):
        res = c["wo"] if key == "obs" else c["wms"][key]
        if res is None:
            _empty_panel(ax, "constant series:\nno wavelet power")
        else:
            mappable = sp.plot_power(res, time=c["case"].time, ax=ax, physical=True, levels=levels,
                                     period_scale=1 / YEAR, period_ticks=PERIOD_TICKS, period_label="",
                                     colorbar=False, period_lim=MAP_PERIODS)
        _panel_title(ax, "FRF surveys (observed)" if key == "obs" else key,
                     None if key == "obs" else colors[fam[key]])
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
    return save(fig, out / f"fig8_power_atlas_profile{p}.png", args, dpi=min(args.dpi, ATLAS_DPI))


def fig_single_coherence(p: str, m: str, c: dict, args, out: Path) -> Path | None:
    if c["coh"][m] is None:
        return None
    coh, sig = c["coh"][m]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sp.plot_coherence(coh, time=c["case"].time, ax=ax, sig=sig, period_scale=1 / YEAR,
                      period_ticks=PERIOD_TICKS, period_label="Period", period_lim=MAP_PERIODS,
                      title=f"{PROFILE_NAME.get(p, p)}: FRF surveys vs {m} "
                            "(arrows: right = in phase, down = model lags)")
    fig.tight_layout()
    return save(fig, out / "wtc" / f"profile{p}_{slug(m)}.png", args, dpi=args.map_dpi, pdf=False)


# --- summary tables ------------------------------------------------------------------------------

METRICS = dict(median_nse=("nse", "median"), median_sig_frac=("sig_frac", "median"),
               median_std_ratio=("std_ratio", "median"))


def group_tables(skill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-team medians, and per-family medians in which every team counts once.

    Several teams submitted many runs (e.g. 12 deep-learning runs from one
    team), so a plain median over runs would mostly describe the largest team.
    """
    team = (skill.groupby(["profile", "team", "band"], sort=False)
            .agg(n_runs=("model", "nunique"), **METRICS).reset_index())
    per_team = (skill.groupby(["profile", "family", "team", "band"], sort=False)[["nse", "sig_frac", "std_ratio"]]
                .median().reset_index())
    fam = (per_team.groupby(["profile", "family", "band"], sort=False)
           .agg(n_teams=("team", "nunique"), median_nse=("nse", "median"),
                median_sig_frac=("sig_frac", "median"), median_std_ratio=("std_ratio", "median"))
           .reset_index())
    runs = skill.groupby(["profile", "family", "band"], sort=False)["model"].nunique().rename("n_runs")
    fam = fam.join(runs, on=["profile", "family", "band"])
    return fam, team


def dominant_teams(skill: pd.DataFrame) -> list[str]:
    """Notes for families where one team supplies most of the runs."""
    notes = []
    runs = skill.drop_duplicates(["model", "team", "family"])
    for fam, d in runs.groupby("family"):
        counts = d["team"].value_counts()
        if len(d) >= 4 and counts.iloc[0] / len(d) > 0.5:
            share = f"all {len(d)}" if counts.iloc[0] == len(d) else f"{counts.iloc[0]} of the {len(d)}"
            notes.append(f"{share} {fam} runs come from {counts.index[0]}")
    return notes


def _md_table(df: pd.DataFrame, p: str, key: str, names: list[str], col: str, fmt, label,
              exploratory: set[str] = frozenset()) -> list[str]:
    d_p = df[df["profile"] == p]
    head = f"| {label} | " + " | ".join(one_line(b) + ("\\*" if b in exploratory else "") for b in BANDS) + " |"
    rows = [head, "|---" * (len(BANDS) + 1) + "|"]
    for name in names:
        d = d_p[d_p[key] == name].set_index("band")
        if d.empty:
            continue
        count = (f"{int(d['n_runs'].iloc[0])}/{int(d['n_teams'].iloc[0])}" if "n_teams" in d
                 else f"{int(d['n_runs'].iloc[0])}")
        cells = [fmt(d.loc[b, col]) if b in d.index and np.isfinite(d.loc[b, col]) else "–" for b in BANDS]
        rows.append(f"| {name} ({count}) | " + " | ".join(cells) + " |")
    return rows


_DATED = re.compile(r"^(starts|ends) (\d{4}-\d{2}-\d{2}), (.+)$")  # coverage reasons with a date


def left_out_lines(skipped: dict[str, str], aside: dict[str, str], p: str) -> list[str]:
    """Runs left out at one profile; those that start too late (or end too early) share one line."""
    groups, lines = {}, []
    for m, why in sorted(skipped.items(), key=lambda kv: kv[0].lower()):
        if aside.pop(f"{m} at profile {p}", None):
            lines.append(f"  * {m}: {why}; left out before the window was chosen.")
            continue
        hit = _DATED.match(why)
        if hit:
            groups.setdefault((hit[1], hit[3]), []).append((hit[2], m))
        else:
            lines.append(f"  * {m}: {why}.")
    grouped = []
    for (verb, rest), items in groups.items():
        runs = ", ".join(f"{m} ({d})" for d, m in sorted(items))
        many = len(items) > 1
        grouped.append(f"  * {len(items)} {'runs' if many else 'run'} {verb[:-1] if many else verb} {rest}: {runs}.")
    return grouped + lines


def write_summary(skill: pd.DataFrame, fam: pd.DataFrame, team: pd.DataFrame, cases: dict, args,
                  window: dict, out: Path, figures: list[Path], teams_without: list[str]) -> Path:
    c0 = next(iter(cases.values()))
    g0, g1 = (pd.Timestamp(c0["case"].time[i]).date() for i in (0, -1))
    chk_all = [c["checks"] for c in cases.values()]
    lines = ["# Duck single-profile hindcasts (1980-2023 submissions) vs FRF surveys: wavelet comparison", ""]
    if args.note:
        lines += [f"> {args.note}", ""]

    def _by(text):
        return text if text == "set by hand" else f"set by {text}"

    if window["mode"] == "surveys":
        what = ": the whole survey record shared by the profiles"
    else:
        what = ": the longest stretch covered by the surveys and by every run that can take part"
    if "set by hand" in (window["start_set_by"], window["end_set_by"]):
        what = ""
    lines += [f"* Window {window['start']} to {window['end']}{what} (start {_by(window['start_set_by'])}; end "
              f"{_by(window['end_set_by'])}). Weekly grid from {g0} to {g1}; survey gaps longer than "
              f"{days(args.max_gap)} are masked.",
              "* Runs taking part: " + "; ".join(f"profile {p}: {len(c['order'])} runs from "
                                                 f"{c['meta']['team'].nunique()} teams"
                                                 for p, c in cases.items())
              + " (see Runs below). Model families are provisional (edit `config/duck_models.csv`).",
              "* Models were read on the survey days and interpolated like the surveys, so both carry the same "
              "sampling filter." if args.sampling == "surveys" else
              f"* Models were averaged over {args.step} (centred) and compared with the interpolated surveys.",
              "* Every series is compared as an anomaly (mean removed, trend kept), so fixed offsets between "
              "submissions do not matter; units, sign convention and shoreline definition still have to agree.",
              "* **Support.** NSE, amplitude and correlation use the times at which the whole band lies outside "
              "the cone of influence and the survey gaps ('% of time', `valid_frac`). Coherence and phase use "
              "every trustworthy time-period cell ('% of cells', `valid_cell_frac`), so their support is wider. "
              "Each figure shows the support of its own metric.",
              "* **Significance** thresholds are nominal: AR(1) red-noise surrogates on the regular grid, not yet "
              "checked against the survey sampling (see Caveats).",
              f"* Bands with fewer than {MIN_CYCLES:.0f} usable cycles are marked \\* and treated as exploratory: "
              "shown, not ranked.",
              "* Monthly = month-to-month changes (1.5-4 months) that the surveys can see; not single storms.", ""]

    kinds = "(.png)" if args.no_pdf else "(.png, .pdf)"
    lines += ["## Figures", "",
              ("PNG versions of each figure are in this folder." if args.no_pdf else
               "PNG and PDF versions of each figure are in this folder."), "",
              "1. `fig1_observed_wavelet_power`: the surveys, their wavelet power, and each band's share of the "
              "resolved variance (only trustworthy cells count).",
              "2. `fig2_variance_ratio_by_period`: model / observed variance at each period "
              "(red = too much, blue = too little).",
              "3. `fig3_coherence_by_band`: share of each band's cells where model and surveys are coherent above "
              "a nominal 95 % threshold (co-variation regardless of amplitude).",
              "4. `fig4_amplitude_by_band`: model / observed standard deviation of the band signal.",
              "5. `fig5_nse_by_band`: Nash-Sutcliffe efficiency of the band signal (1 = perfect, "
              "0 = no better than the mean).",
              "6. `fig6_timing_by_band`: phase where coherent; text = lag in days (+ = model late).",
              "7. `fig7_coherence_atlas_profile*`: coherence maps, one panel per model"
              + ("." if args.no_maps else "; `wtc/` has one larger map per model with phase arrows (PNG only)."),
              "8. `fig8_power_atlas_profile*`: each model's own wavelet power in m², on the same colour "
              "scale as the surveys (first panel).", ""]

    lines += ["## Runs", "",
              "A run takes part when it has a value on every survey day the grid uses: the surveys in the window "
              "and the one just outside each end.", ""]
    aside = dict(window.get("set_aside", {}))
    later = any(_DATED.match(why) for c in cases.values() for why in c["case"].skipped.values())
    for p, c in cases.items():
        lines.append(f"* Profile {p}: {len(c['order'])} runs from {c['meta']['team'].nunique()} teams take part.")
        lines += left_out_lines(c["case"].skipped, aside, p)
    for key, why in aside.items():  # normally empty: a run set aside also fails in the final window
        lines.append(f"* Set aside while choosing the common window: {key} ({why}).")
    if later and window["mode"] == "surveys":
        lines += ["", "Runs that do not cover the whole record are compared with the others over a shorter common "
                  "window: `python scripts/duck_wavelets.py --window common --out outputs/duck_1980-2023_common`."]
    lines += ["", "`models.csv` lists every run with its coverage and status.", ""]

    lines += ["## Results by band", ""]
    for p, c in cases.items():
        df = skill[skill["profile"] == p]
        lines += [f"### {PROFILE_NAME.get(p, p)}", "",
                  "| Band | share of obs. variance | usable: % of time / % of cells (dates, cycles) | best NSE | "
                  "largest coherent share |", "|---|---|---|---|---|"]
        for band in BANDS:
            b = df[df["band"] == band]
            r0 = b.iloc[0]
            support = (f"{r0['valid_frac']:.0%} / {r0['valid_cell_frac']:.0%} ({month(r0['valid_start'])} to "
                       f"{month(r0['valid_end'])}, {cycles(r0['n_cycles'])} cycles)"
                       if np.isfinite(r0["n_cycles"]) else "–")
            share = f"{r0['var_frac_obs']:.0%}" if np.isfinite(r0["var_frac_obs"]) else "–"
            if r0["exploratory"]:
                nse = coh = "exploratory: not ranked"
            else:
                top_nse = b.dropna(subset=["nse"]).nlargest(3, "nse")
                top_coh = b.dropna(subset=["sig_frac"]).nlargest(3, "sig_frac")
                nse = ", ".join(f"{r.model} ({fmt_nse(r.nse)})" for r in top_nse.itertuples()) or "–"
                coh = ", ".join(f"{r.model} ({r.sig_frac:.0%})" for r in top_coh.itertuples()) or "–"
            lines.append(f"| {one_line(band)} | {share} | {support} | {nse} | {coh} |")
        fams = [f for f in FAMILY_ORDER if f in set(fam["family"])] + sorted(set(fam["family"]) - set(FAMILY_ORDER))
        teams = sorted(set(team.loc[team["profile"] == p, "team"]), key=str.lower)
        expl = set(df.loc[df["exploratory"], "band"])
        lines += ["", "Median NSE per family (runs/teams; each team counts once):", ""]
        lines += _md_table(fam, p, "family", fams, "median_nse", fmt_nse, "Family (runs/teams)", expl)
        lines += ["", "Median coherent share per family (runs/teams; each team counts once):", ""]
        lines += _md_table(fam, p, "family", fams, "median_sig_frac", lambda v: f"{v:.0%}", "Family (runs/teams)",
                           expl)
        lines += ["", "Median NSE per team (runs):", ""]
        lines += _md_table(team, p, "team", teams, "median_nse", fmt_nse, "Team (runs)", expl)
        if expl:
            lines += ["", EXPLORATORY_MD]
        lines.append("")
    notes = dominant_teams(skill)
    if notes:
        lines += ["In some families the medians mostly reflect one team: " + "; ".join(notes) + ".", ""]

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
        chk = c["checks"]
        lines.append(f"* Profile {p}: {chk['n']} surveys used ({chk['first']} to {chk['last']}), "
                     f"median spacing {chk['median_spacing']:.0f} days (5-year medians {chk['block_min']:.0f}-"
                     f"{chk['block_max']:.0f} days, sparsest from {chk['sparsest']}), {chk['n_long_gaps']} gaps "
                     f"longer than {days(args.max_gap)} (longest {chk['longest_gap']:.0f} days).")
        for where, d0, d1, gap in chk["edge_gaps"]:
            lines.append(f"  * The record {'starts' if where == 'start' else 'ends'} with a {gap:.0f}-day gap "
                         f"(surveys of {d0} and {d1}), so the {where} of the grid is masked.")
        for d, v in chk["spikes"]:
            lines.append(f"  * Isolated survey {v:+.0f} m off its neighbours on {d} (possible outlier; kept).")
        for d0, d1, v in chk["steps"]:
            lines.append(f"  * Lasting step of {v:+.0f} m between the surveys of {d0} and {d1} "
                         f"(e.g. a nourishment). To leave it out: `--profiles {p} --end {d0}`.")
        tr = c["trend"]
        for r in tr[tr["detrended_std"] < 0.15 * tr["obs_detrended_std"]].itertuples():
            lines.append(f"  * {r.model} is almost a straight line here (spread around its trend "
                         f"{r.detrended_std:.1f} m vs {r.obs_detrended_std:.1f} m observed).")
        for m in sorted(c["constant"]):
            lines.append(f"  * {m} does not vary: amplitude and NSE are reported, coherence is undefined.")
        sk = skill[skill["profile"] == p]
        first = sk.drop_duplicates("band").set_index("band")
        worst = sk.groupby("band")["nse"].min()
        for b in BANDS:
            if b not in first.index:
                continue
            r0 = first.loc[b]
            if r0["exploratory"] and np.isfinite(r0["n_cycles"]):
                lines.append(f"  * {one_line(b)}: {r0['n_cycles']:.1f} cycles usable ({month(r0['valid_start'])} "
                             f"to {month(r0['valid_end'])}); shown for completeness, not ranked. Coherence there "
                             "is just as uncertain as NSE.")
            elif np.isfinite(r0["var_frac_obs"]) and r0["var_frac_obs"] < 0.05 and worst.get(b, 0) < -3:
                lines.append(f"  * {one_line(b)}: little observed variance ({r0['var_frac_obs']:.0%}), so small "
                             "errors give large negative NSE.")
        same, close = c["dups"]
        for a, b in same:
            lines.append(f"  * {a} and {b} are identical in this window (counted twice in the figures).")
        for a, b, r in close:
            lines.append(f"  * {a} and {b} are nearly identical here (r = {r:.3f}).")
    if teams_without:
        lines.append(f"* Team folders without a `mipDuck_1980-2023*.csv` file: {', '.join(teams_without)}.")

    last_survey = max(c["case"].obs_raw.index.max() for c in cases.values()).date()
    sparse = max(chk["block_max"] for chk in chk_all)
    lines += ["", "## Caveats and known limitations", "",
              f"* The FRF surveys provided to the teams end on {last_survey}, so every score here is in-sample: "
              "the teams could calibrate on these surveys. The scores show how well each model reproduces "
              "the record, not blind skill." + (" Models in the 'Equilibrium + DA' family assimilate the surveys."
                                               if "Equilibrium + DA" in set(skill["family"]) else ""),
              f"* Resolution: in the sparsest years the surveys are about {sparse:.0f} days apart, so periods "
              f"shorter than about {2 * sparse:.0f} days are not resolved then. The monthly band describes "
              "month-to-month variability that the surveys can see, not single storms.",
              "* Significance: the red-noise surrogates are generated on the regular grid and do not go through "
              "the survey sampling and interpolation, so the 95 % levels (contours in fig1 and fig7, thresholds "
              "in fig3) are nominal, not calibrated, especially at short periods.",
              "* The coherent share (`sig_frac`) is a descriptive area fraction, not a band-level p-value: "
              "significant cells come in patches, so values well above 5 % can still arise by chance.",
              "* Lags are weighted averages over coherent cells; they can hide opposite lags at different times.",
              "* The coherence mask uses each transform's gap mask; the extra smoothing in coherence spreads gap "
              "effects a little further than that.",
              "* A band is a range of time scales, not a process. A lasting step (e.g. a nourishment) or a trend "
              "puts variance into several bands at once.",
              "* Families are provisional labels for grouping only.", ""]
    lines += ["## Files", ""] + [f"* `{f.relative_to(out).with_suffix('')}` {kinds}" for f in figures] + \
             ["* `skill_by_band.csv`: every metric for every model, band and profile (sig_frac = coherent "
              "share, mean_rsq = mean coherence, lag in days, valid_* = support of each score)",
              "* `family_summary.csv`: medians per family, each team counted once",
              "* `team_summary.csv`: medians per team",
              "* `trend_by_model.csv`: linear trend and detrended spread of every model",
              "* `models.csv`: every run per profile with its coverage, status and the reason if left out",
              "* `variance_ratio_by_period_profile*.csv`: model / observed variance per period (years)",
              "* `run_info.json`: settings, window, runs per profile, code version, software versions and "
              "SHA-256 of every input file", ""]
    path = out / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- run record --------------------------------------------------------------------------------

def models_table(subs: pd.DataFrame, cases: dict, profiles) -> pd.DataFrame:
    """Every run per profile: coverage, status and the reason if it was left out."""
    cov = duck.coverage(subs, profiles).set_index(["model", "profile"])
    rows = []
    for p, c in cases.items():
        included = set(c["order"])
        for r in subs.itertuples():
            first = last = pd.NaT
            if (r.model, str(p)) in cov.index:
                first, last = cov.loc[(r.model, str(p)), ["first", "last"]]
            rows.append(dict(profile=p, model=r.model, team=r.team, family=r.family, file=r.file,
                             first_day=first, last_day=last,
                             status="included" if r.model in included else "not included",
                             reason="" if r.model in included else c["case"].skipped.get(r.model, "")))
    return pd.DataFrame(rows)


def profile_record(c: dict) -> dict:
    case = c["case"]
    t = pd.DatetimeIndex(case.time)
    return dict(grid_start=str(t[0].date()), grid_end=str(t[-1].date()), n_grid=int(t.size),
                n_gap_points=int(case.gap.sum()), n_runs=len(c["order"]), n_teams=int(c["meta"]["team"].nunique()),
                runs=c["order"], not_included=case.skipped)


def code_version() -> dict:
    """Git commit of the repository and whether it has uncommitted changes."""
    def _git(*cmd):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    from shoreshop3 import __version__
    return dict(package=__version__, git_commit=commit or "unknown",
                git_uncommitted_changes=bool(status) if commit else None)


def software_versions() -> dict:
    import matplotlib as mpl
    import scipy

    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, matplotlib=mpl.__version__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_digests(raw: Path, subs: pd.DataFrame) -> dict:
    """SHA-256 of the survey archive and of every submission read."""
    raw = Path(raw)
    frf = raw / duck.FRF_ZIP
    out = dict(frf_profiles=dict(path=str(duck.FRF_ZIP), bytes=frf.stat().st_size, sha256=_sha256(frf))
               if frf.exists() else "not found as a zip")
    out["submissions"] = [dict(path=str(Path(r.path).relative_to(raw)) if Path(r.path).is_relative_to(raw)
                               else Path(r.path).name, bytes=Path(r.path).stat().st_size,
                               sha256=_sha256(Path(r.path))) for r in subs.itertuples()]
    return out


# --- main ------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "duck_1980-2023")
    ap.add_argument("--window", choices=duck.WINDOW_MODES, default="surveys",
                    help="surveys: the whole survey record, runs that do not cover it are left out (default); "
                         "common: the longest window covered by every run that can take part")
    ap.add_argument("--start", default=None, help="fix the first day by hand (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="fix the last day by hand (YYYY-MM-DD)")
    ap.add_argument("--step", default="7D")
    ap.add_argument("--max-gap", default="60D")
    ap.add_argument("--sampling", choices=duck.SAMPLING, default="surveys",
                    help="surveys: read models on the survey days (default); mean: weekly means")
    ap.add_argument("--n-surrogates", type=int, default=300)
    ap.add_argument("--profiles", nargs="+", default=list(duck.PROFILES))
    ap.add_argument("--no-maps", action="store_true", help="skip one-map-per-model figures")
    ap.add_argument("--dpi", type=int, default=300, help="resolution of the PNG figures")
    ap.add_argument("--map-dpi", type=int, default=200, help="resolution of the one-per-model maps")
    ap.add_argument("--no-pdf", action="store_true", help="PNG only")
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
    colors = family_colors(subs["family"])  # from every submission, so colours match between runs
    teams_without = sorted(set(duck.team_folders(args.raw)) - set(subs["folder"]), key=str.lower)
    max_gap_days = pd.Timedelta(args.max_gap) / pd.Timedelta("1D")
    w = duck.choose_window(args.raw, args.profiles, subs, mode=args.window,
                           start=None if args.start in (None, "auto") else args.start,
                           end=None if args.end in (None, "auto") else args.end)
    start_s, end_s = str(w.start.date()), str(w.end.date())
    window = dict(start=start_s, end=end_s, mode=w.mode, start_set_by=w.start_set_by, end_set_by=w.end_set_by,
                  set_aside=w.set_aside)
    print(f"window ({w.mode}) {start_s} to {end_s} (start: {w.start_set_by}; end: {w.end_set_by})")
    for key, why in w.set_aside.items():
        print(f"  set aside before choosing the window: {key}: {why}")

    cases, skills = {}, []
    for p in args.profiles:
        case = duck.build_case(args.raw, p, start_s, end_s, step=args.step, max_gap=args.max_gap,
                               sampling=args.sampling, submissions=subs)
        order = ordered_models(case.meta)
        aligned = pd.DataFrame({"obs": case.obs, "obs_gap": case.gap, **case.models},
                               index=pd.DatetimeIndex(case.time, name="time"))
        aligned.to_csv(PROCESSED / f"duck_profile{p}_{start_s}_{end_s}_{args.step}_{args.sampling}.csv")
        print(f"profile {p}: {len(order)} runs, {case.obs.size} grid points, "
              f"{int(case.gap.sum())} in survey gaps; not included: {len(case.skipped)}")
        for m, why in case.skipped.items():
            print(f"  - {m}: {why}")

        skill = wv.compare_models(case.obs, {m: case.models[m] for m in order}, case.dt_days, BANDS,
                                  invalid_obs=case.gap, n_surrogates=args.n_surrogates,
                                  cache_dir=INTERIM / "wtc_sig", alpha_decimals=2)
        info = case.meta.set_index("model")[["team", "family", "file"]]
        skill = add_support_dates(skill.join(info, on="model"), case)
        skill.insert(0, "profile", p)
        skills.append(skill)

        wo = wv.cwt(case.obs, case.dt_days, invalid=case.gap)
        a_obs = wv.ar1(case.obs)
        ratio, coh, wms, constant = {}, {}, {}, set()
        inv_mod = case.gap if case.sampling == "surveys" else None
        per = wo.periods
        obs_resolved = wo.valid_mask().any(axis=1)
        for m in order:
            x = case.models[m]
            if wv.is_constant(x, case.obs):
                constant.add(m)
                wms[m] = coh[m] = None
                ratio[m] = np.where(obs_resolved, 0.0, np.nan)  # no variance at any resolved period
                continue
            wm = wms[m] = wv.cwt(x, case.dt_days, invalid=inv_mod)
            _, p_obs, p_mod = wv.paired_global_spectra(wo, wm)
            ratio[m] = p_mod / p_obs
            sig = wv.coherence_significance(case.obs.size, case.dt_days, a_obs, wv.ar1(x),
                                            n_surrogates=args.n_surrogates,
                                            cache_dir=INTERIM / "wtc_sig", alpha_decimals=2)
            coh[m] = (wv.wavelet_coherence(case.obs, x, case.dt_days, invalid_x=case.gap), sig)
        ratio_df = pd.DataFrame(ratio).T
        ratio_df.columns = np.round(per / YEAR, 4)
        ratio_df.rename_axis(index="model", columns="period_years").to_csv(
            out / f"variance_ratio_by_period_profile{p}.csv")
        cases[p] = dict(case=case, wo=wo, wms=wms, order=order, meta=case.meta, constant=constant,
                        ratio=(per, pd.DataFrame(ratio).T), coh=coh, checks=survey_checks(case, max_gap_days),
                        dups=duplicate_models(case, order), trend=trends(case, order, case.meta))
        print(f"  analysed in {time.time() - t_start:.0f} s")

    present = set().union(*(set(c["meta"]["family"]) for c in cases.values()))
    colors = {f: col for f, col in colors.items() if f in present}  # legend: families taking part only
    skill_all = pd.concat(skills, ignore_index=True)
    fam, team = group_tables(skill_all)
    skill_all.assign(band=skill_all["band"].map(one_line)).to_csv(out / "skill_by_band.csv", index=False)
    fam.assign(band=fam["band"].map(one_line)).to_csv(out / "family_summary.csv", index=False)
    team.assign(band=team["band"].map(one_line)).to_csv(out / "team_summary.csv", index=False)
    pd.concat([c["trend"] for c in cases.values()], ignore_index=True).to_csv(
        out / "trend_by_model.csv", index=False, float_format="%.4g")
    models_table(subs, cases, args.profiles).to_csv(out / "models.csv", index=False)

    figures = [
        fig_observations(cases, args, out),
        fig_spectral_ratio(cases, colors, args, out),
        fig_band_heatmap(skill_all, cases, colors, "sig_frac", args, out, "fig3_coherence_by_band.png",
                         "Co-variation: share of each band's cells where model and surveys are coherent above a "
                         "nominal 95 % threshold (AR(1) red noise on the regular grid; not yet checked against "
                         "the survey sampling)", fmt="{:.0%}", support="cells"),
        fig_band_heatmap(skill_all, cases, colors, "std_ratio", args, out, "fig4_amplitude_by_band.png",
                         "Amplitude: model / observed standard deviation of the band signal (1 = right size)"),
        fig_band_heatmap(skill_all, cases, colors, "nse", args, out, "fig5_nse_by_band.png",
                         "Overall skill: Nash-Sutcliffe efficiency of the band signal (1 = perfect, "
                         "0 = no better than the mean)", fmt=fmt_nse),
        fig_band_heatmap(skill_all, cases, colors, "phase_deg", args, out, "fig6_timing_by_band.png",
                         "Timing where coherent: colour = phase, text = lag in days (+ = model late; an average "
                         "that can hide opposite lags)", fmt="{:.0f} d", text_metric="lag", support="cells"),
    ]
    figures += [fig_coherence_atlas(p, c, colors, args, out) for p, c in cases.items()]
    figures += [fig_power_atlas(p, c, colors, args, out) for p, c in cases.items()]
    if not args.no_maps:
        for p, c in cases.items():
            for m in c["order"]:
                fig_single_coherence(p, m, c, args, out)
    summary = write_summary(skill_all, fam, team, cases, args, window, out, figures, teams_without)

    def _rel(v):
        try:
            return str(Path(v).resolve().relative_to(ROOT))
        except ValueError:
            return Path(v).name

    info = dict(settings={k: (_rel(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                window=window,
                profiles={p: profile_record(c) for p, c in cases.items()},
                bands_days={one_line(k): v for k, v in BANDS.items()}, min_cycles=MIN_CYCLES,
                significance=dict(method="AR(1) red-noise surrogate pairs on the regular grid (nominal)",
                                  n_surrogates=args.n_surrogates, seed=0, alpha_decimals=2, level=0.95),
                code=code_version(), software=software_versions(),
                inputs=input_digests(args.raw, subs),
                runtime_s=round(time.time() - t_start), created=time.strftime("%Y-%m-%d %H:%M"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time() - t_start:.0f} s. See {summary}")


if __name__ == "__main__":
    main()
