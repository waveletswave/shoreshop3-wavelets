"""Put irregular observation series on the evenly spaced grid the CWT needs.

Typical use: satellite-derived or surveyed shorelines sampled every few days
with cloud or survey gaps, compared with daily model output.

>>> reg = regularize(obs_time, obs_x, "1D", max_gap="30D")
>>> res = cwt(reg.values, reg.dt, invalid=reg.gap)   # gaps masked like edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["RegularSeries", "regularize"]

_EPOCH = pd.Timestamp("1970-01-01")
_DAY = pd.Timedelta("1D")


@dataclass
class RegularSeries:
    """Evenly spaced, gap-filled series plus flags describing the filling."""

    time: np.ndarray  #: grid (datetime64[ns] for datetime input, else float)
    values: np.ndarray  #: values on the grid, no NaNs
    dt: float  #: grid step as a float (days for datetime input)
    filled: np.ndarray  #: True where no observation fell within half a step
    gap: np.ndarray  #: True inside gaps longer than ``max_gap``; pass to cwt(invalid=...)

    def to_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.time, name="value")


def _as_float_time(times) -> tuple[np.ndarray, bool]:
    idx = pd.Index(times)
    if idx.dtype == object:
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(idx))
        except (ValueError, TypeError):
            pass
    if isinstance(idx, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(idx.dtype):
        idx = pd.DatetimeIndex(idx)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        days = (idx - _EPOCH) / _DAY
        return np.asarray(days, dtype=float), True
    return np.asarray(idx, dtype=float), False


def _as_float_step(step, is_datetime: bool) -> float:
    if step is None:
        return np.inf
    if is_datetime:
        return float(pd.Timedelta(step) / _DAY)
    return float(step)


def regularize(times: Sequence, values: Sequence[float], step, *, method: str = "interp",
               max_gap=None, start=None, end=None) -> RegularSeries:
    """Resample an irregular series onto an even grid.

    Parameters
    ----------
    times : datetimes (or strings) or numbers. values : same length; NaNs dropped.
    step : grid spacing, e.g. "1D" / "7D" for datetimes, or a number.
    method : "interp" (linear interpolation between samples) or "bin"
        (average samples within +/- step/2 of each grid point, then interpolate
        empty bins). Use "bin" when samples are denser than the grid.
    max_gap : gaps longer than this (same kind as ``step``) are flagged in
        ``gap``; grid points outside the observed range are always flagged.
    start, end : optional grid limits (default: first/last observation).
    """
    t, is_dt = _as_float_time(times)
    x = np.asarray(values, dtype=float)
    if t.shape != x.shape:
        raise ValueError("times and values must have the same length")
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size < 2:
        raise ValueError("need at least two valid observations")
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    # average duplicate time stamps
    tu, inv = np.unique(t, return_inverse=True)
    if tu.size != t.size:
        x = np.bincount(inv, weights=x) / np.bincount(inv)
        t = tu

    dt = _as_float_step(step, is_dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("step must be positive")
    gap_len = _as_float_step(max_gap, is_dt)
    t0 = t[0] if start is None else _as_float_time([start])[0][0]
    t1 = t[-1] if end is None else _as_float_time([end])[0][0]
    grid = t0 + dt * np.arange(int(np.floor((t1 - t0) / dt + 1e-9)) + 1)

    outside = (grid < t[0] - 1e-9) | (grid > t[-1] + 1e-9)
    if method == "interp":
        vals = np.interp(grid, t, x)
        i = np.clip(np.searchsorted(t, grid), 1, t.size - 1)
        span = t[i] - t[i - 1]
        near = np.minimum(np.abs(t[i] - grid), np.abs(grid - t[i - 1])) <= dt / 2
        filled = ~near
        gap = (span > gap_len) & ~near
    elif method == "bin":
        b = np.rint((t - t0) / dt).astype(int)
        keep = (b >= 0) & (b < grid.size)
        sums = np.bincount(b[keep], weights=x[keep], minlength=grid.size)
        counts = np.bincount(b[keep], minlength=grid.size)
        filled = counts == 0
        if (~filled).sum() < 2:
            raise ValueError("fewer than two non-empty bins; use a smaller step")
        vals = np.empty(grid.size)
        vals[~filled] = sums[~filled] / counts[~filled]
        vals[filled] = np.interp(grid[filled], grid[~filled], vals[~filled])
        gap = np.zeros(grid.size, bool)
        if np.isfinite(gap_len):
            # runs of empty bins: span between neighbouring non-empty bins
            edges = np.flatnonzero(np.diff(np.r_[0, filled.astype(int), 0]))
            for a, z in zip(edges[::2], edges[1::2]):
                if (z - a + 1) * dt > gap_len:
                    gap[a:z] = True
    else:
        raise ValueError("method must be 'interp' or 'bin'")

    gap = gap | outside
    filled = filled | outside
    out_time = (_EPOCH + pd.to_timedelta(grid, unit="D")).to_numpy() if is_dt else grid
    return RegularSeries(time=out_time, values=vals, dt=dt, filled=filled, gap=gap)
