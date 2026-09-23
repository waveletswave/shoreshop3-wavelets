"""Continuous wavelet tools for comparing ShoreShop3 model output with observations.

Implements the Morlet (omega0 = 6) continuous wavelet transform of Torrence &
Compo (1998), the cross-wavelet transform, and wavelet coherence using the
smoothing operator of Torrence & Webster (1999) as implemented in the Grinsted
et al. (2004) MATLAB toolbox, plus red-noise (AR1) significance tests.

Everything is unit-agnostic: ``dt`` can be in days, years, or metres (for an
alongshore transform). Periods, scales and the cone of influence (COI) come
back in the same unit.

Phase convention: for ``wavelet_coherence(x, y)`` a positive phase means ``x``
leads ``y``. With ``x`` = observations and ``y`` = a model, a positive phase
(and a positive ``lag`` in :func:`band_skill`) means the model lags the
observations.

References
----------
Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet analysis.
    Bull. Amer. Meteor. Soc. 79, 61-78.
Torrence, C. & Webster, P. J. (1999). Interdecadal changes in the ENSO-monsoon
    system. J. Climate 12, 2679-2690.
Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). Application of the cross
    wavelet transform and wavelet coherence to geophysical time series.
    Nonlin. Processes Geophys. 11, 561-566.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import optimize, signal, special, stats

__all__ = [
    "OMEGA0",
    "CWTResult",
    "CrossWaveletResult",
    "CoherenceResult",
    "fourier_factor",
    "make_scales",
    "cwt",
    "reconstruct",
    "band_rows",
    "scale_average",
    "band_variance_fraction",
    "ar1",
    "ar1_noise",
    "red_noise_spectrum",
    "power_significance",
    "global_spectrum",
    "global_significance",
    "paired_global_spectra",
    "cross_wavelet",
    "xwt_significance",
    "wavelet_coherence",
    "coherence_significance",
    "band_skill",
    "compare_models",
]

# --- Morlet constants (Torrence & Compo 1998, Tables 1-2) --------------------
OMEGA0 = 6.0  #: non-dimensional frequency of the Morlet wavelet
C_DELTA = 0.776  #: reconstruction factor for Morlet, omega0 = 6
PSI0_0 = np.pi ** -0.25  #: psi_0(0) for Morlet
GAMMA = 2.32  #: time decorrelation factor (global-spectrum dof)
DJ0 = 0.60  #: scale decorrelation length in octaves (coherence smoothing)
MAX_GAP_FRACTION = 0.25  #: default: distrust a coefficient if >25 % of its wavelet energy sits on gaps
POWER_FLOOR = 1e-6  #: normalised power treated as "no variance" in coherence


def fourier_factor() -> float:
    """Fourier period / wavelet scale for the Morlet wavelet (~1.033)."""
    return 4.0 * np.pi / (OMEGA0 + np.sqrt(2.0 + OMEGA0**2))


def _psi0_hat(s_omega: np.ndarray) -> np.ndarray:
    """Fourier transform of the unit-energy Morlet mother wavelet."""
    return PSI0_0 * np.where(s_omega > 0, np.exp(-0.5 * (s_omega - OMEGA0) ** 2), 0.0)


def _pad_length(n: int, pad: str | None) -> int:
    if pad is None or pad == "none":
        return n
    if pad == "tc":  # Torrence & Compo wavelet.m: 2**(round(log2 n) + 1)
        return int(2 ** (int(np.log2(n) + 0.4999) + 1))
    if pad == "pow2":  # next power of two (pycwt / Grinsted default)
        return int(2 ** np.ceil(np.log2(n)))
    raise ValueError("pad must be 'tc', 'pow2' or None")


def make_scales(n: int, dt: float, dj: float = 1 / 12, s0: float | None = None,
                J: int | None = None) -> np.ndarray:
    """Geometric scale grid s_j = s0 * 2**(j*dj), j = 0..J (T&C eqs. 9-10)."""
    if s0 is None:
        s0 = 2.0 * dt
    if J is None:
        J = int(np.floor(np.log2(n * dt / s0) / dj))
    if J < 0:
        raise ValueError("series too short for the requested smallest scale s0")
    return s0 * 2.0 ** (np.arange(J + 1) * dj)


@dataclass
class CWTResult:
    """Output of :func:`cwt`. Arrays are (n_scales, n_times)."""

    coeffs: np.ndarray  #: complex wavelet coefficients W_n(s)
    scales: np.ndarray  #: wavelet scales (unit of dt)
    periods: np.ndarray  #: equivalent Fourier periods (unit of dt)
    coi: np.ndarray  #: period at the COI edge for each time step
    dt: float
    dj: float
    mean: float  #: mean removed before the transform
    trend: np.ndarray | None  #: linear trend removed (None if detrend=False)
    variance: float  #: variance of the analysed (anomaly) series
    standardized: bool  #: True if the anomaly was divided by its std first
    invalid: np.ndarray | None = None  #: per-sample flags (e.g. long filled gaps)
    max_gap_fraction: float = MAX_GAP_FRACTION
    _gap_frac: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def n(self) -> int:
        return self.coeffs.shape[1]

    @property
    def power(self) -> np.ndarray:
        """|W|^2 in the units of the analysed series squared."""
        return np.abs(self.coeffs) ** 2

    def normalized_power(self) -> np.ndarray:
        """|W|^2 / sigma^2: power relative to white noise of the same variance."""
        return self.power if self.standardized else self.power / self.variance

    def gap_fraction(self) -> np.ndarray:
        """Share of each wavelet's energy that falls on flagged samples, (n_scales, n_times).

        The Morlet energy envelope |psi|^2 is a Gaussian with std s/sqrt(2), so a
        short gap barely affects long periods but dominates short ones.
        """
        if self.invalid is None or not self.invalid.any():
            return np.zeros(self.coeffs.shape)
        if self._gap_frac is None:
            npad = int(2 ** np.ceil(np.log2(2 * self.n)))  # no wrap-around
            k = 2.0 * np.pi * sfft.rfftfreq(npad)
            F = np.exp(-0.25 * (self.scales[:, None] / self.dt) ** 2 * k[None, :] ** 2)
            g = sfft.rfft(self.invalid.astype(float), n=npad)
            frac = sfft.irfft(F * g[None, :], n=npad, axis=1)[:, : self.n]
            self._gap_frac = np.clip(frac, 0.0, 1.0)
        return self._gap_frac

    def gap_mask(self) -> np.ndarray:
        """Boolean (n_scales, n_times): True where gaps dominate the coefficient."""
        if self.invalid is None or not self.invalid.any():
            return np.zeros(self.coeffs.shape, bool)
        return self.gap_fraction() > self.max_gap_fraction

    def valid_mask(self) -> np.ndarray:
        """Boolean (n_scales, n_times): True outside the COI and not dominated by gaps."""
        return (self.periods[:, None] <= self.coi[None, :]) & ~self.gap_mask()


def cwt(x: Sequence[float], dt: float, *, dj: float = 1 / 12, s0: float | None = None,
        J: int | None = None, pad: str | None = "tc", detrend: bool = False,
        standardize: bool = False, invalid: Sequence[bool] | None = None,
        max_gap_fraction: float = MAX_GAP_FRACTION) -> CWTResult:
    """Morlet continuous wavelet transform (Torrence & Compo 1998).

    Parameters
    ----------
    x : 1-D evenly spaced series without NaNs. Fill gaps first (see
        :func:`shoreshop3.timeseries.regularize`) and pass the gap flags as
        ``invalid`` so the affected coefficients are masked.
    dt : sampling interval (any unit).
    dj : scale resolution in octaves (1/12 = 12 voices per octave).
    s0 : smallest scale (default 2*dt). J : number of scales - 1 (default: up
        to the record length).
    pad : 'tc' (T&C zero padding, default), 'pow2', or None.
    detrend : remove a least-squares linear trend before the transform.
    standardize : divide the anomaly by its standard deviation.
    invalid : optional boolean flags per sample (True = do not trust), e.g.
        ``RegularSeries.gap``. Coefficients whose wavelet puts more than
        ``max_gap_fraction`` of its energy on flagged samples are masked.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    n = x.size
    if n < 8:
        raise ValueError("series too short (need at least 8 samples)")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN/inf: fill gaps first (shoreshop3.timeseries."
                         "regularize) and pass the gap flags via `invalid=`")
    if dt <= 0:
        raise ValueError("dt must be positive")

    mean = float(x.mean())
    y = x - mean
    trend = None
    if detrend:
        tt = np.arange(n, dtype=float)
        trend = np.polyval(np.polyfit(tt, y, 1), tt)
        y = y - trend
    variance = float(y.var())
    if variance <= 0:
        raise ValueError("series has zero variance")
    if standardize:
        y = y / np.sqrt(variance)

    scales = make_scales(n, dt, dj, s0, J)
    npad = _pad_length(n, pad)
    Y = sfft.fft(y, n=npad)
    omega = 2.0 * np.pi * sfft.fftfreq(npad, d=dt)
    pos = slice(1, (npad + 1) // 2)  # Morlet (Heaviside) has no negative frequencies
    spec = np.zeros((scales.size, npad), dtype=complex)
    spec[:, pos] = (np.sqrt(2.0 * np.pi * scales[:, None] / dt)
                    * _psi0_hat(scales[:, None] * omega[None, pos]) * Y[None, pos])
    W = sfft.ifft(spec, axis=1, workers=-1)[:, :n]

    ff = fourier_factor()
    idx = np.arange(n)
    dist = np.minimum(idx, n - 1 - idx).astype(float)
    dist[dist == 0] = 1e-5
    coi = ff / np.sqrt(2.0) * dt * dist

    inv = None
    if invalid is not None:
        inv = np.asarray(invalid, dtype=bool)
        if inv.shape != (n,):
            raise ValueError("invalid must have the same length as x")
    return CWTResult(coeffs=W, scales=scales, periods=ff * scales, coi=coi, dt=float(dt),
                     dj=float(dj), mean=mean, trend=trend, variance=variance,
                     standardized=standardize, invalid=inv,
                     max_gap_fraction=float(max_gap_fraction))


def band_rows(res: CWTResult | "CoherenceResult", band: tuple[float, float]) -> np.ndarray:
    """Boolean selector of scales whose period lies in [band[0], band[1])."""
    lo, hi = band
    return (res.periods >= lo) & (res.periods < hi)


def reconstruct(res: CWTResult, band: tuple[float, float] | None = None) -> np.ndarray:
    """Inverse CWT (T&C eq. 11), optionally restricted to a period band.

    With ``band=None`` the full series is rebuilt (mean and trend added back).
    With a band, only the band-limited anomaly is returned (no mean/trend), in
    the physical units of the input. Bands defined as [lo, hi) are additive.
    """
    rows = np.ones(res.scales.size, bool) if band is None else band_rows(res, band)
    factor = res.dj * np.sqrt(res.dt) / (C_DELTA * PSI0_0)
    out = factor * np.sum(res.coeffs[rows].real / np.sqrt(res.scales[rows])[:, None], axis=0)
    if res.standardized:
        out = out * np.sqrt(res.variance)
    if band is None:
        out = out + res.mean + (0.0 if res.trend is None else res.trend)
    return out


def scale_average(res: CWTResult, band: tuple[float, float]) -> np.ndarray:
    """Scale-averaged wavelet power over a period band (T&C eq. 24), per time step."""
    rows = band_rows(res, band)
    p = res.power[rows] / res.scales[rows][:, None]
    out = res.dj * res.dt / C_DELTA * p.sum(axis=0)
    return out * res.variance if res.standardized else out


def band_variance_fraction(res: CWTResult, band: tuple[float, float]) -> float:
    """Fraction of the analysed variance carried by a period band (T&C eq. 14)."""
    return float(scale_average(res, band).mean() / res.variance)


# --- Red-noise background and significance --------------------------------------

def ar1(x: Sequence[float]) -> float:
    """Lag-1 autoregressive coefficient, alpha = (a1 + sqrt(a2)) / 2 (T&C Sec. 4).

    Note: shoreline series with trends or strong seasonal cycles give alpha
    close to 1 (clipped at 0.999); detrend/deseasonalise first if you want a
    stricter test.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    c0 = float(x @ x)
    if c0 == 0:
        return 0.0
    a1 = float(x[:-1] @ x[1:]) / c0
    a2 = float(x[:-2] @ x[2:]) / c0
    alpha = (a1 + np.sqrt(a2)) / 2.0 if a2 > 0 else a1
    return float(np.clip(alpha, 0.0, 0.999))


def ar1_noise(n: int, alpha: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Stationary unit-variance AR1 red noise of length n."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    rng = np.random.default_rng() if rng is None else rng
    e = rng.standard_normal(n) * np.sqrt(1.0 - alpha**2)
    e[0] = rng.standard_normal()  # start in the stationary distribution
    return signal.lfilter([1.0], [1.0, -alpha], e)


def red_noise_spectrum(periods: np.ndarray, dt: float, alpha: float) -> np.ndarray:
    """Normalised AR1 Fourier spectrum at the given periods (T&C eq. 16)."""
    f = dt / np.asarray(periods, dtype=float)
    return (1.0 - alpha**2) / (1.0 + alpha**2 - 2.0 * alpha * np.cos(2.0 * np.pi * f))


def power_significance(res: CWTResult, alpha: float, level: float = 0.95) -> np.ndarray:
    """Per-scale threshold for :meth:`CWTResult.normalized_power` (T&C eq. 18).

    ``normalized_power() / threshold[:, None] >= 1`` marks power that is
    significant against an AR1 background with coefficient ``alpha``.
    """
    dof = 2.0  # complex Morlet
    return red_noise_spectrum(res.periods, res.dt, alpha) * stats.chi2.ppf(level, dof) / dof


def global_spectrum(res: CWTResult, exclude_coi: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged normalised power per scale and the number of samples used."""
    p = res.normalized_power()
    if not exclude_coi:
        return p.mean(axis=1), np.full(res.scales.size, res.n)
    mask = res.valid_mask()
    count = mask.sum(axis=1)
    total = np.where(mask, p, 0.0).sum(axis=1)
    gws = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    return gws, count


def global_significance(res: CWTResult, alpha: float, level: float = 0.95,
                        n_used: np.ndarray | None = None) -> np.ndarray:
    """Significance threshold for :func:`global_spectrum` (T&C eq. 23)."""
    n_a = np.full(res.scales.size, res.n) if n_used is None else np.asarray(n_used, float)
    dof = 2.0 * np.sqrt(1.0 + (np.maximum(n_a, 1) * res.dt / (GAMMA * res.scales)) ** 2)
    dof = np.maximum(dof, 2.0)
    return red_noise_spectrum(res.periods, res.dt, alpha) * stats.chi2.ppf(level, dof) / dof


# --- Cross-wavelet transform and coherence --------------------------------------

@dataclass
class CrossWaveletResult:
    """Output of :func:`cross_wavelet` (both inputs standardised)."""

    wx: CWTResult
    wy: CWTResult
    wxy: np.ndarray  #: Wx * conj(Wy)

    @property
    def periods(self) -> np.ndarray:
        return self.wx.periods

    @property
    def power(self) -> np.ndarray:
        """|Wxy| / (sigma_x sigma_y)."""
        return np.abs(self.wxy)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.wxy)

    def valid_mask(self) -> np.ndarray:
        return self.wx.valid_mask() & self.wy.valid_mask()


def _check_pair(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length and sampling")
    return x, y


def cross_wavelet(x, y, dt: float, *, dj: float = 1 / 12, s0: float | None = None,
                  J: int | None = None, pad: str | None = "tc", detrend: bool = False,
                  invalid_x=None, invalid_y=None) -> CrossWaveletResult:
    """Cross-wavelet transform of two standardised series (Grinsted et al. 2004)."""
    x, y = _check_pair(x, y)
    kw = dict(dj=dj, s0=s0, J=J, pad=pad, detrend=detrend, standardize=True)
    wx = cwt(x, dt, invalid=invalid_x, **kw)
    wy = cwt(y, dt, invalid=invalid_y, **kw)
    return CrossWaveletResult(wx, wy, wx.coeffs * np.conj(wy.coeffs))


def _z2(level: float) -> float:
    """Quantile of sqrt(A*B), A, B ~ chi2(2) independent (T&C eq. 30; 3.999 at 95%)."""
    f = lambda u: 1.0 - 2.0 * np.sqrt(u) * special.k1(2.0 * np.sqrt(u)) - level  # noqa: E731
    return 2.0 * np.sqrt(optimize.brentq(f, 1e-12, 1e4))


def xwt_significance(res: CrossWaveletResult, alpha_x: float, alpha_y: float,
                     level: float = 0.95) -> np.ndarray:
    """Per-scale threshold for :attr:`CrossWaveletResult.power` (T&C eq. 31)."""
    px = red_noise_spectrum(res.periods, res.wx.dt, alpha_x)
    py = red_noise_spectrum(res.periods, res.wx.dt, alpha_y)
    return _z2(level) / 2.0 * np.sqrt(px * py)


def _scale_kernel(dj: float, dj0: float = DJ0) -> np.ndarray:
    """Boxcar in scale used by Grinsted's smoothwavelet.m (width ~dj0 octaves)."""
    steps = dj0 / (dj * 2.0)
    m = int(np.floor(steps + 0.5))  # MATLAB-style rounding
    frac = steps % 1.0
    kernel = np.r_[frac, np.ones(max(2 * m - 1, 1)), frac]
    return kernel / kernel.sum()


def _gauss_filter(scales: np.ndarray, dt: float, npad: int) -> np.ndarray:
    """Fourier transform of a Gaussian with std = scale, on rfft frequencies."""
    k = 2.0 * np.pi * sfft.rfftfreq(npad)  # rad / sample
    return np.exp(-0.5 * (scales[:, None] / dt) ** 2 * k[None, :] ** 2)


def _smooth(W: np.ndarray, scales: np.ndarray, dt: float, dj: float,
            scale_kernel: np.ndarray | None = None, F: np.ndarray | None = None) -> np.ndarray:
    """Coherence smoothing (Grinsted's smoothwavelet.m).

    Gaussian in time with std = scale (via FFT, zero padded to a power of 2),
    then a boxcar of ~0.6 octaves in scale.
    """
    n = W.shape[1]
    npad = int(2 ** np.ceil(np.log2(n)))
    if F is None:
        F = _gauss_filter(scales, dt, npad)
    if np.isrealobj(W):
        T = sfft.irfft(F * sfft.rfft(W, n=npad, axis=1, workers=-1), n=npad, axis=1,
                       workers=-1)[:, :n]
    else:  # symmetric filter on the full (two-sided) spectrum
        Fc = np.concatenate([F, F[:, npad // 2 - 1:0:-1]], axis=1)
        T = sfft.ifft(Fc * sfft.fft(W, n=npad, axis=1, workers=-1), axis=1, workers=-1)[:, :n]
    kernel = _scale_kernel(dj) if scale_kernel is None else np.asarray(scale_kernel, float)
    return _convolve_scales(T, kernel)


def _convolve_scales(T: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-padded 'same' convolution along the scale axis (= conv2d(T, k[:, None], 'same'))."""
    J = T.shape[0]
    c = (kernel.size - 1) // 2
    out = np.zeros_like(T)
    for i, w in enumerate(kernel):
        s = c - i  # out[j] += w * T[j + s]
        if abs(s) >= J or w == 0:
            continue
        if s >= 0:
            out[: J - s] += w * T[s:]
        else:
            out[-s:] += w * T[: J + s]
    return out


@dataclass
class CoherenceResult:
    """Output of :func:`wavelet_coherence`. Arrays are (n_scales, n_times)."""

    rsq: np.ndarray  #: squared wavelet coherence in [0, 1]
    phase: np.ndarray  #: phase of the smoothed cross spectrum (rad); > 0: x leads y
    wx: CWTResult
    wy: CWTResult

    @property
    def periods(self) -> np.ndarray:
        return self.wx.periods

    @property
    def scales(self) -> np.ndarray:
        return self.wx.scales

    @property
    def dt(self) -> float:
        return self.wx.dt

    @property
    def coi(self) -> np.ndarray:
        return np.minimum(self.wx.coi, self.wy.coi)

    def gap_mask(self) -> np.ndarray:
        return self.wx.gap_mask() | self.wy.gap_mask()

    def valid_mask(self) -> np.ndarray:
        return self.wx.valid_mask() & self.wy.valid_mask()


def _coherence_from(wx: CWTResult, wy: CWTResult, scale_kernel: np.ndarray | None = None,
                    power_floor: float = POWER_FLOOR) -> CoherenceResult:
    if wx.coeffs.shape != wy.coeffs.shape or not np.allclose(wx.scales, wy.scales):
        raise ValueError("both transforms must use the same scales and length")
    sinv = 1.0 / wx.scales[:, None]
    F = _gauss_filter(wx.scales, wx.dt, int(2 ** np.ceil(np.log2(wx.n))))
    args = (wx.scales, wx.dt, wx.dj, scale_kernel, F)
    # coherence is scale-free: work with standardised coefficients
    cx = wx.coeffs if wx.standardized else wx.coeffs / np.sqrt(wx.variance)
    cy = wy.coeffs if wy.standardized else wy.coeffs / np.sqrt(wy.variance)
    sx = _smooth(sinv * np.abs(cx) ** 2, *args)
    sy = _smooth(sinv * np.abs(cy) ** 2, *args)
    sxy = _smooth(sinv * cx * np.conj(cy), *args)
    # a series with (numerically) no variance at a scale shares none: R^2 -> 0
    floor = power_floor * sinv
    with np.errstate(divide="ignore", invalid="ignore"):
        rsq = np.abs(sxy) ** 2 / ((sx + floor) * (sy + floor))
    rsq = np.clip(np.nan_to_num(rsq, nan=0.0), 0.0, 1.0)
    return CoherenceResult(rsq=rsq, phase=np.angle(sxy), wx=wx, wy=wy)


def wavelet_coherence(x, y, dt: float, *, dj: float = 1 / 12, s0: float | None = None,
                      J: int | None = None, pad: str | None = "tc", detrend: bool = False,
                      invalid_x=None, invalid_y=None,
                      scale_kernel: np.ndarray | None = None,
                      power_floor: float = POWER_FLOOR) -> CoherenceResult:
    """Squared wavelet coherence and phase (Torrence & Webster 1999; Grinsted 2004).

    R^2 = |S(Wxy/s)|^2 / (S(|Wx|^2/s) S(|Wy|^2/s)). Values near 1 mean the two
    series co-vary at that period and time, regardless of amplitude. Where a
    series has essentially no variance (normalised power below
    ``power_floor``, e.g. a smooth model at short periods) R^2 goes to 0
    instead of the unstable 0/0.
    """
    x, y = _check_pair(x, y)
    kw = dict(dj=dj, s0=s0, J=J, pad=pad, detrend=detrend, standardize=True)
    wx = cwt(x, dt, invalid=invalid_x, **kw)
    wy = cwt(y, dt, invalid=invalid_y, **kw)
    return _coherence_from(wx, wy, scale_kernel, power_floor)


_SIG_CACHE: dict[str, np.ndarray] = {}


def coherence_significance(n: int, dt: float, alpha_x: float, alpha_y: float, *,
                           dj: float = 1 / 12, s0: float | None = None, J: int | None = None,
                           pad: str | None = "tc", level: float = 0.95,
                           n_surrogates: int = 300, length_factor: int = 2, seed: int = 0,
                           cache_dir: str | Path | None = None,
                           scale_kernel: np.ndarray | None = None,
                           alpha_decimals: int = 3) -> np.ndarray:
    """Monte Carlo significance level of R^2 per scale against AR1 noise.

    Follows Grinsted's wtcsignif.m: pairs of red-noise series with the given
    AR1 coefficients (rounded to ``alpha_decimals``, max 0.999) are transformed with the same
    scales, and the ``level`` quantile of R^2 outside the COI is taken per scale.
    Surrogates are ``length_factor`` times longer than the data to sample long
    periods better. Results are cached in memory and optionally on disk.
    Use the same ``scale_kernel`` as in :func:`wavelet_coherence` (default:
    Grinsted's 0.6-octave boxcar). Rounding the AR1 coefficients to 2 decimals
    lets many series share one Monte Carlo run.

    Returns an array (n_scales,) aligned with ``make_scales(n, dt, dj, s0, J)``.
    """
    scales = make_scales(n, dt, dj, s0, J)
    s0_, J_ = scales[0], scales.size - 1
    ax, ay = (min(max(round(float(a), int(alpha_decimals)), 0.0), 0.999) for a in (alpha_x, alpha_y))
    kern = None if scale_kernel is None else [round(float(v), 10) for v in scale_kernel]
    key_dict = dict(n=int(n), dt=float(dt), dj=float(dj), s0=float(s0_), J=int(J_), pad=pad,
                    level=float(level), n_surrogates=int(n_surrogates),
                    length_factor=int(length_factor), seed=int(seed), ax=ax, ay=ay,
                    kernel=kern, v=2)
    key = hashlib.sha1(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()[:16]
    if key in _SIG_CACHE:
        return _SIG_CACHE[key].copy()
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"wtc_sig_{key}.npy"
        if path.exists():
            _SIG_CACHE[key] = np.load(path)
            return _SIG_CACHE[key].copy()

    rng = np.random.default_rng(seed)
    m = int(n * max(1, length_factor))
    nbins = 1000
    hist = np.zeros((scales.size, nbins))
    rows = np.arange(scales.size)[:, None]
    for _ in range(int(n_surrogates)):
        kw = dict(dj=dj, s0=s0_, J=J_, pad=pad, standardize=True)
        wx = cwt(ar1_noise(m, ax, rng), dt, **kw)
        wy = cwt(ar1_noise(m, ay, rng), dt, **kw)
        res = _coherence_from(wx, wy, scale_kernel)
        mask = res.valid_mask()
        bins = np.minimum((res.rsq * nbins).astype(int), nbins - 1)
        r = np.broadcast_to(rows, bins.shape)
        np.add.at(hist, (r[mask], bins[mask]), 1.0)

    centers = (np.arange(nbins) + 0.5) / nbins
    sig = np.full(scales.size, np.nan)
    for j in range(scales.size):
        h = hist[j]
        nz = h > 0
        if nz.sum() < 2:
            continue
        cdf = np.cumsum(h[nz])
        cdf = (cdf - 0.5) / cdf[-1]
        sig[j] = np.interp(level, cdf, centers[nz])
    _SIG_CACHE[key] = sig
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, sig)
    return sig.copy()


def paired_global_spectra(obs: CWTResult, model: CWTResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-averaged power of obs and model over the cells valid for both.

    Returns (periods, obs_power, model_power) in the squared units of the
    series (e.g. m^2); NaN where a scale has no valid cell. The ratio
    model/obs tells at which periods a model is too damped (< 1) or too
    active (> 1), using exactly the same times for both.
    """
    if obs.coeffs.shape != model.coeffs.shape:
        raise ValueError("both transforms must share scales and length")
    mask = obs.valid_mask() & model.valid_mask()
    count = mask.sum(axis=1)

    def _mean(res: CWTResult) -> np.ndarray:
        p = res.power * (res.variance if res.standardized else 1.0)
        tot = np.where(mask, p, 0.0).sum(axis=1)
        return np.divide(tot, count, out=np.full(tot.shape, np.nan), where=count > 0)

    return obs.periods, _mean(obs), _mean(model)


# --- Scale-dependent skill ------------------------------------------------------------

def _anomaly(x: np.ndarray, detrend: bool) -> np.ndarray:
    """Mean-removed (optionally linearly detrended) copy, as analysed by cwt()."""
    y = np.asarray(x, float) - np.mean(x)
    if detrend:
        tt = np.arange(y.size, dtype=float)
        y = y - np.polyval(np.polyfit(tt, y, 1), tt)
    return y


def _circmean(phase: np.ndarray) -> float:
    if phase.size == 0:
        return np.nan
    return float(np.angle(np.exp(1j * phase).mean()))


def _band_metrics(wo: CWTResult, wm: CWTResult, coh: CoherenceResult, name: str,
                  band: tuple[float, float], rsq_sig: np.ndarray | None,
                  phase_min_rsq: float = 0.5, min_phase_frac: float = 0.1) -> dict:
    rows = band_rows(wo, band)
    out = dict(band=name, period_min=band[0], period_max=band[1], n_scales=int(rows.sum()))
    nan_keys = ["valid_frac", "n_valid", "var_frac_obs", "var_frac_model", "std_obs",
                "std_model", "std_ratio", "corr", "rmse", "nse", "mean_rsq", "sig_frac",
                "phase_frac", "phase_deg", "lag"]
    out.update({k: np.nan for k in nan_keys})
    out["n_valid"] = 0
    if not rows.any():
        return out

    valid = coh.valid_mask()[rows]
    out["var_frac_obs"] = band_variance_fraction(wo, band)
    out["var_frac_model"] = band_variance_fraction(wm, band) * wm.variance / wo.variance

    # band-limited signals, compared where every scale of the band is trustworthy
    t_ok = valid.all(axis=0)
    out["valid_frac"] = float(t_ok.mean())
    out["n_valid"] = int(t_ok.sum())
    if t_ok.sum() >= 3:
        ob, mb = reconstruct(wo, band)[t_ok], reconstruct(wm, band)[t_ok]
        so, sm = ob.std(), mb.std()
        out["std_obs"], out["std_model"] = float(so), float(sm)
        out["std_ratio"] = float(sm / so) if so > 0 else np.nan
        out["corr"] = float(np.corrcoef(ob, mb)[0, 1]) if so > 0 and sm > 0 else np.nan
        err = mb - ob
        out["rmse"] = float(np.sqrt(np.mean(err**2)))
        denom = np.sum((ob - ob.mean()) ** 2)
        out["nse"] = float(1.0 - np.sum(err**2) / denom) if denom > 0 else np.nan

    # coherence statistics use every trustworthy (scale, time) cell of the band
    if not valid.any():
        return out
    rsq_b = coh.rsq[rows]
    out["mean_rsq"] = float(rsq_b[valid].mean())
    thr = rsq_sig[rows][:, None] if rsq_sig is not None else None
    if thr is not None:
        out["sig_frac"] = float((rsq_b >= thr)[valid].mean())
    # phase / lag only where the series are coherent (significant, else R^2 >= phase_min_rsq)
    coherent = valid & (rsq_b >= (np.nan_to_num(thr, nan=np.inf) if thr is not None else phase_min_rsq))
    out["phase_frac"] = float(coherent.sum() / valid.sum())
    if out["phase_frac"] >= min_phase_frac:
        # band cross-covariance per scale: sum of Wx conj(Wy) / s over coherent cells
        wxy = (wo.coeffs[rows] * np.conj(wm.coeffs[rows])) / wo.scales[rows][:, None]
        per_scale = np.where(coherent, wxy, 0.0).sum(axis=1)
        weight = np.abs(per_scale)
        if weight.sum() > 0:
            lags = np.angle(per_scale) / (2 * np.pi) * wo.periods[rows]
            out["lag"] = float(np.sum(weight * lags) / weight.sum())
            out["phase_deg"] = float(np.degrees(np.angle(per_scale.sum())))
    return out


def band_skill(obs, model, dt: float, bands: Mapping[str, tuple[float, float]], *,
               dj: float = 1 / 12, s0: float | None = None, J: int | None = None,
               pad: str | None = "tc", detrend: bool = False, invalid_obs=None,
               invalid_model=None, rsq_sig: np.ndarray | None = None,
               phase_min_rsq: float = 0.5) -> pd.DataFrame:
    """Scale-by-scale comparison of a model with observations.

    For each named period band [lo, hi) the band-limited signals (inverse CWT)
    of obs and model are compared where the whole band is outside the COI and
    away from flagged gaps.

    Columns
    -------
    valid_frac     fraction of time steps where the band can be evaluated
    var_frac_obs   share of observed variance in the band (importance of band)
    var_frac_model model band variance / observed total variance
    std_ratio      model / obs standard deviation of the band signal (amplitude)
    corr, rmse     correlation and RMSE of the band-limited signals
    nse            Nash-Sutcliffe efficiency of the band signal (1 = perfect)
    mean_rsq       mean squared wavelet coherence in the band (timing/co-variation)
    sig_frac       fraction of band cells with R^2 above ``rsq_sig`` (if given)
    phase_frac     fraction of band cells coherent enough to read a phase
    phase_deg      phase of the band cross-spectrum (deg); > 0: model lags obs
    lag            the phase as a time shift (unit of dt); > 0: model lags obs

    Signal metrics (std_ratio .. nse) need the whole band outside the COI, so
    long bands may be NaN in short records; coherence metrics use every
    trustworthy cell. Phase and lag use only coherent cells (R^2 above
    ``rsq_sig`` if given, else ``phase_min_rsq``) and are NaN when fewer than
    10 % of the band's cells are coherent (timing is meaningless there).
    """
    obs, model = _check_pair(obs, model)
    kw = dict(dj=dj, s0=s0, J=J, pad=pad, detrend=detrend)
    wo = cwt(obs, dt, invalid=invalid_obs, **kw)
    wm = cwt(model, dt, invalid=invalid_model, **kw)
    coh = _coherence_from(wo, wm)
    return pd.DataFrame([_band_metrics(wo, wm, coh, k, tuple(v), rsq_sig, phase_min_rsq)
                         for k, v in bands.items()])


def compare_models(obs, models: Mapping[str, Sequence[float]], dt: float,
                   bands: Mapping[str, tuple[float, float]], *, dj: float = 1 / 12,
                   s0: float | None = None, J: int | None = None, pad: str | None = "tc",
                   detrend: bool = False, invalid_obs=None,
                   invalid_models: Mapping[str, Sequence[bool]] | None = None,
                   n_surrogates: int = 0, cache_dir: str | Path | None = None,
                   seed: int = 0, phase_min_rsq: float = 0.5,
                   alpha_decimals: int = 3) -> pd.DataFrame:
    """Run :func:`band_skill` for several models against the same observations.

    Returns a tidy table with one row per (model, band). Set ``n_surrogates``
    (e.g. 300) to add Monte Carlo coherence significance (``sig_frac``).
    """
    obs = np.asarray(obs, float)
    kw = dict(dj=dj, s0=s0, J=J, pad=pad, detrend=detrend)
    wo = cwt(obs, dt, invalid=invalid_obs, **kw)
    a_obs = ar1(_anomaly(obs, detrend)) if n_surrogates else None
    frames = []
    for name, series in models.items():
        series = np.asarray(series, float)
        if series.shape != obs.shape:
            raise ValueError(f"model {name!r} is not on the observation grid")
        inv = None if invalid_models is None else invalid_models.get(name)
        wm = cwt(series, dt, invalid=inv, **kw)
        coh = _coherence_from(wo, wm)
        sig = None
        if n_surrogates:
            a_mod = ar1(_anomaly(series, detrend))
            sig = coherence_significance(obs.size, dt, a_obs, a_mod, dj=dj, s0=wo.scales[0],
                                         J=wo.scales.size - 1, pad=pad,
                                         n_surrogates=n_surrogates, seed=seed,
                                         cache_dir=cache_dir, alpha_decimals=alpha_decimals)
        df = pd.DataFrame([_band_metrics(wo, wm, coh, k, tuple(v), sig, phase_min_rsq)
                           for k, v in bands.items()])
        df.insert(0, "model", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
