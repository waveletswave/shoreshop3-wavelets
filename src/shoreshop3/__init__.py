"""ShoreShop3 data organisation and wavelet analysis tools."""

from . import duck, inventory, paths, timeseries, wavelets
from .timeseries import RegularSeries, regularize
from .wavelets import (
    ar1,
    band_skill,
    coherence_significance,
    compare_models,
    cross_wavelet,
    cwt,
    global_spectrum,
    power_significance,
    reconstruct,
    wavelet_coherence,
)

__version__ = "0.2.0"

__all__ = [
    "duck",
    "inventory",
    "paths",
    "timeseries",
    "wavelets",
    "RegularSeries",
    "regularize",
    "ar1",
    "band_skill",
    "coherence_significance",
    "compare_models",
    "cross_wavelet",
    "cwt",
    "global_spectrum",
    "power_significance",
    "reconstruct",
    "wavelet_coherence",
]
