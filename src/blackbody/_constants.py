import typing

import numpy as np
from scipy.constants import c, h, k, sigma
from scipy.special import lambertw, zeta

from ._types import AreaUnit, SpectralUnit

__all__ = [
    "FLUX_UNITS",
    "SPECTRAL_UNITS",
    "AREA_UNITS",
    "AREA_FACTORS",
    "STEFAN_BOLTZMANN_CONSTANTS",
    "WIEN_CONSTANTS",
    "RADIATION_CONSTANTS",
]


EMAXEXP = np.log(2) * np.finfo(np.float64).maxexp

FLUX_UNITS: tuple[str, ...] = (
    "radiant",
    "photon",
)

SPECTRAL_UNITS: tuple[str, ...] = typing.get_args(SpectralUnit)

AREA_UNITS: tuple[str, ...] = typing.get_args(AreaUnit)

AREA_FACTORS: dict[str, float] = {
    "m^2": 1,
    "cm^2": 1 / 1e4,
}

STEFAN_BOLTZMANN_CONSTANTS: dict[str, float] = {
    "radiant": sigma,
    "photon": 4 * np.pi * zeta(3) * k**3 / h**3 / c**2,
}

WIEN_CONSTANTS: dict[tuple[str, str], float] = {
    ("radiant", "Hz"): k / h * abs((3 + lambertw(-3 * np.exp(-3), 0))),
    ("photon", "Hz"): k / h * abs((2 + lambertw(-2 * np.exp(-2), 0))),
    ("radiant", "THz"): 1 / 1e12 * k / h * abs((3 + lambertw(-3 * np.exp(-3), 0))),
    ("photon", "THz"): 1 / 1e12 * k / h * abs((2 + lambertw(-2 * np.exp(-2), 0))),
    ("radiant", "um"): 1e6 * h * c / k / abs((5 + lambertw(-5 * np.exp(-5), 0))),
    ("photon", "um"): 1e6 * h * c / k / abs((4 + lambertw(-4 * np.exp(-4), 0))),
    ("radiant", "cm^-1"): k / (100 * h * c) * abs((3 + lambertw(-3 * np.exp(-3), 0))),
    ("photon", "cm^-1"): k / (100 * h * c) * abs((2 + lambertw(-2 * np.exp(-2), 0))),
}

RADIATION_CONSTANTS: dict[tuple[str, str], tuple[float, float]] = {
    ("radiant", "Hz"): (2 * h / c**2, h / k),
    ("photon", "Hz"): (2 / c**2, h / k),
    ("radiant", "THz"): (2 * h / c**2 * 1e48, h / k * 1e12),
    ("photon", "THz"): (2 / c**2 * 1e36, h / k * 1e12),
    ("radiant", "um"): (2 * h * c**2 * 1e24, h * c / k * 1e6),
    ("photon", "um"): (2 * c * 1e18, h * c / k * 1e6),
    ("radiant", "cm^-1"): (2 * h * c**2 * 1e8, h * c / k * 100),
    ("photon", "cm^-1"): (2 * c * 1e6, h * c / k * 100),
}
