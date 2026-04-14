from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._constants import (
    AREA_FACTORS,
    AREA_UNITS,
    FLUX_UNITS,
    RADIATION_CONSTANTS,
    SPECTRAL_UNITS,
    STEFAN_BOLTZMANN_CONSTANTS,
    WIEN_CONSTANTS,
)
from ._planck import (
    PLANCK_INTEGRALS,
    PLANCK_SPECTRA,
)
from ._types import (
    AreaUnit,
    SpectralUnit,
)
from ._validators import (
    check_arguments_integrated,
    check_arguments_spectral,
)

__all__ = [
    "FLUX_UNITS",
    "SPECTRAL_UNITS",
    "AREA_UNITS",
    "AREA_FACTORS",
    "RADIATION_CONSTANTS",
    "STEFAN_BOLTZMANN_CONSTANTS",
    "WIEN_CONSTANTS",
    "PLANCK_INTEGRALS",
    "PLANCK_SPECTRA",
    "spectral_radiant_sterance",
    "spectral_photon_sterance",
    "integrated_radiant_sterance",
    "integrated_photon_sterance",
]


@check_arguments_spectral
def spectral_radiant_sterance(
    T: ArrayLike, x: ArrayLike, *, spectral_unit: SpectralUnit, area_unit: AreaUnit
) -> NDArray[np.float64]:
    """
    Spectral radiant sterance

    Arguments:
        T: blackbody temperature (K)
        x: spectral variable in units of `spectral_unit`
        spectral_unit: units of the spectral variable
        area_unit: units of the area element

    Returns:
        spectral radiant sterance
    """
    T = cast(NDArray[np.float64], T)
    x = cast(NDArray[np.float64], x)

    (c1, c2) = RADIATION_CONSTANTS[("radiant", spectral_unit)]

    _planck_spectrum = PLANCK_SPECTRA[("radiant", spectral_unit)]

    return np.squeeze(_planck_spectrum(c1, c2, T, x)) * AREA_FACTORS[area_unit]


@check_arguments_spectral
def spectral_photon_sterance(
    T: ArrayLike, x: ArrayLike, *, spectral_unit: SpectralUnit, area_unit: AreaUnit
) -> NDArray[np.float64]:
    """
    Spectral photon sterance

    Arguments:
        T: blackbody temperature (K)
        x: spectral variable in units of `spectral_unit`
        spectral_unit: units of the spectral variable
        area_unit: units of the area element

    Returns:
        spectral photon sterance

    """
    T = cast(NDArray[np.float64], T)
    x = cast(NDArray[np.float64], x)

    (c1, c2) = RADIATION_CONSTANTS[("photon", spectral_unit)]

    _planck_spectrum = PLANCK_SPECTRA[("photon", spectral_unit)]

    return np.squeeze(_planck_spectrum(c1, c2, T, x)) * AREA_FACTORS[area_unit]


@check_arguments_integrated
def integrated_radiant_sterance(
    T: ArrayLike, x_ab: ArrayLike, *, spectral_unit: SpectralUnit, area_unit: AreaUnit
) -> NDArray[np.float64]:
    """
    Integrated radiant sterance

    Arguments:
        T: blackbody temperature (K)
        x_ab: spectral interval in units of `spectral_unit`
        spectral_unit: units of the spectral variable
        area_unit: units of the area element

    Returns:
        integrated radiant sterance
    """
    T = cast(NDArray[np.float64], T)
    x_ab = cast(NDArray[np.float64], x_ab)

    (c1, c2) = RADIATION_CONSTANTS[("radiant", spectral_unit)]

    _planck_integral = PLANCK_INTEGRALS[("radiant", spectral_unit)]

    i1 = _planck_integral(c1, c2, T, x_ab[..., 0])
    i2 = _planck_integral(c1, c2, T, x_ab[..., 1])

    return np.squeeze(np.abs(i2 - i1)) * AREA_FACTORS[area_unit]


@check_arguments_integrated
def integrated_photon_sterance(
    T: ArrayLike, x_ab: ArrayLike, *, spectral_unit: SpectralUnit, area_unit: AreaUnit
) -> NDArray[np.float64]:
    """
    Integrated photon sterance

    Arguments:
        T: blackbody temperature (K)
        x_ab: spectral interval in units of `spectral_unit`
        spectral_unit: units of the spectral variable
        area_unit: units of the area element

    Returns:
        integrated photon sterance
    """
    T = cast(NDArray[np.float64], T)
    x_ab = cast(NDArray[np.float64], x_ab)

    (c1, c2) = RADIATION_CONSTANTS[("photon", spectral_unit)]

    _planck_integral = PLANCK_INTEGRALS[("photon", spectral_unit)]

    i1 = _planck_integral(c1, c2, T, x_ab[..., 0])
    i2 = _planck_integral(c1, c2, T, x_ab[..., 1])

    return np.squeeze(np.abs(i2 - i1)) * AREA_FACTORS[area_unit]
