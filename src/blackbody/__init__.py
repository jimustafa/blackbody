from functools import wraps
from typing import Callable, cast

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
    INTEGRATED_PLANCK_DISTRIBUTIONS,
    PLANCK_DISTRIBUTIONS,
)
from ._types import (
    AreaUnit,
    SpectralUnit,
)

__all__ = [
    "FLUX_UNITS",
    "SPECTRAL_UNITS",
    "AREA_UNITS",
    "AREA_FACTORS",
    "RADIATION_CONSTANTS",
    "STEFAN_BOLTZMANN_CONSTANTS",
    "WIEN_CONSTANTS",
    "INTEGRATED_PLANCK_DISTRIBUTIONS",
    "PLANCK_DISTRIBUTIONS",
    "spectral_radiant_sterance",
    "spectral_photon_sterance",
    "integrated_radiant_sterance",
    "integrated_photon_sterance",
]


def check_arguments_spectral(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(
        T: ArrayLike, x: ArrayLike, *, spectral_unit: SpectralUnit, area_unit: AreaUnit
    ) -> NDArray[np.float64]:
        if spectral_unit not in SPECTRAL_UNITS:
            raise ValueError(f"`spectral_unit` must be one of {repr(SPECTRAL_UNITS)}")

        if area_unit not in AREA_UNITS:
            raise ValueError(f"`area_unit` must be one of {repr(AREA_UNITS)}")

        T = np.atleast_1d(T).astype(np.float64, casting="safe").reshape(-1, 1)
        x = np.atleast_1d(x).astype(np.float64, casting="safe").reshape(1, -1)

        if not np.all(T > 0):
            raise ValueError("`T` must be greater than zero")

        if not np.all(x > 0):
            raise ValueError("`x` must be greater than zero")

        return fn(T, x, spectral_unit=spectral_unit, area_unit=area_unit)

    return wrapper


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

    (c1, c2) = RADIATION_CONSTANTS[("energy", spectral_unit)]

    _planck_distribution = PLANCK_DISTRIBUTIONS[("energy", spectral_unit)]

    return np.squeeze(_planck_distribution(c1, c2, T, x)) * AREA_FACTORS[area_unit]


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

    _planck_distribution = PLANCK_DISTRIBUTIONS[("photon", spectral_unit)]

    return np.squeeze(_planck_distribution(c1, c2, T, x)) * AREA_FACTORS[area_unit]


def check_arguments_integrated(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(
        T: ArrayLike, x_ab: ArrayLike, *, spectral_unit: SpectralUnit, area_unit: AreaUnit
    ) -> NDArray[np.float64]:
        if spectral_unit not in SPECTRAL_UNITS:
            raise ValueError(f"`spectral_unit` must be one of {repr(SPECTRAL_UNITS)}")

        if area_unit not in AREA_UNITS:
            raise ValueError(f"`area_unit` must be one of {repr(AREA_UNITS)}")

        T = np.atleast_1d(T)
        x_ab = np.atleast_2d(x_ab)

        if not x_ab.shape[-1] == 2:
            raise ValueError("`x_ab` must have shape (..., 2)")

        T = T.astype(np.float64, casting="safe").reshape(-1, 1, 1)
        x_ab = x_ab.astype(np.float64, casting="safe").reshape(1, -1, 2)

        if not np.all(T > 0):
            raise ValueError("`T` must be greater than zero")

        if not np.all(x_ab > 0):
            raise ValueError("`x_ab` must be greater than zero")

        return fn(T, x_ab, spectral_unit=spectral_unit, area_unit=area_unit)

    return wrapper


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

    (c1, c2) = RADIATION_CONSTANTS[("energy", spectral_unit)]

    _integrated_planck_distribution = INTEGRATED_PLANCK_DISTRIBUTIONS[("energy", spectral_unit)]

    i1 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 0])
    i2 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 1])

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

    _integrated_planck_distribution = INTEGRATED_PLANCK_DISTRIBUTIONS[("photon", spectral_unit)]

    i1 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 0])
    i2 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 1])

    return np.squeeze(np.abs(i2 - i1)) * AREA_FACTORS[area_unit]
