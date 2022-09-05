from functools import wraps

import numpy as np

from ._constants import (
    FLUX_UNITS,
    SPECTRAL_UNITS,
    AREA_UNITS,
    AREA_FACTORS,
    RADIATION_CONSTANTS,
    STEFAN_BOLTZMANN_CONSTANTS,
    WIEN_CONSTANTS,
)
from ._planck import (
    INTEGRATED_PLANCK_DISTRIBUTIONS,
    PLANCK_DISTRIBUTIONS,
)


__all__ = [
    'FLUX_UNITS',
    'SPECTRAL_UNITS',
    'AREA_UNITS',
    'AREA_FACTORS',
    'RADIATION_CONSTANTS',
    'STEFAN_BOLTZMANN_CONSTANTS',
    'WIEN_CONSTANTS',
    'INTEGRATED_PLANCK_DISTRIBUTIONS',
    'PLANCK_DISTRIBUTIONS',
    'spectral_radiant_sterance',
    'spectral_photon_sterance',
    'integrated_radiant_sterance',
    'integrated_photon_sterance',
]


def check_arguments_spectral(fn):
    @wraps(fn)
    def wrapper(T, x, *, spectral_unit, area_unit):
        if spectral_unit not in SPECTRAL_UNITS:
            raise ValueError(f"`spectral_unit` must be one of {repr(SPECTRAL_UNITS)}")

        if area_unit not in AREA_UNITS:
            raise ValueError(f"`area_unit` must be one of {repr(AREA_UNITS)}")

        T = np.atleast_1d(T).astype(np.float64, casting='safe')
        x = np.atleast_1d(x).astype(np.float64, casting='safe')

        if not np.all(T > 0):
            raise ValueError("`T` must be greater than zero")

        if not np.all(x > 0):
            raise ValueError("`x` must be greater than zero")

        return fn(T, x, spectral_unit=spectral_unit, area_unit=area_unit)
    return wrapper


@check_arguments_spectral
def spectral_radiant_sterance(T, x, *, spectral_unit, area_unit):
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
    (c1, c2) = RADIATION_CONSTANTS[('energy', spectral_unit)]

    _planck_distribution = PLANCK_DISTRIBUTIONS[('energy', spectral_unit)]

    return _planck_distribution(c1, c2, T, x)*AREA_FACTORS[area_unit]


@check_arguments_spectral
def spectral_photon_sterance(T, x, *, spectral_unit, area_unit):
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
    (c1, c2) = RADIATION_CONSTANTS[('photon', spectral_unit)]

    _planck_distribution = PLANCK_DISTRIBUTIONS[('photon', spectral_unit)]

    return _planck_distribution(c1, c2, T, x)*AREA_FACTORS[area_unit]


def check_arguments_integrated(fn):
    @wraps(fn)
    def wrapper(T, x_ab, *, spectral_unit, area_unit):
        if spectral_unit not in SPECTRAL_UNITS:
            raise ValueError(f"`spectral_unit` must be one of {repr(SPECTRAL_UNITS)}")

        if area_unit not in AREA_UNITS:
            raise ValueError(f"`area_unit` must be one of {repr(AREA_UNITS)}")

        T = np.atleast_1d(T).astype(np.float64, casting='safe')
        x_ab = np.atleast_2d(x_ab)

        if not np.all(T > 0):
            raise ValueError("`T` must be greater than zero")

        return fn(T, x_ab, spectral_unit=spectral_unit, area_unit=area_unit)
    return wrapper


@check_arguments_integrated
def integrated_radiant_sterance(T, x_ab, *, spectral_unit, area_unit):
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
    (c1, c2) = RADIATION_CONSTANTS[('energy', spectral_unit)]

    _integrated_planck_distribution = INTEGRATED_PLANCK_DISTRIBUTIONS[('energy', spectral_unit)]

    i1 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 0])
    i2 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 1])

    return np.abs(i2-i1)*AREA_FACTORS[area_unit]


@check_arguments_integrated
def integrated_photon_sterance(T, x_ab, *, spectral_unit, area_unit):
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
    (c1, c2) = RADIATION_CONSTANTS[('photon', spectral_unit)]

    _integrated_planck_distribution = INTEGRATED_PLANCK_DISTRIBUTIONS[('photon', spectral_unit)]

    i1 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 0])
    i2 = _integrated_planck_distribution(c1, c2, T, x_ab[..., 1])

    return np.abs(i2-i1)*AREA_FACTORS[area_unit]
