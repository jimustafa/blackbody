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
from ._planck import PLANCK_DISTRIBUTIONS


def check_arguments(fn):
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


@check_arguments
def spectral_radiant_sterance(T, x, *, spectral_unit, area_unit):
    (c1, c2) = RADIATION_CONSTANTS[('energy', spectral_unit)]

    _planck_distribution = PLANCK_DISTRIBUTIONS[('energy', spectral_unit)]

    return _planck_distribution(c1, c2, T, x)*AREA_FACTORS[area_unit]


@check_arguments
def spectral_photon_sterance(T, x, *, spectral_unit, area_unit):
    (c1, c2) = RADIATION_CONSTANTS[('photon', spectral_unit)]

    _planck_distribution = PLANCK_DISTRIBUTIONS[('photon', spectral_unit)]

    return _planck_distribution(c1, c2, T, x)*AREA_FACTORS[area_unit]
