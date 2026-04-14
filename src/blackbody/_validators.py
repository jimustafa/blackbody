from functools import wraps

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._constants import AREA_UNITS, SPECTRAL_UNITS
from ._types import AreaUnit, SpectralUnit, _IntegratedFn, _SpectralFn


def check_arguments_spectral(fn: _SpectralFn) -> _SpectralFn:
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

    return wrapper  # type: ignore[return-value]


def check_arguments_integrated(fn: _IntegratedFn) -> _IntegratedFn:
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

    return wrapper  # type: ignore[return-value]
