from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

SpectralUnit: TypeAlias = Literal[
    "Hz",
    "THz",
    "um",
    "cm^-1",
]

AreaUnit: TypeAlias = Literal[
    "m^2",
    "cm^2",
]


class _SpectralFn(Protocol):
    def __call__(
        self,
        T: ArrayLike,
        x: ArrayLike,
        *,
        spectral_unit: SpectralUnit,
        area_unit: AreaUnit,
    ) -> NDArray[np.float64]: ...


class _IntegratedFn(Protocol):
    def __call__(
        self,
        T: ArrayLike,
        x_ab: ArrayLike,
        *,
        spectral_unit: SpectralUnit,
        area_unit: AreaUnit,
    ) -> NDArray[np.float64]: ...
