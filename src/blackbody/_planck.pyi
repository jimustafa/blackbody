from typing import Callable

from numpy import float64
from numpy.typing import NDArray


PLANCK_DISTRIBUTIONS: dict[tuple[str, str], Callable[[float, float, NDArray[float64], NDArray[float64]], NDArray[float64]]]
INTEGRATED_PLANCK_DISTRIBUTIONS: dict[tuple[str, str], Callable[[float, float, NDArray[float64], NDArray[float64]], NDArray[float64]]]
