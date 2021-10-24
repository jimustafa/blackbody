import itertools

import blackbody as bb

import pytest


@pytest.mark.parametrize(
    'flux_unit,spectral_unit',
    itertools.product(*[bb.FLUX_UNITS, bb.SPECTRAL_UNITS])
)
def test_keys(flux_unit, spectral_unit):
    assert (flux_unit, spectral_unit) in bb.RADIATION_CONSTANTS
    assert (flux_unit, spectral_unit) in bb.STEFAN_BOLTZMANN_CONSTANTS
    assert (flux_unit, spectral_unit) in bb.WIEN_CONSTANTS


@pytest.mark.parametrize(
    'area_unit',
    bb.AREA_UNITS
)
def test_area_factors_keys(area_unit):
    assert area_unit in bb.AREA_FACTORS
