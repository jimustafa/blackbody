import itertools

import pytest

import blackbody as bb


@pytest.mark.parametrize(
    "flux_unit,spectral_unit", itertools.product(bb.FLUX_UNITS, bb.SPECTRAL_UNITS)
)
def test_units_keys(flux_unit, spectral_unit):
    assert (flux_unit, spectral_unit) in bb.PLANCK_SPECTRA
    assert (flux_unit, spectral_unit) in bb.PLANCK_INTEGRALS
    assert (flux_unit, spectral_unit) in bb.RADIATION_CONSTANTS
    assert (flux_unit, spectral_unit) in bb.WIEN_CONSTANTS


@pytest.mark.parametrize("flux_unit", bb.FLUX_UNITS)
def test_stefan_boltzmann_keys(flux_unit):
    assert flux_unit in bb.STEFAN_BOLTZMANN_CONSTANTS


@pytest.mark.parametrize("area_unit", bb.AREA_UNITS)
def test_area_factors_keys(area_unit):
    assert area_unit in bb.AREA_FACTORS
