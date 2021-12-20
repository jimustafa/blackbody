import itertools

import numpy as np
import scipy.constants
import scipy.integrate
import scipy.optimize

import blackbody as bb

import pytest


@pytest.fixture(scope='module')
def T():
    return 500


@pytest.mark.parametrize(
    'flux_unit,spectral_unit,method',
    itertools.product(*[bb.FLUX_UNITS, bb.SPECTRAL_UNITS, ['scipy.integrate.quad', 'INTEGRATED_PLANCK_DISTRIBUTIONS']])
)
def test_Stefan_Boltzmann(T, flux_unit, spectral_unit, method):
    if method == 'scipy.integrate.quad':
        if spectral_unit == 'Hz':
            pytest.skip()
        if flux_unit == 'energy':
            L = lambda x: bb.spectral_radiant_sterance(T, x, spectral_unit=spectral_unit, area_unit='m^2')
        if flux_unit == 'photon':
            L = lambda x: bb.spectral_photon_sterance(T, x, spectral_unit=spectral_unit, area_unit='m^2')
        L_integral = scipy.integrate.quad(L, 0, np.inf)[0]

    if method == 'INTEGRATED_PLANCK_DISTRIBUTIONS':
        (c1, c2) = bb.RADIATION_CONSTANTS[(flux_unit, spectral_unit)]

        _integrated_planck_distribution = bb.INTEGRATED_PLANCK_DISTRIBUTIONS[(flux_unit, spectral_unit)]

        if spectral_unit == 'um':
            L_integral = _integrated_planck_distribution(c1, c2, T, 1e12)
        else:
            L_integral = _integrated_planck_distribution(c1, c2, T, 0)

    sigma = bb.STEFAN_BOLTZMANN_CONSTANTS[(flux_unit, spectral_unit)]

    if flux_unit == 'energy':
        assert np.allclose(L_integral/T**4, sigma/np.pi, atol=0, rtol=1e-6)
    if flux_unit == 'photon':
        assert np.allclose(L_integral/T**3, sigma/np.pi, atol=0, rtol=1e-6)


@pytest.mark.parametrize(
    'flux_unit,spectral_unit',
    itertools.product(*[bb.FLUX_UNITS, bb.SPECTRAL_UNITS])
)
def test_Wien(T, flux_unit, spectral_unit):
    if spectral_unit in ['Hz', 'THz', 'cm^-1']:
        x_ref = bb.WIEN_CONSTANTS[(flux_unit, spectral_unit)]*T
    if spectral_unit == 'um':
        x_ref = bb.WIEN_CONSTANTS[(flux_unit, spectral_unit)]/T

    if flux_unit == 'energy':
        L = lambda x: bb.spectral_radiant_sterance(T, x, spectral_unit=spectral_unit, area_unit='m^2')
    if flux_unit == 'photon':
        L = lambda x: bb.spectral_photon_sterance(T, x, spectral_unit=spectral_unit, area_unit='m^2')

    assert np.logical_and(L(x_ref) > L(x_ref*0.99), L(x_ref) > L(x_ref*1.01))
