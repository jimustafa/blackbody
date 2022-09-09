import numpy as np

from ._constants import EMAXEXP


__all__ = [
    'PLANCK_DISTRIBUTIONS',
    'INTEGRATED_PLANCK_DISTRIBUTIONS',
]


def _spectral_radiant_sterance_nu(c1, c2, T, nu):
    x = nu/T
    xmax = EMAXEXP/c2
    x[x>xmax] = xmax

    return c1*nu**3 * 1/(np.exp(c2*x)-1)


def _spectral_photon_sterance_nu(c1, c2, T, nu):
    x = nu/T
    xmax = EMAXEXP/c2
    x[x>xmax] = xmax

    return c1*nu**2 * 1/(np.exp(c2*x)-1)


def _spectral_radiant_sterance_lambda(c1, c2, T, xlambda):
    x = 1/(xlambda*T)
    xmax = EMAXEXP/c2
    x[x>xmax] = xmax

    return c1/xlambda**5 * 1/(np.exp(c2*x)-1)


def _spectral_photon_sterance_lambda(c1, c2, T, xlambda):
    x = 1/(xlambda*T)
    xmax = EMAXEXP/c2
    x[x>xmax] = xmax

    return c1/xlambda**4 * 1/(np.exp(c2*x)-1)


def _spectral_radiant_sterance_sigma(c1, c2, T, sigma):
    x = sigma/T
    xmax = EMAXEXP/c2
    x[x>xmax] = xmax

    return c1*sigma**3 * 1/(np.exp(c2*x)-1)


def _spectral_photon_sterance_sigma(c1, c2, T, sigma):
    x = sigma/T
    xmax = EMAXEXP/c2
    x[x>xmax] = xmax

    return c1*sigma**2 * 1/(np.exp(c2*x)-1)


def _planck_integral_2(x, N=1024):
    x = np.atleast_1d(x)
    n = np.arange(N)+1

    ex = np.exp(-n*x[..., np.newaxis])
    xx = np.array([x**2, 2*x, 2*np.ones_like(x)])
    nx = np.array([1/n, 1/n**2, 1/n**3])

    return np.einsum('...n,i...,in->...', ex, xx, nx)


def _planck_integral_3(x, N=1024):
    x = np.atleast_1d(x)
    n = np.arange(N)+1

    ex = np.exp(-n*x[..., np.newaxis])
    xx = np.array([x**3, 3*x**2, 6*x, 6*np.ones_like(x)])
    nx = np.array([1/n, 1/n**2, 1/n**3, 1/n**4])

    return np.einsum('...n,i...,in->...', ex, xx, nx)


def _integrated_radiant_sterance_nu(c1, c2, T, nu):
    return c1/c2**4*T**4*_planck_integral_3(c2*nu/T)


def _integrated_photon_sterance_nu(c1, c2, T, nu):
    return c1/c2**3*T**3*_planck_integral_2(c2*nu/T)


def _integrated_radiant_sterance_lambda(c1, c2, T, xlambda):
    return c1/c2**4*T**4*_planck_integral_3(c2/xlambda/T)


def _integrated_photon_sterance_lambda(c1, c2, T, xlambda):
    return c1/c2**3*T**3*_planck_integral_2(c2/xlambda/T)


def _integrated_radiant_sterance_sigma(c1, c2, T, sigma):
    return c1/c2**4*T**4*_planck_integral_3(c2*sigma/T)


def _integrated_photon_sterance_sigma(c1, c2, T, sigma):
    return c1/c2**3*T**3*_planck_integral_2(c2*sigma/T)


PLANCK_DISTRIBUTIONS = {
    ('energy', 'Hz'): _spectral_radiant_sterance_nu,
    ('photon', 'Hz'): _spectral_photon_sterance_nu,
    ('energy', 'THz'): _spectral_radiant_sterance_nu,
    ('photon', 'THz'): _spectral_photon_sterance_nu,
    ('energy', 'um'): _spectral_radiant_sterance_lambda,
    ('photon', 'um'): _spectral_photon_sterance_lambda,
    ('energy', 'cm^-1'): _spectral_radiant_sterance_sigma,
    ('photon', 'cm^-1'): _spectral_photon_sterance_sigma,
}


INTEGRATED_PLANCK_DISTRIBUTIONS = {
    ('energy', 'Hz'): _integrated_radiant_sterance_nu,
    ('photon', 'Hz'): _integrated_photon_sterance_nu,
    ('energy', 'THz'): _integrated_radiant_sterance_nu,
    ('photon', 'THz'): _integrated_photon_sterance_nu,
    ('energy', 'um'): _integrated_radiant_sterance_lambda,
    ('photon', 'um'): _integrated_photon_sterance_lambda,
    ('energy', 'cm^-1'): _integrated_radiant_sterance_sigma,
    ('photon', 'cm^-1'): _integrated_photon_sterance_sigma,
}
