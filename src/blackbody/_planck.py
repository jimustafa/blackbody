import numpy as np

from ._constants import EMAXEXP


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


PLANCK_DISTRIBUTIONS = {
    ('energy', 'Hz'): _spectral_radiant_sterance_nu,
    ('photon', 'Hz'): _spectral_photon_sterance_nu,
    ('energy', 'um'): _spectral_radiant_sterance_lambda,
    ('photon', 'um'): _spectral_photon_sterance_lambda,
    ('energy', 'cm^-1'): _spectral_radiant_sterance_sigma,
    ('photon', 'cm^-1'): _spectral_photon_sterance_sigma,
}
