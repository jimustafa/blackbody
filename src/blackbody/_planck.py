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


def _planck_integral_2(x, N):
    x = np.atleast_1d(x)
    n = np.arange(N)+1

    ex = np.exp(-n*x[...,np.newaxis])
    xx = np.array([x**2, 2*x, 2*np.ones_like(x)])
    nx = np.array([1/n, 1/n**2, 1/n**3])

    return np.einsum('...n,i...,in->...', ex, xx, nx)


def _planck_integral_3(x, N):
    x = np.atleast_1d(x)
    n = np.arange(N)+1

    ex = np.exp(-n*x[...,np.newaxis])
    xx = np.array([x**3, 3*x**2, 6*x, 6*np.ones_like(x)])
    nx = np.array([1/n, 1/n**2, 1/n**3, 1/n**4])

    return np.einsum('...n,i...,in->...', ex, xx, nx)


def _integrated_radiant_sterance_nu(c1, c2, T, nu_ab, Nterms):
    nu_ab = np.atleast_2d(nu_ab)

    x1 = c2*nu_ab[..., 0]/T
    x2 = c2*nu_ab[..., 1]/T

    i1 = _planck_integral_3(x1, Nterms)
    i2 = _planck_integral_3(x2, Nterms)

    return c1/c2**4*T**4*(i1-i2)


def _integrated_photon_sterance_nu(c1, c2, T, nu_ab, Nterms):
    nu_ab = np.atleast_2d(nu_ab)

    x1 = c2*nu_ab[..., 0]/T
    x2 = c2*nu_ab[..., 1]/T

    i1 = _planck_integral_2(x1, Nterms)
    i2 = _planck_integral_2(x2, Nterms)

    return c1/c2**3*T**3*(i1-i2)


def _integrated_radiant_sterance_lambda(c1, c2, T, xlambda_ab, Nterms):
    xlambda_ab = np.atleast_2d(xlambda_ab)

    x1 = c2/xlambda_ab[..., 0]/T
    x2 = c2/xlambda_ab[..., 1]/T

    i1 = _planck_integral_3(x1, Nterms)
    i2 = _planck_integral_3(x2, Nterms)

    return c1/c2**4*T**4*(i2-i1)


def _integrated_photon_sterance_lambda(c1, c2, T, xlambda_ab, Nterms):
    xlambda_ab = np.atleast_2d(xlambda_ab)

    x1 = c2/xlambda_ab[..., 0]/T
    x2 = c2/xlambda_ab[..., 1]/T

    i1 = _planck_integral_2(x1, Nterms)
    i2 = _planck_integral_2(x2, Nterms)

    return c1/c2**3*T**3*(i2-i1)


def _integrated_radiant_sterance_sigma(c1, c2, T, sigma_ab, Nterms):
    sigma_ab = np.atleast_2d(sigma_ab)

    x1 = c2*sigma_ab[..., 0]/T
    x2 = c2*sigma_ab[..., 1]/T

    i1 = _planck_integral_3(x1, Nterms)
    i2 = _planck_integral_3(x2, Nterms)

    return c1/c2**4*T**4*(i1-i2)


def _integrated_photon_sterance_sigma(c1, c2, T, sigma_ab, Nterms):
    sigma_ab = np.atleast_2d(sigma_ab)

    x1 = c2*sigma_ab[..., 0]/T
    x2 = c2*sigma_ab[..., 1]/T

    i1 = _planck_integral_2(x1, Nterms)
    i2 = _planck_integral_2(x2, Nterms)

    return c1/c2**3*T**3*(i1-i2)


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
