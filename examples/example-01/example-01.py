import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import blackbody as bb


mpl.style.use([
    pathlib.Path(__file__).parent / '../mplstyle',
])


(fig, [ax1, ax2]) = plt.subplots(nrows=2, ncols=1, figsize=(9,6), sharex=True)

nu = np.linspace(1e11, 8e13, 800)

for T in [100, 200, 300, 400, 500]:
    ax1.plot(nu, bb.spectral_radiant_sterance(T, nu, spectral_unit='Hz', area_unit='m^2'), label=f'$T={T}\,\mathrm{{K}}$')
    ax2.plot(nu, bb.spectral_photon_sterance(T, nu, spectral_unit='Hz', area_unit='m^2'), label=f'$T={T}\,\mathrm{{K}}$')

Tx = np.linspace(10, 600, 60)
nu_peak = bb.WIEN_CONSTANTS[('energy', 'Hz')]*Tx
ax1.plot(
    nu_peak,
    bb.spectral_radiant_sterance(Tx, nu_peak, spectral_unit='Hz', area_unit='m^2'),
    '.', color='black'
)
nu_peak = bb.WIEN_CONSTANTS[('photon', 'Hz')]*Tx
ax2.plot(
    nu_peak,
    bb.spectral_photon_sterance(Tx, nu_peak, spectral_unit='Hz', area_unit='m^2'),
    '.', color='black'
)

for ax in [ax1, ax2]:
    ax.legend()
    ax.grid(True)

ax2.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Spectral Sterance\n(W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)')
ax2.set_ylabel('Spectral Sterance\n(photons s$^{-1}$ m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)')
ax2.set_xlim([0, 8e13])
ax1.set_ylim([0, 2.5e-11])
ax2.set_ylim([0, 2e9])

fig.savefig('spectral-sterance_nu.png')


(fig, [ax1, ax2]) = plt.subplots(nrows=2, ncols=1, figsize=(9,6), sharex=True)

xlambda = np.linspace(0.2, 25, 125)

for T in [100, 200, 300, 400, 500]:
    ax1.plot(xlambda, bb.spectral_radiant_sterance(T, xlambda, spectral_unit='um', area_unit='m^2'), label=f'$T={T}\,\mathrm{{K}}$')
    ax2.plot(xlambda, bb.spectral_photon_sterance(T, xlambda, spectral_unit='um', area_unit='m^2'), label=f'$T={T}\,\mathrm{{K}}$')

Tx = np.linspace(10, 600, 60)
lambda_peak = bb.WIEN_CONSTANTS[('energy', 'um')]/Tx
ax1.plot(
    lambda_peak,
    bb.spectral_radiant_sterance(Tx, lambda_peak, spectral_unit='um', area_unit='m^2'),
    '.', color='black'
)
lambda_peak = bb.WIEN_CONSTANTS[('photon', 'um')]/Tx
ax2.plot(
    lambda_peak,
    bb.spectral_photon_sterance(Tx, lambda_peak, spectral_unit='um', area_unit='m^2'),
    '.', color='black'
)

for ax in [ax1, ax2]:
    ax.legend()
    ax.grid(True)

ax2.set_xlabel('Wavelength (µm)')
ax1.set_ylabel('Spectral Sterance\n(W m$^{-2}$ sr$^{-1}$ µm$^{-1}$)')
ax2.set_ylabel('Spectral Sterance\n(photons s$^{-1}$ m$^{-2}$ sr$^{-1}$ µm$^{-1}$)')
ax2.set_xlim([0, 25])
ax1.set_ylim([0, 50])
ax2.set_ylim([0, 2e21])

fig.savefig('spectral-sterance_lambda.png')


(fig, [ax1, ax2]) = plt.subplots(nrows=2, ncols=1, figsize=(9,6), sharex=True)

sigma = np.linspace(10, 2000, 200)

for T in [100, 200, 300, 400, 500]:
    ax1.plot(sigma, bb.spectral_radiant_sterance(T, sigma, spectral_unit='cm^-1', area_unit='m^2'), label=f'$T={T}\,\mathrm{{K}}$')
    ax2.plot(sigma, bb.spectral_photon_sterance(T, sigma, spectral_unit='cm^-1', area_unit='m^2'), label=f'$T={T}\,\mathrm{{K}}$')

Tx = np.linspace(10, 600, 60)
sigma_peak = bb.WIEN_CONSTANTS[('energy', 'cm^-1')]*Tx
ax1.plot(
    sigma_peak,
    bb.spectral_radiant_sterance(Tx, sigma_peak, spectral_unit='cm^-1', area_unit='m^2'),
    '.', color='black'
)
sigma_peak = bb.WIEN_CONSTANTS[('photon', 'cm^-1')]*Tx
ax2.plot(
    sigma_peak,
    bb.spectral_photon_sterance(Tx, sigma_peak, spectral_unit='cm^-1', area_unit='m^2'),
    '.', color='black'
)

for ax in [ax1, ax2]:
    ax.legend()
    ax.grid(True)

ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
ax1.set_ylabel('Spectral Sterance\n(W m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
ax2.set_ylabel('Spectral Sterance\n(photons s$^{-1}$ m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
ax2.set_xlim([0, 2000])
ax1.set_ylim([0, 0.8])
ax2.set_ylim([0, 5e19])

fig.savefig('spectral-sterance_sigma.png')
