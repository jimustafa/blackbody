import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import blackbody as bb


mpl.style.use([
    pathlib.Path(__file__).parent / '../mplstyle',
])


(fig, ax1) = plt.subplots()
ax2 = ax1.twinx()

x1 = np.linspace(0, 18, 37)
x2 = np.linspace(18, 35, 35)

ax1.plot(x1, bb._planck._planck_integral_3(x1, 100))
ax2.plot(x2, bb._planck._planck_integral_3(x2, 100))

ax1.axvline(18, color='black', linestyle='dashed')

ax1.set_xlabel('$x$')
ax1.set_ylabel(R'$\int_x^\infty \mathrm{d}x \frac{x^3}{e^x-1}$')
ax2.set_ylabel(R'$\int_x^\infty \mathrm{d}x \frac{x^3}{e^x-1}$')

ax1.set_xlim([0, 35])
ax1.set_yscale('log')
ax1.set_ylim([1e-4, 1e1])
ax1.grid(True)

ax2.set_yscale('log')
ax2.set_ylim([1e-9, 1e-4])

fig.savefig('Widger–Woodall_Fig2.png')


sigma_ab = np.array([
    [2380, 2940],
    [1400, 1800],
    [910, 1000],
    [625, 720],
    [400, 500],
])

Tx = np.linspace(200, 300, 21)

(c1, c2) = bb.RADIATION_CONSTANTS[('energy', 'cm^-1')]

Le = 1e-4*bb._planck._integrated_radiant_sterance_sigma(c1, c2, Tx[..., np.newaxis], sigma_ab, 100)

(fig, ax) = plt.subplots()

ax.plot(Tx, Le[:, 0])
ax.plot(Tx, Le[:, 1])
ax.plot(Tx, Le[:, 2])
ax.plot(Tx, Le[:, 3])
ax.plot(Tx, Le[:, 4])

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Radiant Sterance (W cm$^{-2}$ sr$^{-1}$)')

ax.set_yscale('log')
ax.set_ylim([1e-7, 1e-2])
ax.grid(True)

fig.savefig('Widger–Woodall_Fig3.png')
