import itertools

import numpy as np
from scipy.constants import c, h, k, sigma
from scipy.special import lambertw, zeta


EMAXEXP = np.log(2)*np.finfo(np.float64).maxexp

FLUX_UNITS = [
    'energy',
    'photon',
]

SPECTRAL_UNITS = [
    'Hz',
    'THz',
    'um',
    'cm^-1',
]

AREA_UNITS = [
    'm^2',
    'cm^2',
]

AREA_FACTORS = {
    'm^2': 1,
    'cm^2': 1/1e4,
}

STEFAN_BOLTZMANN_CONSTANTS = {
    ('energy' , 'Hz'   ): sigma,
    ('photon' , 'Hz'   ): 4*np.pi*zeta(3)*k**3/h**3/c**2,
    ('energy' , 'THz'  ): sigma,
    ('photon' , 'THz'  ): 4*np.pi*zeta(3)*k**3/h**3/c**2,
    ('energy' , 'um'   ): sigma,
    ('photon' , 'um'   ): 4*np.pi*zeta(3)*k**3/h**3/c**2,
    ('energy' , 'cm^-1'): sigma,
    ('photon' , 'cm^-1'): 4*np.pi*zeta(3)*k**3/h**3/c**2,
}

WIEN_CONSTANTS = {
    ('energy' , 'Hz'   ): k/h*abs((3+lambertw(-3*np.exp(-3), 0))),
    ('photon' , 'Hz'   ): k/h*abs((2+lambertw(-2*np.exp(-2), 0))),
    ('energy' , 'THz'  ): 1/1e12*k/h*abs((3+lambertw(-3*np.exp(-3), 0))),
    ('photon' , 'THz'  ): 1/1e12*k/h*abs((2+lambertw(-2*np.exp(-2), 0))),
    ('energy' , 'um'   ): 1e6*h*c/k/abs((5+lambertw(-5*np.exp(-5), 0))),
    ('photon' , 'um'   ): 1e6*h*c/k/abs((4+lambertw(-4*np.exp(-4), 0))),
    ('energy' , 'cm^-1'): k/(100*h*c)*abs((3+lambertw(-3*np.exp(-3), 0))),
    ('photon' , 'cm^-1'): k/(100*h*c)*abs((2+lambertw(-2*np.exp(-2), 0))),
}

RADIATION_CONSTANTS = {
    ('energy' , 'Hz'   ): (2*h/c**2      , h  /k     ),
    ('photon' , 'Hz'   ): (2  /c**2      , h  /k     ),
    ('energy' , 'THz'  ): (2*h/c**2*1e48 , h  /k*1e12),
    ('photon' , 'THz'  ): (2  /c**2*1e36 , h  /k*1e12),
    ('energy' , 'um'   ): (2*h*c**2*1e24 , h*c/k*1e6 ),
    ('photon' , 'um'   ): (2  *c   *1e18 , h*c/k*1e6 ),
    ('energy' , 'cm^-1'): (2*h*c**2*1e8  , h*c/k*100 ),
    ('photon' , 'cm^-1'): (2  *c   *1e6  , h*c/k*100 ),
}
