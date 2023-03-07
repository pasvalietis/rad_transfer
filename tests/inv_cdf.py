"""
This program tests the Inverse CDF method for different functions

For the reference see:
https://dk81.github.io/dkmathstats_site/prob-inverse-cdf.html
"""

# Draw samples from the distribution
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
#%%
rng = RandomState()
#s = rng.poisson(5, 10000)
s = rng.poisson(lam=100, size=500)

#%%
# Generate photon sampling with a power-law distribution function
nphot = 1000
alpha = 2.  # power-law index
emin = 2.  # in keV
emax = 700.
norm_fac = emax ** (1.0 - alpha) - emin ** (1.0 - alpha)

ems = np.zeros(nphot)

u = rng.uniform(size=nphot)

ems = (u*norm_fac + emin**(1. - alpha))
ems **= 1. / (1. - alpha)

#%%
bins = 10**(np.linspace(-1.,3.,200))
count, bins, ignored = plt.hist(ems, density=True, log=True, bins=bins)
plt.xscale('log')
plt.show()

#%%
"""
Thermal bremsstrahlung
"""

