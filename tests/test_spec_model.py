#import yt
import numpy as np
from soxs.thermal_spectra import CIEGenerator
import matplotlib.pyplot as plt
from pyxsim.spectral_models import SpectralInterpolator1D

'''
Initial parameters of the spectral model /see thermal_sources.py/
'''
model = "apec"
emin = 6.0
emax = 12.0
nbins = 100
binscale = "linear"  # or "log"
var_elem = None
model_root = None
model_vers = None
thermal_broad = True
nolines = False
abund_table = "angr"
nei = False  # don't use the non-equilibrium ionization tables
#***
kT_min = 0.025
kT_max = 64.0

cgen = CIEGenerator(
            model,
            emin,
            emax,
            nbins,
            binscale=binscale,
            var_elem=var_elem,
            model_root=model_root,
            model_vers=model_vers,
            broadening=thermal_broad,
            nolines=nolines,
            abund_table=abund_table,
            nei=nei,
        )

zobs = 0.0

idx_min = max(np.searchsorted(cgen.Tvals, kT_min) - 1, 0)
idx_max = min(np.searchsorted(cgen.Tvals, kT_max) + 1, cgen.nT - 1)

Tvals = cgen.Tvals[idx_min : idx_max]

cosmic_spec, metal_spec, var_spec = cgen._get_table(
            list(range(idx_min, idx_max)), zobs, 0.0
        )

si = SpectralInterpolator1D(
            Tvals, cosmic_spec, metal_spec, var_spec
        )


kTi = 10**(5 + 3*np.random.rand(10)) #kT[ibegin:iend]
cspec, mspec, vspec = si(kTi)

#kTi

'''
Thermal emission spectrum returned by get_spectrum function
'''

tot_spec = cspec
