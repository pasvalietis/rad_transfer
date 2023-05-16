import matplotlib.pyplot as plt
import yt
import numpy as np
import sunpy
import pickle

import astropy.units as u
from aiapy.response import Channel
from scipy.io import readsav
from scipy import interpolate

# c = Channel(131*u.angstrom)
# trin=readsav('sdo_aia/aia_tresp_en.dat')
outfile = 'sdo_aia/aia_temp_response.npy'
# np.save(outfile, aia_trm, allow_pickle=True, fix_imports=True)

aia_trm = np.load(outfile, allow_pickle=True)

#%%
ch_ = 1 #131A
aia_trm_interpf = interpolate.interp1d(
    aia_trm.item()['logt'],
    aia_trm.item()['temp_response'][:,ch_],
    fill_value="extrapolate",
    kind='cubic',
)

#%%
import matplotlib.pyplot as plt
temp_x = np.logspace(5,8,290)
tlog = np.log10(temp_x)
plt.loglog(tlog, aia_trm_interpf(tlog))
plt.show()

#%%
'''
Sample 3d np.array of temperatures
Try to pass it into a 3d interpolated function
'''

temps = 10**(5 + np.random.rand(3,3,3))
