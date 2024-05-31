# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import os, sys, re
# import pickle

import numpy as np
# import astropy.units as u
# from astropy.coordinates import SkyCoord
# from astropy.time import Time, TimeDelta

import sunpy.map
# from sunpy.net import Fido
# from sunpy.net import attrs as a
# import warnings
# warnings.filterwarnings("ignore")
# sys.path.insert(0, '/home/ivan/Study/Astro/solar')
import matplotlib.colors as colors

sys.path.insert(0,'/home/saber/rad_transfer/tests/sab_tests/spx')
import stackplotX as stp

obs_lvl15_flare_data = '/media/saber/My_Passport/Sabastian/xrt/2013_05_15/aia_lvl_15_ROI/'

print('\nLoading mapsequence...')
mapsequence = sunpy.map.Map(obs_lvl15_flare_data + '*.fits', sequence=True)
print('Done. \n')

print('Loading stackplotX...')
st = stp.Stackplot(mapsequence)
print('Done. \n')

print('Resampling mapsequence...')
binpix = 2
res_maps = []
st.mapseq_resample(binpix=binpix)
print('Done. \n')

# print('Plotting mapsequence...')
# st.mapseq_mkdiff(mode='rratio', dt=60.)
# st.plot_mapseq(diff=True, norm=colors.LogNorm())
# print('Done. \n')
# plt.show()
# plt.close()

print('Loading slit file...')
slit_file = 'slit_file_1.pkl'
if os.path.isfile(slit_file):
    st.cutslit_fromfile(infile=slit_file)
else:
    print('No slit file found! Creating new one...')
    st.cutslit_tofile(outfile=slit_file)
print('Done. \n')

print('Plotting stackplot...')
st.plot_stackplot(cmap='Greys', uni_cm=False)
print('Done. \n')

print("Execution complete.\n")
plt.show()
plt.close()