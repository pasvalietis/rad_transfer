import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, sys, re
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import sunpy.map

import stackplotX as stp
import matplotlib.colors as colors

if __name__ == '__main__':
    print('Plotting TD plot with StackplotX')
    time_range = ['2011-03-07T14:03:00', '2011-03-07T14:13:00']
    tstp = Time(time_range, format='isot', scale='utc')
#%%
    st = stp.Stackplot()
#%%
    # st.make_mapseq(trange=tstp, outfile='mapseq.p', fov=None, wavelength='131', binpix=1, dt_data=12., derotate=False,
    #                 tosave=True, superpixel=False, aia_prep=True, mapinterp=False, overwrite=False, dtype=None,
    #                 normalize=True)

    obs_lvl15_flare_data = '/media/ivan/TOSHIBA EXT/aia_img/2011_event/flare_roi/calibrated/'
    mapsequence = sunpy.map.Map(obs_lvl15_flare_data + 'aia.lev1.5_euv_12s_roi.2011-03-07T*.fits', sequence=True)
    binpix = 2

    res_maps = []
    sequence = mapsequence  # sunpy.map.Map(res_maps, sequence=True)
    st = stp.Stackplot(sequence)

    st.mapseq_resample(binpix=2)
    st.mapseq_mkdiff(mode='rdiff', dt=12.)
    st.plot_mapseq(diff=True, norm=colors.LogNorm(vmin=1e1, vmax=8e2))

    #%%
    st.cutslitbd.update()
    st.plot_stackplot(norm=colors.LogNorm(vmin=1e-1, vmax=8e1), cmap='Greys', uni_cm=True)
