# Set proper matplotlib backend to enable interactive plots in PyCharm (qt5agg)
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

import numpy as np
import os, sys
import pickle

import sunpy.map
import stackplotX as stp
import matplotlib.colors as colors

import yt
sys.path.insert(0, '/home/ivan/Study/Astro/solar')

'''
Export sunpy_map generic datacube containing info about current density 
to plot time-distance diagram of current density overlaid over aia image
 that can be used to identify relative locations of y-point to the tip of the EUV cusp (in AIA 131) 
'''

if __name__ == '__main__':

    """
    Loading pre-processed j_z projections from original datacube saved as a list of sunpy maps
    """
    file = open("cur_dens_map_list.pkl", 'rb')
    cur_dens_list = pickle.load(file)
    cur_dens_box = sunpy.map.Map(cur_dens_list, sequence=True)

    binpix = 2
    res_maps = []

    st = stp.Stackplot(cur_dens_box)

    # %%
    # st.mapseq_resample(binpix=2)

    st.mapseq_mkdiff(mode='dtrend', dt=12.)
    # # st.mapseq_mkdiff(mode='rdiff', dt=12.)
    # st.plot_mapseq(diff=True, norm=colors.LogNorm(vmin=1e0, vmax=8e1))
    st.plot_mapseq(diff=False, norm=colors.LogNorm(vmin=1e4, vmax=8e4))

#%%
    slit_file = 'td_slit_synth_vertical_jz.pickle'
    st.cutslit_fromfile(infile=slit_file)
    # slit_file = './td_slit_synth_horizontal.pickle'
    # if os.path.isfile(slit_file):
    #     st.cutslit_fromfile(infile=slit_file)
    # else:
    #     st.cutslit_tofile(outfile='td_slit_synth_vertical_jz.pickle')  #'td_slit_synth_horizontal.pickle')
    #
    # st.plot_stackplot(cmap='Greys', uni_cm=False)  # norm=colors.Normalize(vmin=-74, vmax=74),
#%%
    st.plot_stackplot(cmap='Greys', uni_cm=False, norm=colors.LogNorm(vmin=1e4, vmax=8e4))