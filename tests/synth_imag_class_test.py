# Provide pixel coords to obtain brightness profiles
# Convert synthetic image to real physical coordinates (Mm) and similarly the AIA image and plot brightness profiles on the same scale

import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as colors
import matplotlib
import scipy.ndimage
import astropy.units as u
from astropy.coordinates import SkyCoord

import scipy.ndimage
import scipy.misc
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import math
from visualization.colormaps import color_tables

import sunpy.map
import sunpy.visualization.colormaps as cm
#from sunpy.data.sample import AIA_131_IMAGE

import yt
import os, sys

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

'''
Import dataset
'''

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path, units_override=units_override, hint='AthenaDataset')

#%%
instr = 'aia'  # aia or xrt
channel = '131'
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

aia_synthetic = synt_img(subs_ds, instr, channel)
samp_resolution = 585
synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector': [0.25, 0.0, 0.866],
                       'north_vector': [0.0, 1.0, 0.0]}

aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings, view_settings=synth_view_settings)

#%%

synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28', observer='earth')
