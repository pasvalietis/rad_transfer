# Base imports
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Units
import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map

# Import synthetic image manipulation tools
import yt
from utils.proj_imag import SyntheticFilterImage as synt_img
from emission_models import uv, xrt

aia_imag_lvl1 = sunpy.map.Map('/home/saber/CoronalLoopBuilder/examples/testing/downloaded_events/aia_lev1_193a_2012_07_19t06_40_08_90z_image_lev1.fits')

downs_file_path = '/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path)  # , hint='AthenaDataset', units_override=units_override)
cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

instr = 'aia'  # aia or xrt
channel = 131
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

#~~~~~~~~~~~~~~~~~~~~~~~~
cusp_submap = aia_imag_lvl1
#~~~~~~~~~~~~~~~~~~~~~~~~

aia_synthetic = synt_img(cut_box, instr, channel)
# Match parameters of the synthetic image to observed one
# samp_resolution = cusp_submap.data.shape[0]
samp_resolution = 512
obs_scale = [cusp_submap.scale.axis1, cusp_submap.scale.axis2]*(u.arcsec/u.pixel)
reference_pixel = u.Quantity([cusp_submap.reference_pixel[0].value,
                              cusp_submap.reference_pixel[1].value], u.pixel)
reference_coord = cusp_submap.reference_coordinate

img_tilt = -23*u.deg

synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector': [0.12, 0.05, 0.916],    # Line of sight
                       'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}

aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                            view_settings=synth_view_settings,
                            image_shift=[-52, 105],
                            bkg_fill=10)

#Import scale from an AIA image:
synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28',
                                             observer='earth',
                                             detector='Synthetic AIA',
                                             scale=obs_scale,
                                             reference_coord=reference_coord,
                                             reference_pixel=reference_pixel)  # .rotate(angle=0.0 * u.deg)

synth_map.plot()
plt.savefig('figures/default.jpg', dpi=250, orientation='landscape')
plt.show()
plt.close()

