import yt
import os

from utils import proj_imag

import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map

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

aia_imag = './aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits'
aia_map = sunpy.map.Map(aia_imag)
#%%
instr = 'aia' #aia or xrt
channel = '131'

#%%
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

# proj_imag.make_filter_image(subs_ds, instr, channel,
#                             vmin=1, vmax=300, norm_vec=[0.09, 0.0, 1.0],
#                             north_vector=[0., 1., 0.], figpath=fname)
imag = proj_imag.make_filter_image(subs_ds, instr, channel, resolution=585,
                                    vmin=1, vmax=300, norm_vec=[0.2, 0.0, 1.0],
                                    north_vector=[0., 1., 0.], figpath=fname)


#%%
# directory = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes'
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f) and f.endswith('.h5') and filename.startswith('subs'):
#         print(filename)
#         subs_ds = yt.load(f, units_override=units_override, hint='AthenaDataset')
#         timestep = f[-7:-3]
#         fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep
#
#         proj_imag.make_filter_image(subs_ds, instr, channel,
#                                     vmin=1, vmax=300, norm_vec=[0.2, 0.0, 1.0],
#                                     north_vector=[0., 1., 0.], figpath=fname)
#         #ds = yt.load(f, units_override=units_override, hint='AthenaDataset')
#         #rad_buffer_obj = downsample(ds, rad_fields=True, n=downs_factor)
