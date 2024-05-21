CLB_PATH = '/home/ivan/Study/Astro/solar/codes/CoronalLoopBuilder'

import sys
sys.path.insert(1, CLB_PATH)
# noinspection PyUnresolvedReferences
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d
import matplotlib.pyplot as plt
from utils.lso_align import calc_vect, synthmap_plot
from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a
from aiapy.calibrate import normalize_exposure, register, update_pointing
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u

from astropy.time import Time, TimeDelta

import pickle



#%%
# load calibrated aia full disk map
aia_imag = './aia_data/aia.lev1.5_euv_12s.2011-03-07T180624.341.fits'
#'./aia_data/aia.lev1_euv_12s.2011-03-07T180625Z.171.image_lev1.fits' # 'aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits'
aia_map = Map(aia_imag)

#%%
# print(type(aia_map))
# m_updated_pointing = update_pointing(aia_map)
# m_registered = register(m_updated_pointing)
# cal_map = normalize_exposure(m_registered)
# cal_map.save('./aia_data/aia.lev1.5_euv_12s.'+str(cal_map.date)+'.fits')
#%%

stereo_imag = './stereo_secchi/20110307_180600_n5euB.fts'
stereo_map = Map(stereo_imag)

#Crop maps
asec2cm = 0.725e8  # converting from arcsec to cm
roi_bottom_left = SkyCoord(Tx=850*u.arcsec, Ty=50*u.arcsec, frame=stereo_map.coordinate_frame)
width_Mm = (np.abs(-500+150)*asec2cm*u.cm).to(u.Mm).value
roi_top_right = SkyCoord(Tx=1050*u.arcsec, Ty=350*u.arcsec, frame=stereo_map.coordinate_frame)
height_Mm = (np.abs(500-150)*asec2cm*u.cm).to(u.Mm).value

#cusp_submap = xrt_map.submap(roi_bottom_left, top_right=roi_top_right)
stereo_cusp_submap = stereo_map.submap(roi_bottom_left, top_right=roi_top_right)

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=200*u.arcsec, frame=aia_map.coordinate_frame)
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=550*u.arcsec, frame=aia_map.coordinate_frame)
aia_cusp_submap = aia_map.submap(bottom_left=roi_bottom_left, top_right=roi_top_right)

#%%
'''
Plotting SDO and STEREO views and run CLB to infer loop parameters
'''
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection=aia_map)
aia_map.plot(axes=ax1)
ax2 = fig.add_subplot(122, projection=stereo_map)
stereo_map.plot(axes=ax2)
#synth_axs, synth_map = synthmap_plot(MAP_PATH, PARAMS_PATH, fig, plot="synth", instr='aia', channel=171)

param_path = './loop_params/loop1_params_20110307.pkl'
param_file = open(param_path, 'rb')
loop_params = pickle.load(param_file)

coronal_loop1 = CoronalLoopBuilder(fig, [ax1, ax2], [aia_map, stereo_map])#, pkl=param_path)
#coronal_loop1.set_loop(**loop_params)

plt.show()
#coronal_loop1.save_params_to_pickle('loop1_params_20110307.pkl')
'''
radius = 48.0 * u.Mm, height = 0.0 * u.Mm, phi0 = 340.50 * u.deg, theta0 = 11.43 * u.deg, el = 127.26 * u.deg, az = 113.40 * u.deg, samples_num = 130
'''
# plt.show()
# plt.close()

# coronal_loop1.save_params_to_pickle('2012/front_2012.pkl')
# coronal_loop1.save_params_to_pickle('2012/front_2012_testing.pkl')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate vectors to infer the placement of the MHD datacube
params_path = './loop_params/loop1_params_20110307.pkl'
norm, north, lat, lon = calc_vect(pkl='./loop_params/loop1_params_20110307.pkl')

mhd_datacube = '/media/ivan/TOSHIBA EXT/subs/subs_3_flarecs-id_0050.h5'
synth_map = synthmap_plot(params_path, map_path=None, smap=aia_cusp_submap,
                          fig=None, plot=None, datacube=mhd_datacube)
#%%
fig = plt.figure()
# Add the main axes. Note this is resized to leave room for the slider axes
ax = fig.add_axes([0.1, 0.2, 0.9, 0.7], projection=synth_map)
synth_map.plot(axes=ax)
plt.show()