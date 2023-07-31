
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
sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

#%%
'''
Import AIA image
'''

aia_imag = './aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits'
aia_map = sunpy.map.Map(aia_imag, autoalign=True)
aia_rotated = aia_map #.rotate(angle=0.0 * u.deg)

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=150*u.arcsec, frame=aia_rotated.coordinate_frame)
#width_Mm = (np.abs(-500+150)*asec2cm*u.cm).to(u.Mm).value
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=500*u.arcsec, frame=aia_rotated.coordinate_frame)
#height_Mm =  (np.abs(500-150)*asec2cm*u.cm).to(u.Mm).value
cusp_submap = aia_rotated.submap(roi_bottom_left, top_right=roi_top_right)

#%%
'''
Import dataset
'''

# L_0 = (1.5e8, "m")
# units_override = {
#     "length_unit": L_0,
#     "time_unit": (109.8, "s"),
#     "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
#     "velocity_unit": (1.366e6, "m/s"),
#     "temperature_unit": (1.13e8, "K"),
# }

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path)  # , hint='AthenaDataset', units_override=units_override)
cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
instr = 'aia'  # aia or xrt
channel = '131'
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

aia_synthetic = synt_img(cut_box, instr, channel)
# Match parameters of the synthetic image to observed one
samp_resolution = cusp_submap.data.shape[0]
obs_scale = [cusp_submap.scale.axis1, cusp_submap.scale.axis2]*(u.arcsec/u.pixel)
reference_pixel = u.Quantity([cusp_submap.reference_pixel[0].value,
                              cusp_submap.reference_pixel[1].value], u.pixel)
reference_coord = cusp_submap.reference_coordinate

img_tilt = -23*u.deg

synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector': [0.08, 0.08, 0.916],
                       'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}

aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                            view_settings=synth_view_settings,
                            image_shift=[-65, 200])
#Import scale from an AIA image:
synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28',
                                             observer='earth',
                                             scale=obs_scale,
                                             reference_coord=reference_coord,
                                             reference_pixel=reference_pixel) # .rotate(angle=0.0 * u.deg)

#%%
#  Update physical scaling (only arcsec/pix right now)
#synth_map.scale._replace(axis1=cusp_submap.scale.axis1, axis2=cusp_submap.scale.axis2)
#%%
'''
Plot obs and synthetic image
'''

from mpl_toolkits.axes_grid1 import make_axes_locatable

#fig = plt.figure(figsize=(10, 4))
fig = plt.figure(constrained_layout=True, figsize=(10, 6))
fig.set_tight_layout(True)
plt.ioff()
plt.style.use('fast')
gs = fig.add_gridspec(3, 4)

#ax1 = fig.add_subplot(212, projection=cusp_submap)
ax1 = fig.add_subplot(gs[:2, :2], projection=cusp_submap)
cusp_submap.plot_settings['norm'] = colors.LogNorm(10, cusp_submap.max())
#cusp_submap.plot_settings['cmap'] =
cusp_submap.plot(axes=ax1)
# plt.colorbar()  # (cax=cax, orientation='vertical')
#ax1.plot_coord(intensity_coords)
#ax1.plot_coord(line_coords[0], marker="o", color="blue", label="start")
#ax1.plot_coord(line_coords[1], marker="o", color="green", label="end")
#ax1.legend()

line_coords = SkyCoord([-310, -380], [320, 480], unit=(u.arcsec, u.arcsec),
                       frame=cusp_submap.coordinate_frame) # [x1, x2], [y1, y2]
intensity_coords = sunpy.map.pixelate_coord_path(cusp_submap, line_coords)
intensity = sunpy.map.sample_at_coords(cusp_submap, intensity_coords)

line_coords_ = SkyCoord([-310, -380], [320, 480], unit=(u.arcsec, u.arcsec),
                       frame=synth_map.coordinate_frame) # [x1, x2], [y1, y2]
intensity_coords_ = sunpy.map.pixelate_coord_path(synth_map, line_coords_)
intensity_ = sunpy.map.sample_at_coords(synth_map, intensity_coords_)

angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
angular_separation_ = intensity_coords_.separation(intensity_coords_[0]).to(u.arcsec)

ax1.plot_coord(intensity_coords)

#divider = make_axes_locatable(ax1)
#cax = divider.append_axes('right', size='5%', pad=0.05)

#ax2 = fig.add_subplot(221, projection=synth_map)
ax2 = fig.add_subplot(gs[:2, 2:], projection=synth_map)
#cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
synth_map.plot_settings['norm'] = colors.LogNorm(10, 10*synth_map.max())
synth_map.plot_settings['cmap'] = aia_map.plot_settings['cmap']
synth_map.plot(axes=ax2)
ax2.plot_coord(intensity_coords_)


plt.colorbar()

#ax3 = fig.add_subplot(222)
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(angular_separation, intensity)
ax3.plot(angular_separation_, intensity_)
ax3.set_xlabel("Angular distance along slit [arcsec]")
ax3.set_ylabel(f"Intensity [{cusp_submap.unit}]")
ax3.set_yscale('log')
#plt.tight_layout()
plt.savefig('slit_profiles.png')


# # # sdoaia131 = matplotlib.colormaps['sdoaia131']
# #
# fig = plt.figure(constrained_layout=True, figsize=(10, 6))
# plt.ioff()
# plt.style.use('fast')
# #
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# gs = fig.add_gridspec(2, 4)
#
# ax1 = fig.add_subplot(gs[:2, :2], projection=cusp_submap)
# #cusp_submap.plot(axes=ax1, clip_interval=(1, 99.99)*u.percent)
# stonyhurst_grid = cusp_submap.draw_grid(axes=ax1, system='stonyhurst')
#
# ax1.axis('image')
# ax1.set_title('AIA/SDO 131A 2011-03-07 18:09:09', pad=44)
# ax1.set_xlabel('X, pix')
# ax1.set_ylabel('Y, pix')
#
# ax2 = fig.add_subplot(gs[:2, 2:], projection=synth_map)
# #divider = make_axes_locatable(ax2)
# #cax = divider.append_axes('right', size='5%', pad=0.05)
# ax2.set_title('Synthetic image')
# ax2.set_xlabel('X, Pixel')
# ax2.set_ylabel('Y, Pixel')
#
# # plt.ioff()
# # plt.savefig('slit_profiles.png')
# # plt.close()