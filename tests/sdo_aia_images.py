import matplotlib.pyplot as plt
import numpy as np
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


aia_imag = './aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits'
aia_map = sunpy.map.Map(aia_imag)

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=150*u.arcsec, frame=aia_map.coordinate_frame)
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=500*u.arcsec, frame=aia_map.coordinate_frame)
cusp_submap = aia_map.submap(roi_bottom_left, top_right=roi_top_right)

# fig = plt.figure()
# ax = fig.add_subplot(projection=cusp_submap)
# cusp_submap.plot(axes=ax, clip_interval=(1,99.5)*u.percent, norm=colors.LogNorm())
# cbar = plt.colorbar()
# cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
# plt.savefig('aia_131_imag.png')
#
# #plt.show()
# plt.close()

class ProfilePlot():

    def __init__(self, pix1, pix2, res, imag):
        self.pix1 = pix1
        self.pix2 = pix2
        self.imag = imag
        self.num = res

        x, y = np.linspace(self.pix1[0], self.pix2[0], self.num), \
               np.linspace(self.pix1[1], self.pix2[1], self.num)

        self.profile = scipy.ndimage.map_coordinates(self.imag, np.vstack((x, y)))

# def cut_across_imag(pix1, pix2, res, imag):
#     x0, y0 = pix1[0], pix1[1]  # These are in _pixel_ coordinates!!
#     x1, y1 = pix2[0], pix2[1]
#     num = res
#     x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
#     profile = scipy.ndimage.map_coordinates(imag, np.vstack((x, y)))
#     return profile


'''
Examine AIA brightness distribution along the line
'''

# Extract the values along the line, using cubic interpolation
sdo_img = np.flip(np.array(cusp_submap._data), axis=0)
sdo_img_rot = rotate(sdo_img, -19, reshape=False)
#sdo_img_mag = clipped_zoom(sdo_img_rot, 1.5)

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
synth_plot_settings = {'resolution': sdo_img_rot.shape[0]}
synth_view_settings = {'normal_vector': [0.35, 0.0, 0.836],
                       'north_vector': [0.0, 1.0, 0.0]}
aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings, view_settings=synth_view_settings)

#%%
synth_imag_rot = np.rot90(np.array(aia_synthetic.image))  # np.rot90(aia_synthetic.image) #rotate(np.array(imag), 90, reshape=False)

'''
Examine synth_image brightness distribution along the line
'''

zi = ProfilePlot([310, 300], [310, 5], 500, sdo_img_rot)
prof2 = ProfilePlot([535, 236], [100, 236], 500, synth_imag_rot)

'''
Sample imag
'''
# #-- Generate some data...
# x, y = np.mgrid[-5:5:585j, -5:5:585j]
# z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)
#
# #-- Extract the line...
# # Make a line with "num" points...
# x0, y0 = 5, 4.5 # These are in _pixel_ coordinates!!
# x1, y1 = 60, 75
# num = 1000
# x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
#
# # Extract the values along the line, using cubic interpolation
# zi = scipy.ndimage.map_coordinates(z, np.vstack((x,y)))

'''
Plot obs and synthetic image
'''

#%%
sdoaia131 = matplotlib.colormaps['sdoaia131']

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
plt.ioff()
plt.style.use('fast')

from mpl_toolkits.axes_grid1 import make_axes_locatable
gs = fig.add_gridspec(3, 4)

ax1 = fig.add_subplot(gs[:2, :2])

im = ax1.imshow(sdo_img_rot, norm=colors.LogNorm(vmin=1, vmax=800), cmap=sdoaia131)
ax1.plot([zi.pix1[0], zi.pix2[0]], [zi.pix1[1], zi.pix2[1]], 'ro-')
ax1.axis('image')
ax1.set_title('AIA/SDO 131A 2011-03-07 18:09:09')
ax1.set_xlabel('X, Pixel')
ax1.set_ylabel('Y, Pixel')

ax2 = fig.add_subplot(gs[:2, 2:])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
ax2.set_title('Synthetic image')
ax2.set_xlabel('X, Pixel')
ax2.set_ylabel('Y, Pixel')
ax2.imshow(synth_imag_rot, norm=colors.LogNorm(vmin=1, vmax=800), cmap=sdoaia131)
ax2.plot([prof2.pix1[1], prof2.pix2[1]], [prof2.pix1[0], prof2.pix2[0]], 'ro-')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)

ax3 = fig.add_subplot(gs[2, :])
ax3.plot(zi.profile, label='AIA')
ax3.plot(prof2.profile, label='MHD Synthetic')
#ax3.plot(prof2.profile/1.5, label='MHD Synthetic (2)', linestyle='--')
ax3.legend()
ax3.set_yscale('log')
#ax4 = fig.add_subplot(gs[-1, -2])

#plt.tight_layout()

plt.savefig('cut_side_4.png')
plt.close()
