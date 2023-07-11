import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib
import scipy.ndimage
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import math
from visualization.colormaps import color_tables

import scipy.ndimage
import scipy.misc

import sunpy.map
import sunpy.visualization.colormaps as cm
#from sunpy.data.sample import AIA_131_IMAGE

import yt
import os
from utils import proj_imag


aia_imag = './aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits'
aia_map = sunpy.map.Map(aia_imag)

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=150*u.arcsec, frame=aia_map.coordinate_frame)
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=500*u.arcsec, frame=aia_map.coordinate_frame)
cusp_submap = aia_map.submap(roi_bottom_left, top_right=roi_top_right)

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

# fig = plt.figure()
# ax = fig.add_subplot(projection=cusp_submap)
# cusp_submap.plot(axes=ax, clip_interval=(1,99.5)*u.percent, norm=colors.LogNorm())
# cbar = plt.colorbar()
# cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
# plt.savefig('aia_131_imag.png')
#
# #plt.show()
# plt.close()

'''
Examine AIA brightness distribution along the line
'''

x0, y0 = 310, 300  # These are in _pixel_ coordinates!!
x1, y1 = 310, 20
num = 500
x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

# Extract the values along the line, using cubic interpolation
sdo_img = np.flip(np.array(cusp_submap._data), axis=0)
sdo_img_rot = rotate(sdo_img, -19, reshape=False)
#sdo_img_mag = clipped_zoom(sdo_img_rot, 1.5)

zi = scipy.ndimage.map_coordinates(sdo_img_rot, np.vstack((x,y)))

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
instr = 'aia' #aia or xrt
channel = '131'
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep
imag = proj_imag.make_filter_image(subs_ds, instr, channel, resolution=585,
                                    vmin=1, vmax=300, norm_vec=[0.2, 0.0, 1.0],
                                    north_vector=[0., 1., 0.], figpath=fname)

imag_rot = np.rot90(imag) #rotate(np.array(imag), 90, reshape=False)

'''
Examine synth_image brightness distribution along the line
'''

x0_, y0_ = 240, 450  # These are in _pixel_ coordinates!!
x1_, y1_ = 240, 420
num = 500
x_, y_ = np.linspace(x0_, x1_, num), np.linspace(y0_, y1_, num)

# Extract the values along the line, using cubic interpolation
#sdo_img_mag = clipped_zoom(sdo_img_rot, 1.5)

prof2 = scipy.ndimage.map_coordinates(imag_rot, np.vstack((x_,y_)))

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
#divider = make_axes_locatable(ax1)
#cax = divider.append_axes('right', size='5%', pad=0.05)

im = ax1.imshow(sdo_img_rot, norm=colors.LogNorm(vmin=1, vmax=800), cmap=sdoaia131)
ax1.plot([x0, x1], [y0, y1], 'ro-')
ax1.set_title('AIA/SDO 131A 2011-03-07 18:09:09')
ax1.set_xlabel('X, Pixel')
ax1.set_ylabel('Y, Pixel')
#cbar = fig.colorbar(im, cax=cax, orientation='vertical')
#cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)


ax2 = fig.add_subplot(gs[:2, 2:])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
ax2.set_title('Synthetic image')
ax2.set_xlabel('X, Pixel')
ax2.set_ylabel('Y, Pixel')
ax2.plot([x0_, x1_], [y0_, y1_], 'ro-')
ax2.imshow(imag_rot, norm=colors.LogNorm(vmin=1, vmax=800), cmap=sdoaia131)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)

ax3 = fig.add_subplot(gs[2, :])
ax3.set_yscale('log')
ax3.plot(zi)
ax3.plot(prof2)
#ax4 = fig.add_subplot(gs[-1, -2])

#plt.tight_layout()

plt.savefig('cut.png')
plt.close()
