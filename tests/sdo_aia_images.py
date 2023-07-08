import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import scipy.ndimage
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import math

import scipy.ndimage
import scipy.misc

import sunpy.map
#from sunpy.data.sample import AIA_131_IMAGE

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
sdo_img_mag = clipped_zoom(sdo_img_rot, 1.5)
zi = scipy.ndimage.map_coordinates(sdo_img_rot, np.vstack((x,y)))
#%%
#-- Plot...

# fig, axes = plt.subplots(nrows=3)
# axes[0].imshow(sdo_img_rot, norm=colors.LogNorm(vmin=1, vmax=100))
# axes[0].plot([x0, x1], [y0, y1], 'ro-')
# axes[0].axis('image')
#
#
#
# axes[2].plot(zi)
#
# plt.savefig('aia_brightness.png')
# #plt.show()
# plt.close()
#
#%%
# Placing the plots in the plane
plot1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
plot2 = plt.subplot2grid((3, 3), (0, 2), rowspan=3, colspan=2)
plot3 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)

# Using Numpy to create an array x
x = np.arange(1, 10)

# Plot for square root
plot2.plot(x, x ** 0.5)
plot2.set_title('Square Root')

# Plot for exponent
plot1.plot(x, np.exp(x))
plot1.set_title('Exponent')

# Plot for Square
plot3.plot(x, x * x)
plot3.set_title('Square')

# Packing all the plots and displaying them
plt.tight_layout()
plt.show()