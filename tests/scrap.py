import pickle

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as const
import sunpy.map

clb_path = '/home/saber/CoronalLoopBuilder/examples/testing/'
map_path = (clb_path + 'maps/2013/AIA-171.pkl')
with open(map_path, 'rb') as f:
    img = pickle.load(f)
    f.close()

params_path = clb_path + 'loop_params/2013/front_2013.pkl'
with open(params_path, 'rb') as f:
    params = pickle.load(f)
    f.close()
lat = params['theta0'].value
lon = params['phi0'].value
height = params['height']

coord = SkyCoord(lon=lon * u.deg, lat=lat * u.deg, radius=const.R_sun, frame='heliographic_stonyhurst')
print(coord)

pixels = img.wcs.world_to_pixel(coord)
center_image_pix = [img.data.shape[0]/2., img.data.shape[1]/2.]*u.pix

ref_coord = img.reference_coordinate
ref_pix = img.reference_pixel

import matplotlib.pyplot as plt
fig = plt.figure()
ax=plt.subplot(projection=img)
img.plot()

ax.plot_coord(coord,'o', color='b')
ax.plot(pixels[0]*u.pix, pixels[1]*u.pix, 'x', color='r')
ax.plot(center_image_pix[0], center_image_pix[1], 'x', color='k')
ax.plot_coord(ref_coord,'o', color='k')
ax.plot(ref_pix.x, ref_pix.y, 'x', color='g')

plt.show()
plt.close()
