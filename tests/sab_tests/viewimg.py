import sunpy.map
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

img_path = ('/home/saber/CoronalLoopBuilder/examples/testing/downloaded_events/'
            'aia_lev1_131a_2012_07_19t06_40_11_97z_image_lev1.fits')

img = sunpy.map.Map(img_path)
# Crop image map
bl = SkyCoord(850 * u.arcsec, -330 * u.arcsec, frame=img.coordinate_frame)
res = 250 * u.arcsec
img = img.submap(bottom_left=bl, width=res, height=res)

plt.figure()
img.plot()

plt.savefig('figs/nosynth.jpg', dpi=250, bbox_inches='tight')

plt.show()
plt.close()
