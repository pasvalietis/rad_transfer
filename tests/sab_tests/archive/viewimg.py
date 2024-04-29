import sunpy.map
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pickle
from astropy.coordinates import SkyCoord
import astropy.units as u
from aiapy.calibrate import normalize_exposure, register, update_pointing

CLB_PATH = '/home/saber/CoronalLoopBuilder'
import sys

sys.path.insert(1, CLB_PATH)
# noinspection PyUnresolvedReferences
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d


def draw_all():
    XRT_EV_PATH = CLB_PATH + '/examples/testing/downloaded_events'
    for filename in os.listdir(XRT_EV_PATH)[11:12]:
        img = sunpy.map.Map(str(os.path.join(XRT_EV_PATH, filename)))

        plt.figure()
        img.plot()

    plt.show()
    plt.close()


# draw_all()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Select date (0=2012, 1=2013)
select = 1

CLB_PATH = '/home/saber/CoronalLoopBuilder'
CLB_PATH += '/examples/testing'
XRT_EV_PATH = CLB_PATH + '/downloaded_events'
RAD_PATH = '/home/saber/rad_transfer/tests/sab_tests/'

# Prepare correct paths
if select == 0:
    MAP_PATH = CLB_PATH + '/maps/2012/AIA-131.pkl'
    IMG_PATH = XRT_EV_PATH + '/2012-07-19/XRTL1_2012_new/L1_XRT20120719_113821.1.fits'
    PARAMS_PATH = RAD_PATH + 'loop_params/2012/front_2012_testing.pkl'
else:
    MAP_PATH = CLB_PATH + '/maps/2013/AIA-171.pkl'
    IMG_PATH = XRT_EV_PATH + '/2013-05-15/XRTL1_2013_new/L1_XRT20130515_040058.0.fits'
    PARAMS_PATH = RAD_PATH + 'loop_params/2013/front_2013_testing.pkl'

# Load AIA Map
with open(MAP_PATH, 'rb') as f:
    img = pickle.load(f)
    f.close()
# Load XRT Map
imgx = sunpy.map.Map(IMG_PATH)

# Crop XRT Map
# top_right = SkyCoord(1100*u.arcsec, -100*u.arcsec, frame=imgx.coordinate_frame)
# bottom_left = SkyCoord(750*u.arcsec, -450*u.arcsec, frame=imgx.coordinate_frame)
#
# imgx = imgx.submap(bottom_left, top_right=top_right)

fig = plt.figure(figsize=(10, 6))
fig.set_tight_layout(True)
plt.ioff()  # Turn interactive mode off
plt.style.use('fast')
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0, 0], projection=img)
img.plot_settings['norm'] = colors.LogNorm(10, img.max())
img.plot(axes=ax1)
ax1.grid(False)
img.draw_limb()

ax2 = fig.add_subplot(gs[0, 1], projection=imgx)
imgx.plot_settings['norm'] = colors.LogNorm(10, imgx.max())
imgx.plot(axes=ax2)
ax2.grid(False)
imgx.draw_limb()

coronal_loop1 = CoronalLoopBuilder(fig, [ax1, ax2], [img, imgx], pkl=PARAMS_PATH)

plt.show()
plt.close()
