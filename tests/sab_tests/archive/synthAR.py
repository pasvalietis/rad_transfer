import astropy.units as u
import astropy.time
from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_earth
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/home/saber/synthesizAR')

# noinspection PyUnresolvedReferences
import synthesizAR
# noinspection PyUnresolvedReferences
from synthesizAR.models import semi_circular_arcade
# noinspection PyUnresolvedReferences
from synthesizAR.interfaces import RTVInterface
# noinspection PyUnresolvedReferences
from synthesizAR.instruments import InstrumentSDOAIA
# noinspection PyUnresolvedReferences
from synthesizAR.visualize.fieldlines import plot_fieldlines

obstime = astropy.time.Time('2022-11-14T22:00:00')
# Coordinate of 'center' loop (at a distance 'radius' away)
pos = SkyCoord(lon=15*u.deg, lat=25*u.deg, radius=1*u.AU, obstime=obstime, frame='heliographic_stonyhurst')
# Length, Width, Num Strands, Observer
arcade_coords = semi_circular_arcade(80*u.Mm, 5*u.deg, 50, pos, inclination=10*u.deg)

strands = [synthesizAR.Loop(f'strand{i}', c) for i, c in enumerate(arcade_coords)]
arcade = synthesizAR.Skeleton(strands)

earth_observer = get_earth(obstime)
# arcade.peek(observer=earth_observer,
#             axes_limits=[(150, 300)*u.arcsec, (275, 425)*u.arcsec])

fig, axs, image_map = plot_fieldlines(*[_.coordinate for _ in arcade.loops], observer=earth_observer,
                axes_limits=[(150, 300)*u.arcsec, (275, 425)*u.arcsec])

# fig = plt.figure()
centers = arcade.all_coordinates_centers
# ax = fig.add_subplot(projection=pos)

ptn = axs.plot_coord(centers, color='b', marker='.', ms=1)
# ax.plot_coord(centers)

plt.show()
plt.close()

