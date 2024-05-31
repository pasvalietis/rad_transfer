import os
import sys
from sunpy.map import Map
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u


lvl15 = '/media/saber/My_Passport/Sabastian/xrt/2013_05_15/aia_lvl_15'
lvl15ROI = '/media/saber/My_Passport/Sabastian/xrt/2013_05_15/aia_lvl_15_ROI'
files = os.listdir(lvl15)

for i in tqdm(range(0, len(files))):
    f = files[i]
    m = Map(lvl15 + '/' + f)

    top_right = SkyCoord(-790 * u.arcsec, 275 * u.arcsec, frame=m.coordinate_frame)
    bottom_left = SkyCoord(-940 * u.arcsec, 125 * u.arcsec, frame=m.coordinate_frame)

    lvl15roi_map = m.submap(bottom_left, top_right=top_right)

    lvl15roi_map.save(lvl15ROI + '/' + f, overwrite=True)
