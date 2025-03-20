from rushlight.config import config
from rushlight.utils.lso_align import synthmap_plot
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d # type: ignore

from sunpy.map import Map
import astropy.units as u
import pickle

import matplotlib.pyplot as plt

def select_date(yrsel: int):
    """Returns the proper paths and images for analysis in runner

    :param yrsel: integer indicating year of observation (2011-2013)
    :type yrsel: int
    :return: Paths to AIA map, Stereo map, XRT map, and CLB params
    :rtype: dict
    """

    CLB_PATH = config.CLB_PATH + '/examples/testing'
    XRT_EV_PATH = CLB_PATH + '/downloaded_events'
    RAD_PATH = config.RAD_PATH

    # Prepare correct paths
    if yrsel == 2012:
        MAP_PATH = CLB_PATH + '/maps/2012/AIA-171.pkl'
        STE_PATH = CLB_PATH + '/maps/2012/STEREOA-171.pkl'
        IMG_PATH = XRT_EV_PATH + '/2012-07-19/XRTL1_2012_new/L1_XRT20120719_113821.1.fits'
        # PARAMS_PATH = RAD_PATH + 'loop_params/2012/front_2012_testing.pkl'
        PARAMS_PATH = CLB_PATH + '/loop_params/2012/back_2012_center.pkl'
        zoom = (1/2)

    elif yrsel == 2013:
        MAP_PATH = CLB_PATH + '/maps/2013/AIA-171.pkl'
        STE_PATH = CLB_PATH + '/maps/2013/STEREOB-171.pkl'
        IMG_PATH = XRT_EV_PATH + '/2013-05-15/XRTL1_2013_new/L1_XRT20130515_040058.0.fits'
        PARAMS_PATH = RAD_PATH + 'loop_params/2013/front_2013_testing.pkl'
        zoom = (1/3)

    elif yrsel == 2011:
        MAP_PATH = CLB_PATH + '/maps/AIA-131-2011.pkl'
        STE_PATH = CLB_PATH + '/maps/STEREOB-171-2011.pkl'
        IMG_PATH = XRT_EV_PATH + '/L1_XRT20110307_180901.3.fits'
        PARAMS_PATH = RAD_PATH + 'loop_params/loop1_params_20110307.pkl'
        zoom = (1/8)

    with open(MAP_PATH, 'rb') as f:
        img = pickle.load(f)
        f.close()

    PATHS = {
        'MAP_PATH': MAP_PATH,
        'STE_PATH': STE_PATH,
        'IMG_PATH': IMG_PATH,
        'PARAMS_PATH': PARAMS_PATH,
        'zoom': zoom
    }

    return PATHS
