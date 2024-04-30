CLB_PATH = '/home/saber/CoronalLoopBuilder'

import sys
sys.path.insert(1, CLB_PATH)
# noinspection PyUnresolvedReferences
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d
from utils.lso_align import synthmap_plot
from sunpy.map import Map
import pickle

import matplotlib.pyplot as plt

CLB_PATH = '/home/saber/CoronalLoopBuilder'
CLB_PATH += '/examples/testing'
XRT_EV_PATH = CLB_PATH + '/downloaded_events'
RAD_PATH = '/home/saber/rad_transfer/tests/sab_tests/'

# Select date (0=2012, 1=2013)
yrsel = 0

# Prepare correct paths
if yrsel == 0:
    MAP_PATH = CLB_PATH + '/maps/2012/AIA-131.pkl'
    IMG_PATH = XRT_EV_PATH + '/2012-07-19/XRTL1_2012_new/L1_XRT20120719_113821.1.fits'
    PARAMS_PATH = RAD_PATH + 'loop_params/2012/front_2012_testing.pkl'
elif yrsel == 1:
    MAP_PATH = CLB_PATH + '/maps/2013/AIA-171.pkl'
    IMG_PATH = XRT_EV_PATH + '/2013-05-15/XRTL1_2013_new/L1_XRT20130515_040058.0.fits'
    PARAMS_PATH = RAD_PATH + 'loop_params/2013/front_2013_testing.pkl'

with open(MAP_PATH, 'rb') as f:
    img = pickle.load(f)
    f.close()

# Select function
# 0=real/synth + loop, 1=slit analysis, 2=real/synth compare, 3=xrt/synth compare, 4=synth + loop
fsel = 1

# Plot real and synthetic views
if fsel == 0:
    fig = plt.figure()
    synth_axs, synth_map = synthmap_plot(PARAMS_PATH, fig=fig, map_path=MAP_PATH, plot="synth", instr='aia', channel=171)
    coronal_loop1 = CoronalLoopBuilder(fig, [synth_axs], [img], pkl=PARAMS_PATH)

    plt.show()
    plt.close()

    # coronal_loop1.save_params_to_pickle('2012/front_2012.pkl')
    # coronal_loop1.save_params_to_pickle('2012/front_2012_testing.pkl')

# Slit Intensity
elif fsel == 1:
    from utils.analysis import slit_intensity

    # 2012 Slit
    if yrsel==0:
        norm_slit = [[975, 1050], [-240, -260]]   # x1, x2, y1, y2
        perp_slit = [[1025, 1010], [-225, -275]]
    # 2013 Slits
    elif yrsel == 1:
        norm_slit = [[-925, -975], [215, 225]]
        perp_slit = [[-935, -945], [250, 195]]

    xrt_map = Map(IMG_PATH)
    slit_intensity(PARAMS_PATH, norm_slit, perp_slit, xsmap=xrt_map ,smap=img, clb=True)

# Compare Contrast Synth/AIA
elif fsel == 2:
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    ax, synthmap = synthmap_plot(PARAMS_PATH, smap=img, fig=subfigs[0], plot='synth')

    ax2 = subfigs[1].add_subplot(projection=img)
    img.plot(axes=ax2)

    coronal_loop1 = CoronalLoopBuilder(fig, [ax, ax2], [synthmap, img], pkl=PARAMS_PATH)

    plt.show()
    plt.close()

# Compare Contrast Synth/XRT
elif fsel == 3:
    # DO NOT USE XRT IMAGE TO CREATE SYNTHETIC IMAGE - ONLY AIA
    # WRONG WCS - INCORRECT ALIGNMENT

    fig = plt.figure()
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    xrt_map = Map(IMG_PATH)
    ax, synthmap = synthmap_plot(PARAMS_PATH, smap=xrt_map, fig=subfigs[0], plot='synth')

    ax2 = subfigs[1].add_subplot(projection=xrt_map)
    xrt_map.plot(axes=ax2)

    coronal_loop1 = CoronalLoopBuilder(fig, [ax, ax2], [synthmap, xrt_map], pkl=PARAMS_PATH)

    plt.show()
    plt.close()

# Synth with CLB
elif fsel == 4:
    fig = plt.figure()
    ax, synthmap = synthmap_plot(PARAMS_PATH, map_path=MAP_PATH, fig=fig, plot='synth')

    coronal_loop1 = CoronalLoopBuilder(fig, [ax], [synthmap], pkl=PARAMS_PATH)

    plt.show()
    plt.close()

