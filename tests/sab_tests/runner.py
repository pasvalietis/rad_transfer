CLB_PATH = '/home/saber/CoronalLoopBuilder'

import sys
sys.path.insert(1, CLB_PATH)
# noinspection PyUnresolvedReferences
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d
from utils.lso_align import synthmap_plot
from sunpy.map import Map
import pickle

import matplotlib.pyplot as plt

# Path to local Coronal Loop Builder
CLB_PATH = CLB_PATH + '/examples/testing'
# Path to target sunpy map
MAP_PATH = CLB_PATH + '/maps/2012/AIA-171.pkl'
# Path to clb loop parameters
PARAMS_PATH = ('/home/saber/rad_transfer/tests/sab_tests/'
               'loop_params/2012/front_2012_testing.pkl')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot real and synthetic views

# with open(MAP_PATH, 'rb') as f:
#     img = pickle.load(f)
#     f.close()
#
# fig = plt.figure()
# synth_axs, synth_map = synthmap_plot(PARAMS_PATH, fig=fig, map_path=MAP_PATH, plot="synth", instr='aia', channel=171)
# coronal_loop1 = CoronalLoopBuilder(fig, [synth_axs], [img], pkl=PARAMS_PATH)
#
# plt.show()
# plt.close()

# coronal_loop1.save_params_to_pickle('2012/front_2012.pkl')
# coronal_loop1.save_params_to_pickle('2012/front_2012_testing.pkl')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Slit Intensity

from utils.analysis import slit_intensity

norm_slit = [[975, 1050], [-240, -260]]
perp_slit = [[1025, 1010], [-225, -275]]

# map = Map('/home/saber/Downloads/2012_lvl1/L1_XRT20120719_113731.0.fits')
# slit_intensity(PARAMS_PATH, norm_slit, perp_slit, map=map)

slit_intensity(PARAMS_PATH, norm_slit, perp_slit, map_path=MAP_PATH)
