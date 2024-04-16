import sys
sys.path.insert(1, '/home/saber/CoronalLoopBuilder')

import matplotlib.pyplot as plt
# Import synthetic image manipulation tools
from utils.lso_align import synthmap_plot
# noinspection PyUnresolvedReferences
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d

# Path to local Coronal Loop Builder
CLB_PATH = '/home/saber/CoronalLoopBuilder/examples/testing/'
# Path to target sunpy map
MAP_PATH = CLB_PATH + 'maps/2012/AIA-171.pkl'
# Path to clb loop parameters
PARAMS_PATH = ('/home/saber/rad_transfer/tests/sab_tests/'
               'loop_params/2012/front_2012_testing.pkl')

# # Plot real and synthetic views
# fig = plt.figure()
# synth_axs, synth_map = synthmap_plot(MAP_PATH, PARAMS_PATH, fig, plot="synth", instr='aia', channel=171)
# # coronal_loop1 = CoronalLoopBuilder(fig, synth_axs, [img], pkl=PARAMS_PATH)
#
# plt.show()
# plt.close()

# coronal_loop1.save_params_to_pickle('2012/front_2012.pkl')
# coronal_loop1.save_params_to_pickle('2012/front_2012_testing.pkl')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Slit Intensity

from utils.analysis import slit_intensity

norm_slit = [[975, 1050], [-240, -260]]
perp_slit = [[1025, 1020], [-225, -275]]

slit_intensity(MAP_PATH, PARAMS_PATH, norm_slit, perp_slit)
