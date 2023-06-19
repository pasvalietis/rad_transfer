import yt
import numpy as np
import sys
import os.path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.buffer import downsample
from rad_transfer.tests.xray_debug import proj_and_imag
from rad_transfer.emission_models import uv
from rad_transfer.emission_models import xrt
from rad_transfer.visualization.colormaps import color_tables

#%% Load input data
downs_factor = 3
original_file_path = '../datacubes/flarecs-id.0035.vtk'
downs_file_path = './subs_dataset_' + str(downs_factor) + '.h5'
rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")
#%%
channel = 'Be-thin'
hinode_xrt_model = xrt.XRTModel("temperature", "density", channel)
hinode_xrt_model.make_intensity_fields(rad_buffer_obj)
#%%
xrt_colormap = color_tables.xrt_color_table()
imag = proj_and_imag(rad_buffer_obj, 'xrt_filter_band', norm_vec=[0.13, 0.1, 0.7],
                     vmin=2, vmax=2e3, cmap=xrt_colormap, label=channel)



