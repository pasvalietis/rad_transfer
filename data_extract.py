import os
import yt
#from cProfile import label
#from colorsys import yiq_to_rgb
import math
#from turtle import color
import numpy as np
from scipy.optimize import curve_fit

import pickle
#import gc
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Athena_Reader
import athena_vtk_reader
import mhdpost_func

# ------------------------------------------------------------------------------
# 0.1 Initialize parameters
# ------------------------------------------------------------------------------
filename = 'flarecs-id.0035.vtk'
vtkfile = './datacubes/' + filename
yc_chosen = 0.48

ds = yt.load(vtkfile)
# ------------------------------------------------------------------------------
# 1. Read 3D vtk data
# ------------------------------------------------------------------------------
#%%
#vtkfile = './datacubes/' + filename
#var = athena_vtk_reader.read(vtkfile, outtype='cons', nscalars=0)