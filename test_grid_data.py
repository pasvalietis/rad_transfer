import yt
yt.enable_parallelism()
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mp, keV, kpc
from yt.units import dimensions
import pyxsim
#%%
ds = yt.load(
    "datacubes/ss3.h5", default_species_fields="ionized")