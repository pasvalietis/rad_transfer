# Base imports
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Units
import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map

# Import synthetic image manipulation tools
import yt
sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

