#!/usr/bin/env python

# Import necessary packages
from rushlight.utils.proj_imag_classified import SyntheticFilterImage as sfi
from CoronalLoopBuilder.builder import CoronalLoopBuilder # type: ignore
from sunpy.map import Map
import pickle
import matplotlib.pyplot as plt
from rushlight.config import config

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect data

datacube = config.SIMULATIONS['DATASET']    # Collect synthetic box

EVENT = {
'IMG_PATH': './images/L1_XRT20120719_113821.1.fits',
'PARAMS_PATH': './loops/back_2012_center.pkl',
'zoom': 0.5
}

IMG_PATH = EVENT['IMG_PATH']        # Path to XRT observation
PARAMS_PATH = EVENT['PARAMS_PATH']  # Path to CLB loop params
zoom = EVENT['zoom']                # Zoom amount for event

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting
fig = plt.figure()
subfigs = fig.subfigures(1, 2, wspace=0.07)

xrt_map = Map(IMG_PATH)                     # Create map from observation .fits
sfiObj = sfi(datacube, smap=xrt_map,        # Generate a Synthetic Filter Image object
             pkl=PARAMS_PATH, zoom=zoom)

ax1, synthmap, norm, north, image_shift = \
    sfiObj.synthmap_plot(fig=subfigs[0],    # Plot synthetic map object on specified subfigure
                         plot='synth')      # Specify hint for synthetic plot

ax2 = subfigs[1] \
      .add_subplot(projection=xrt_map)      
xrt_map.plot(axes=ax2)                      # Plot XRT observation

CoronalLoopBuilder(                     # Display CLB Loops on real + synthetic maps 
    fig, 
    [ax1, ax2], 
    [synthmap, xrt_map], 
    pkl=PARAMS_PATH)

plt.show()
plt.close()
