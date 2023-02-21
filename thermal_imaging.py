import yt
yt.enable_parallelism()
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mp, keV, kpc
import pyxsim

ds = yt.load(
    "datasets/ShockCloud/id1/Cloud-id1.0050.vtk", default_species_fields="ionized"
    #"datacubes/flarecs-id.0035.vtk", default_species_fields="ionized"
)
slc = yt.SlicePlot(
    ds, "z", [("gas", "density"), ("gas", "temperature")], width=(0.05, "m")
)
slc.save()

emin = 5.0
emax = 80.0
#%%
thermal49_model = pyxsim.ThermalBremsstrahlung49(emin, emax, ('gas', 'temperature'))

'''
Export matplotlib image
'''
# _, c = ds.find_max(("gas", "density"))
# proj = ds.proj(("gas", "density"), 0)
#
# width = (10, "kpc")  # we want a 1.5 mpc view
# res = [200, 200]  # create an image with 1000x1000 pixels
# frb = proj.to_frb(width, res, center=c)
#
# plt.imshow(np.array(frb["gas", "density"]))
# plt.savefig("my_perfect_figure.png")
# plt.show()
