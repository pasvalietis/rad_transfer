import matplotlib

matplotlib.rc("font", size=18, family="serif")
import yt
yt.enable_parallelism()
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mp, keV, kpc
import pyxsim

ds = yt.load(
    "GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0100", default_species_fields="ionized"
)
slc = yt.SlicePlot(
    ds, "z", [("gas", "density"), ("gas", "temperature")], width=(1.0, "Mpc")
)
slc.save()

#%%
plaw_model = pyxsim.PowerLawSourceModel(1.0, 1, 80.0, "power_law_emission", 1.0)
# A = yt.YTQuantity(500.0, "cm**2")


# exp_time = yt.YTQuantity(1.0e5, "s")
# redshift = 0.03
# sp = ds.sphere("c", (0.5, "Mpc"))
norm = yt.YTQuantity(1.0e-19, "photons/s/keV")

def _power_law_emission(field, data):
    return norm * data["cell_mass"] / (1.0 * mp)


ds.add_field(
    ("gas", "power_law_emission"),
    function=_power_law_emission,
    sampling_type="local",
    units="photons/s/keV",
)

#npp, npc = pyxsim.make_photons("plaw_photons", sp, redshift, A, exp_time, plaw_model)

xray_fields = plaw_model.make_intensity_fields(ds, 20.0, 30.0, dist=(3, "mpc"))
print(xray_fields)

prj = yt.ProjectionPlot(
    ds, "z", ("gas", "xray_photon_intensity_20.0_30.0_keV"), width=(1.0, "Mpc")
)
prj.save()