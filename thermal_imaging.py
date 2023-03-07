import yt
yt.enable_parallelism()
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mp, keV, kpc
from yt.units import dimensions
import pyxsim

units_override = {
    "length_unit": (1.5e8, "m"),
    "time_unit": (109.8, "s"),
    "mass_unit": (8.43e38, "kg"),
    #"density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}
ds = yt.load(
    #"datasets/ShockCloud/id1/Cloud-id1.0050.vtk"
    "datacubes/flarecs-id.0035_ss3.h5", units_override=units_override, default_species_fields='ionized')
#norm = yt.YTQuantity(1.0, "cm**2*keV/dyne")

u = ds.units
norm = 1. * u.dyn / u.cm**2
renorm = norm.to('code_pressure')
e0 = yt.YTQuantity(1.0, "keV")

def _temperature(field, data):
    return (
        e0 * data['grid', 'total_energy'] / renorm
    )

ds.add_field(
    ("gas", "temperature"),
    function=_temperature,
    sampling_type="local",
    units="keV",
    dimensions=dimensions.temperature,
)

slc = yt.SlicePlot(
    ds, "z", [("gas", "density")], width=(0.01, "m")
)
slc.save()

emin = 0.025
emax = 64.0

thermal_model = pyxsim.CIESourceModel(model="apec",
                                      emin=emin,
                                      emax=emax,
                                      nbins=100,
                                      Zmet=0.3,
                                      binscale='log')
#%%
xray_fields = thermal_model.make_intensity_fields(ds,
                                                  emin=emin,
                                                  emax=emax,
                                                  dist=(1.5e11, "m"))

#%%
fname = 'xray_photon_intensity_'+str(emin)+'_'+str(emax)+'_keV'
slc = yt.SlicePlot(
    ds, "z",  ('gas', fname), width=(0.01, "m")
)
slc.save()


#%%
#thermal49_model = pyxsim.ThermalBremsstrahlung49(emin, emax, ('gas', 'temperature'))
# '''
# Export matplotlib image
# '''
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
