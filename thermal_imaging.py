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

thermal_model = pyxsim.CIESourceModel(model='apec',
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
from more_itertools import chunked

data = ds.all_data()
model='apec',
emin=emin,
emax=emax,
nbins=100,
Zmet=0.3,
binscale='log'
kT_min=0.025,
kT_max=64.0

#%%
from pyxsim.spectral_models import TableCIEModel
#INIT DEFAULTS
var_elem=None
thermal_broad=True
model_root=None
model_vers=None
nolines=False
abund_table="angr"
prng = None
_nei = False
var_elem_keys = list(var_elem.keys()) if var_elem else None
#if model in ["apec", "spex"]:
spectral_model = TableCIEModel(
                model,
                emin,
                emax,
                nbins,
                kT_min,
                kT_max,
                binscale=binscale,
                var_elem=var_elem_keys,
                thermal_broad=thermal_broad,
                model_root=model_root,
                model_vers=model_vers,
                nolines=nolines,
                nei=_nei,
                abund_table=abund_table,
            )
#%%
orig_shape = data[('gas', 'temperature')].shape

if len(orig_shape) == 0:
    orig_ncells = 0
else:
    orig_ncells = np.prod(orig_shape)

ret = np.zeros(orig_ncells)
cut = True

kT = np.ravel(data[('gas', 'temperature')].to_value("keV", "thermal"))
cut &= (kT >= kT_min) & (kT <= kT_max)

#cell_nrm = np.ravel(chunk[self.emission_measure_field].d * spectral_norm)
num_cells = cut.sum()
kT = kT[cut]

num_photons_max = 10000000
number_of_photons = np.zeros(num_cells, dtype="int64")
energies = np.zeros(num_photons_max)

start_e = 0
end_e = 0

idxs = np.where(cut)[0]

for ck in chunked(range(num_cells), 100):

    ibegin = ck[0]
    iend = ck[-1] + 1
    nck = iend - ibegin

    #cnm = cell_nrm[ibegin:iend]

    kTi = kT[ibegin:iend]
    cspec, mspec, vspec = spectral_model.get_spectrum(kTi)

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
