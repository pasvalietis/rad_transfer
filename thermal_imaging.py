import yt
yt.enable_parallelism()
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mp, keV, kpc
from yt.units import dimensions
import pyxsim

# units_override = {
#     "length_unit": (1.5e8, "m"),
#     "time_unit": (109.8, "s"),
#     "mass_unit": (8.43e38, "kg"),
#     #"density_unit": (2.5e14, "kg/m**3"),
#     "velocity_unit": (1.366e6, "m/s"),
#     "temperature_unit": (1.13e8, "K"),
# }

path1 = "datacubes/flarecs-id.0035_ss3.h5"
path2 = "GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150"
#ds = yt.load(
#    "datasets/ShockCloud/id1/Cloud-id1.0050.vtk", units_override=units_override, default_species_fields='ionized')
    #"datacubes/flarecs-id.0035_ss3.h5", units_override=units_override, default_species_fields='ionized')
#norm = yt.YTQuantity(1.0, "cm**2*keV/dyne")

ds = yt.load(
    path2, default_species_fields="ionized"
)

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
"""
GENERATE AN X-RAY IMAGE
"""
#
# slc = yt.SlicePlot(
#     ds, "z", [("gas", "density")], width=(0.01, "m")
# )
# slc.save()
#
# emin = 0.025
# emax = 64.0
#
# thermal_model = pyxsim.CIESourceModel(model='apec',
#                                       emin=emin,
#                                       emax=emax,
#                                       nbins=100,
#                                       Zmet=0.3,
#                                       binscale='log')
# #%%
# xray_fields = thermal_model.make_intensity_fields(ds,
#                                                   emin=emin,
#                                                   emax=emax,
#                                                   dist=(1.5e11, "m"))
#
# #%%
# fname = 'xray_photon_intensity_'+str(emin)+'_'+str(emax)+'_keV'
# slc = yt.SlicePlot(
#     ds, "z",  ('gas', fname), width=(0.01, "m")
# )
# slc.save()


#%%
"""
TEST THE PYXSIM THERMAL MODEL PHOTON SAMPLING
***
Add emission measure field
"""
def _em_field(field, data):
    return (
        data["gas", "density"]**2.
    )

ds.add_field(
    name=("gas", "em_field"),
    function=_em_field,
    sampling_type="local",
    units="g**2/cm**6",
    force_override=True,
)
#%%
"""
Follow the CIESourceModel(ThermalSourceModel) algorithm to get spectrum
consider 'apec' model
"""
from numbers import Number

from more_itertools import chunked
from pyxsim.spectral_models import TableCIEModel
from pyxsim.utils import compute_H_abund, mylog, parse_value
from soxs.utils import parse_prng, regrid_spectrum
from soxs.constants import abund_tables, atomic_weights, elem_names, metal_elem

_nei = False
_density_dependence = False

emin = 0.025
emax = 64.0
from pyxsim.utils import ParallelProgressBar, parse_value
emin = parse_value(emin, "keV")
emax = parse_value(emax, "keV")

data = ds.all_data()
model='apec'
emin=emin
emax=emax
nbins=100
Zmet=0.3
binscale='log'
kT_min=0.025
kT_max=64.0

#INIT DEFAULTS (from CIESourceModel)
temperature_field = ("gas", "temperature")
emission_measure_field = ("gas", "em_field")
h_fraction = None
kT_min = 0.025
kT_max = 64.0
max_density = None
var_elem = None
method = "invert_cdf"
thermal_broad = True
model_root = None
model_vers = None
nolines = False
abund_table = "angr"
prng = None
prng = parse_prng(prng)

density_field = None  # Will be determined later
nh_field = None  # Will be set by the subclass
tot_num_cells = 0  # Will be determined later
ftype = "gas"
binscale = binscale
kT_min = kT_min
kT_max = kT_max

#nH_min = nH_min
#nH_max = nH_max
redshift = None
pbar = None
Zconvert = 1.0
mconvert = {}
atable = abund_tables[abund_table].copy()
if h_fraction is None:
    h_fraction = compute_H_abund(abund_table)
h_fraction = h_fraction
#%%
if var_elem is None:
    var_elem = {}
    var_elem_keys = None
    num_var_elem = 0

var_elem_keys = list(var_elem.keys()) if var_elem else None
#%%
#  'Blackbox part'
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
ebins = spectral_model.ebins
de = spectral_model.de
emid = spectral_model.emid
bin_edges = np.log10(ebins) if binscale == "log" else ebins
nbins = emid.size
model_vers = spectral_model.model_vers

#%%
'''
Required inits from setup_model
'''
redshift = 0.0
spectral_model.prepare_spectrum(redshift)
#%%
mode = "photons"
#%%
"""
Spectral parameters
"""
nbins = 100  # number of nbins : integer the number of channels in the spectrum.
spectral_norm = 1.

spec = np.zeros(nbins)
orig_shape = data[('gas', 'temperature')].shape

if len(orig_shape) == 0:
    orig_ncells = 0
else:
    orig_ncells = np.prod(orig_shape)

ret = np.zeros(orig_ncells)
cut = True

kT = np.ravel(data[('gas', 'temperature')].to_value("keV", "thermal"))
cut &= (kT >= kT_min) & (kT <= kT_max)
#%%
cell_nrm = np.ravel(data[('gas', 'em_field')].d * spectral_norm)
#%%
num_cells = cut.sum()
kT = kT[cut]

if nh_field is not None:
    nH = np.ravel(ds[('gas', nh_field)].d)[cut]
else:
    nH = None

if isinstance(h_fraction, Number):
    X_H = h_fraction
else:
    X_H = np.ravel(ds[('gas', h_fraction)].d)[cut]

var_ion_keys = spectral_model.var_ion_names

if _nei:
    metalZ = np.zeros(num_cells)
    elem_keys = var_ion_keys
else:
    elem_keys = var_elem_keys
    if isinstance(Zmet, Number):
        metalZ = Zmet * np.ones(num_cells)
    else:
        mZ = ds[('gas', Zmet)] #chunk[self.Zmet]
        fac = Zconvert
        if str(mZ.units) != "Zsun":
            fac /= X_H
        metalZ = np.ravel(mZ.d[cut] * fac)

elemZ = None

num_photons_max = 10000000
number_of_photons = np.zeros(num_cells, dtype="int64")
energies = np.zeros(num_photons_max)

start_e = 0
end_e = 0

idxs = np.where(cut)[0]

#%%
for ck in chunked(range(num_cells), 100):

    ibegin = ck[0]
    iend = ck[-1] + 1
    nck = iend - ibegin

    cnm = cell_nrm[ibegin:iend]

    kTi = kT[ibegin:iend]

    cspec, mspec, vspec = spectral_model.get_spectrum(kTi)
    tot_spec = cspec

    tot_spec = cspec
    tot_spec += metalZ[ibegin:iend, np.newaxis] * mspec
    if num_var_elem > 0:
        tot_spec += np.sum(
            elemZ[:, ibegin:iend, np.newaxis] * vspec, axis=0
        )
    np.clip(tot_spec, 0.0, None, out=tot_spec)

    if mode == "photons":
        spec_sum = tot_spec.sum(axis=-1)
        cell_norm = spec_sum * cnm

        cell_n = np.atleast_1d(prng.poisson(lam=cell_norm))

        number_of_photons[ibegin:iend] = cell_n
        end_e += int(cell_n.sum())

        norm_factor = 1.0 / spec_sum
        p = norm_factor[:, np.newaxis] * tot_spec
        cp = np.insert(np.cumsum(p, axis=-1), 0, 0.0, axis=1)
        ei = start_e

        for icell in range(nck):
            cn = cell_n[icell]
            if cn == 0:
                continue
            if method == "invert_cdf":
                randvec = prng.uniform(size=cn)
                randvec.sort()
                cell_e = np.interp(randvec, cp[icell, :], bin_edges)
            elif method == "accept_reject":
                eidxs = prng.choice(nbins, size=cn, p=p[icell, :])
                cell_e = emid[eidxs]
            while ei + cn > num_photons_max:
                num_photons_max *= 2
            if num_photons_max > energies.size:
                energies.resize(num_photons_max, refcheck=False)
            energies[ei: ei + cn] = cell_e
            ei += cn
        start_e = end_e
    #pbar.update(nck)


if mode == "photons":
    active_cells = number_of_photons > 0
    idxs = idxs[active_cells]
    ncells = idxs.size
    ee = energies[:end_e].copy()
    if binscale == "log":
        ee = 10**ee
    #return ncells, number_of_photons[active_cells], idxs, ee
elif mode == "spectrum":
    #return spec
    pass
else:
    #return np.resize(ret, orig_shape)
    pass


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
