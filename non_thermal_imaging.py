#import matplotlib

#matplotlib.rc("font", size=18, family="serif")
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

alpha = 5./3.
#%%
plaw_model = pyxsim.PowerLawSourceModel(3.0, 5.0, 80.0, "power_law_emission", alpha)
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
#%%

emin = 20.0
emax = 30.0

from pyxsim.utils import ParallelProgressBar, parse_value

emin = parse_value(emin, "keV")
emax = parse_value(emax, "keV")

emin_src = emin * (1.0)
emax_src = emax * (1.0)

xray_fields = plaw_model.make_intensity_fields(ds, emin, emax, dist=(0.09, "m"))

#print(xray_fields)
#%%
# prj = yt.ProjectionPlot(
#    ds, "z", ("gas", "xray_intensity_20.0_30.0_keV"), width=(0.5, "m")
# )
slc = yt.SlicePlot(
    ds, "z", ("gas", "xray_intensity_20.0_30.0_keV"), width=(0.05, "m")
)
slc.save()

#%%

ftype = 'gas'#plaw_model.emission_field[0]
force_override = False
# dist_fac, redshift = plaw_model._make_dist_fac(
#             ds, redshift, dist, cosmology, per_sa=True
#         )

eif = plaw_model.make_fluxf(emin_src, emax_src, energy=True)

def _intensity_field(field, data):
    ret = data.ds.arr(
        plaw_model.process_data("energy_field", data, 1.0, fluxf=eif),
        "keV/s",
    )
    idV = data[ftype, "density"] / data[ftype, "mass"]
    I = 1 * ret * idV
    return I.in_units("erg/cm**3/s/arcsec**2")

ei_name = (ftype, f"xray_intensity_{emin.value}_{emax.value}_keV")
ei_dname = rf"I_{{X}} ({emin.value}-{emax.value} keV)"

ds.add_field(
    ei_name,
    function=_intensity_field,
    display_name=ei_dname,
    sampling_type="local",
    units="erg/cm**3/s/arcsec**2",
    force_override=force_override,
)

#%%
data = ds.all_data()
# [self.emission_field] reads
print(plaw_model.emission_field)
# Access the corresponding created field
print(ds.all_data()[('gas', 'power_law_emission')])
num_cells = len(ds.all_data()[('gas', 'power_law_emission')])
#alpha = 2.0
if isinstance(alpha, float):
    alpha = alpha * np.ones(num_cells)
else:
    alpha = ds.all_data()[plaw_model.alpha].d

fluxf=None

if fluxf is None:
    ei = plaw_model.emin.v
    ef = plaw_model.emax.v
else:
    ei = fluxf["emin"].v
    ef = fluxf["emax"].v

#%%
mode = "photon_field"
spectral_norm = 1.0
scale_factor = 1.0
plaw_model.observer = "external"
if mode in ["photons", "photon_field"]:
    norm_fac = ef ** (1.0 - alpha) - ei ** (1.0 - alpha)
    norm_fac[alpha == 1] = np.log(ef / ei)
    norm_fac *= plaw_model.e0.v ** alpha
    #norm = norm_fac * chunk[self.emission_field].d
    norm = norm_fac * ds.all_data()[('gas', 'power_law_emission')]
    if np.any(alpha != 1):
        norm[alpha != 1] /= 1.0 - alpha[alpha != 1]
    if mode == "photons":
        norm *= spectral_norm * scale_factor

    # if plaw_model.observer == "internal":
    #     pos = np.array(
    #         [
    #             #np.ravel(chunk[self.p_fields[i]].to_value("kpc"))
    #
    #             for i in range(3)
    #         ]
    #     )
    #     r2 = self.compute_radius(pos)
    #     norm /= r2

    number_of_photons = plaw_model.prng.poisson(lam=norm)
#%%
    start_e = 0
    end_e = 0

    energies = np.zeros(number_of_photons.sum())
#%%
    for i in range(1):
        if number_of_photons[i] > 0:
            end_e = start_e + number_of_photons[i]
            u = plaw_model.prng.uniform(size=number_of_photons[i])
            if alpha[i] == 1:
                e = ei * (ef / ei) ** u
            else:
                e = ei ** (1.0 - alpha[i]) + u * norm_fac[i]
                e **= 1.0 / (1.0 - alpha[i])
            energies[start_e:end_e] = e * scale_factor
            #start_e = end_e

    active_cells = number_of_photons > 0
    ncells = active_cells.sum()

'''
To plot a histogram for the power-law spectrum
'''
#%%
bins = 10**(np.linspace(-1.,3.,90))
plt.xscale('log')
plt.xlabel('energy')
plt.ylabel('photon flux')
#plt.plot(energies[start_e:end_e], energies[start_e:end_e]**(-2.))
count, bins, ignored = plt.hist(energies[start_e:end_e], bins=bins, density=True, log=True)
plt.show()
#%%

    # return (
    #     ncells,
    #     number_of_photons[active_cells],
    #     active_cells,
    #     energies[:end_e].copy(),
    # )
