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

def _power_law_emission(field, data):
    return data["cell_mass"] / (1.0 * mp)

ds.add_field(
    ("gas", "power_law_emission"),
    function=_power_law_emission,
    sampling_type="local",
    units="photons/s/keV",
)

plaw_model = pyxsim.PowerLawSourceModel(1.0, 1, 80.0, "power_law_emission", 1.0)
#%%
emin = 20.0
emax = 30.0
xray_fields = plaw_model.make_intensity_fields(ds, emin, emax, dist=(3, "mpc"))
print(xray_fields)

#%%
'''
Run the following (make_intensity_fields function from sources.py)
here all selfs are replaced with plaw_model that is defined above
'''
# read  dist_fac, redshift
spectral_norm = 1.0
ftype = plaw_model.ftype # returns 'gas'

# Convert input energies into yt quantities
from pyxsim.utils import parse_value
emin = parse_value(emin, "keV") # calls parse_value from pyxsim.utils
emax = parse_value(emax, "keV")

dist_fac, redshift = plaw_model._make_dist_fac(ds, redshift=0.0, dist=None, cosmology=None, per_sa=True)
emin_src = emin * (1.0 + redshift)
emax_src = emax * (1.0 + redshift)

ei_name = (ftype, f"xray_intensity_{emin.value}_{emax.value}_keV")
ei_dname = rf"I_{{X}} ({emin.value}-{emax.value} keV)"

#%%
eif = plaw_model.make_fluxf(emin_src, emax_src, energy=True)

def _intensity_field(field, data):
    ret = data.ds.arr(
        plaw_model.process_data("energy_field", data, spectral_norm, fluxf=eif),
        "keV/s",
    )
    idV = data[ftype, "density"] / data[ftype, "mass"]
    I = dist_fac * ret * idV
    return I.in_units("erg/cm**3/s/arcsec**2")
#%%
ds.add_field(
            ei_name,
            function=_intensity_field,
            display_name=ei_dname,
            sampling_type="local",
            units="erg/cm**3/s/arcsec**2",
            force_override=True,
        )


#%%
#
# #%%
# plaw_model = pyxsim.PowerLawSourceModel(1.0, 1, 80.0, "power_law_emission", 1.0)
# # A = yt.YTQuantity(500.0, "cm**2")
# # exp_time = yt.YTQuantity(1.0e5, "s")
# # redshift = 0.03
# # sp = ds.sphere("c", (0.5, "Mpc"))
#
# norm = yt.YTQuantity(1.0e-19, "photons/s/keV")
#
# def _power_law_emission(field, data):
#     return norm * data["cell_mass"] / (1.0 * mp)
#
# ds.add_field(
#     ("gas", "power_law_emission"),
#     function=_power_law_emission,
#     sampling_type="local",
#     units="photons/s/keV",
# )
#
# #npp, npc = pyxsim.make_photons("plaw_photons", sp, redshift, A, exp_time, plaw_model)
# #%%
#
# emin = 20.0
# emax = 30.0
#
# from pyxsim.utils import ParallelProgressBar, parse_value
#
# emin = parse_value(emin, "keV")
# emax = parse_value(emax, "keV")
#
# emin_src = emin * (1.0)
# emax_src = emax * (1.0)
#
# #xray_fields = plaw_model.make_intensity_fields(ds, emin, emax, dist=(3, "mpc"))
# #print(xray_fields)
# #%%
# #prj = yt.ProjectionPlot(
# #    ds, "z", ("gas", "xray_photon_intensity_20.0_30.0_keV"), width=(1.0, "Mpc")
# #)
# #prj.save()
#
# #%%
#
# ftype = 'gas'#plaw_model.emission_field[0]
# force_override = False
# # dist_fac, redshift = plaw_model._make_dist_fac(
# #             ds, redshift, dist, cosmology, per_sa=True
# #         )
#
# eif = plaw_model.make_fluxf(emin_src, emax_src, energy=True)
#
# def _intensity_field(field, data):
#     ret = data.ds.arr(
#         plaw_model.process_data("energy_field", data, 1.0, fluxf=eif),
#         "keV/s",
#     )
#     idV = data[ftype, "density"] / data[ftype, "mass"]
#     I = 1 * ret * idV
#     return I.in_units("erg/cm**3/s/arcsec**2")
#
# ei_name = (ftype, f"xray_intensity_{emin.value}_{emax.value}_keV")
# ei_dname = rf"I_{{X}} ({emin.value}-{emax.value} keV)"
#
# ds.add_field(
#     ei_name,
#     function=_intensity_field,
#     display_name=ei_dname,
#     sampling_type="local",
#     units="erg/cm**3/s/arcsec**2",
#     force_override=force_override,
# )