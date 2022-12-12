import yt
#yt.toggle_interactivity()
yt.enable_parallelism() # run in console as  mpirun -n 6 python3 xray_imaging.py
# to run on all processor threads: mpirun --use-hwthread-cpus python3 xray_imaging.py
import pyxsim
#import soxs
units_override = {
    "length_unit": (1.5e8, "m"),
    "time_unit": (109.8, "s"),
    "mass_unit": (8.43e38, "kg"),
    #"density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

ds = yt.load("datacubes/flarecs-id.0035.vtk", units_override=units_override, default_species_fields='ionized')
ds.field_list
print("done")
# #%%

#source_model = pyxsim.CIESourceModel("apec")#, emin, emax, nbins, Zmet)
#thermal_model = pyxsim.CIESourceModel("apec", 0.1, 11.0, 10000, 0.3)
thermal_model = pyxsim.CIESourceModel("apec", 0.025, 64.0, 100, 0.3, binscale='log')
#%%
# slc = yt.SlicePlot(ds, 'z', [('athena', 'density'), ('athena', 'total_energy')], width=(0.75, 1))
#%%
# slc.save()
#%%
#source_model = pyxsim.CIESourceModel("spex", 0.05, 11.0, 1000, 0.3, binscale='log')
'''
Cut different regions in yt:
https://yt-project.org/doc/analyzing/objects.html
Find the path of source pyxsim codes:
/home/ivan/anaconda3/envs/radcube/lib/python3.9/site-packages/pyxsim/__init__.py
'''
# #%%
# xray_fields = thermal_model.make_source_fields(ds, 0.5, 64.0)
# print(xray_fields)
# print('done')
# #%%
# #print(ds["gas","xray_luminosity_0.5_7.0_keV"])
# #print(ds.sum(("gas","xray_luminosity_0.5_7.0_keV")))
# #%%
# print('projecting')
# prj = yt.ProjectionPlot(ds, "z", ("gas", "xray_emissivity_0.5_64.0_keV"), width=(0.75, 1))
# prj.save()
#%%
'''
Try to plot an intensity map here
'''
xray_fields = thermal_model.make_intensity_fields(ds, 0.5, 64.0, dist=(1.5e11, "m"))
print(xray_fields)
print('done')
#%%
# Project and save image
prj = yt.OffAxisProjectionPlot(ds, [0.0, 0.0, 1.0], xray_fields[-1],
                           width=(1.5e8, "m"), north_vector=[-0.8, 0.6, 0.0])
prj.save()