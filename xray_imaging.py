import yt
#yt.toggle_interactivity()
yt.enable_parallelism() # run in console as  mpirun -n 6 python3 xray_imaging.py
import pyxsim
#import soxs

ds = yt.load("datacubes/flarecs-id.0035.vtk", default_species_fields='ionized')
ds.field_list
#source_model = pyxsim.CIESourceModel("apec")#, emin, emax, nbins, Zmet)
#thermal_model = pyxsim.CIESourceModel("apec", 0.1, 11.0, 10000, 0.3)
thermal_model = pyxsim.CIESourceModel("apec", 0.025, 64.0, 1000, 1, binscale='log')
#%%
#slc = yt.SlicePlot(ds, 'z', [('athena', 'density'), ('athena', 'total_energy')], width=(0.75, 1))
#slc.save()
#%%
#source_model = pyxsim.CIESourceModel("spex", 0.05, 11.0, 1000, 0.3, binscale='log')
#%%
xray_fields = thermal_model.make_source_fields(ds, 0.5, 64.0)
print(xray_fields)
print('done')
#%%
#print(ds["gas","xray_luminosity_0.5_7.0_keV"])
#print(ds.sum(("gas","xray_luminosity_0.5_7.0_keV")))
#%%
print('projecting')
prj = yt.ProjectionPlot(ds, "z", ("gas", "xray_emissivity_0.5_64.0_keV"), width=(0.75, 1))
prj.save()

print('done')