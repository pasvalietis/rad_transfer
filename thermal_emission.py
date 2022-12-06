import yt
yt.enable_parallelism()
#yt.toggle_interactivity()
#yt.enable_parallelism()
import pyxsim
import soxs

#%%
ds = yt.load("GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150",
             default_species_fields="ionized")

#%%
slc = yt.SlicePlot(ds, "z", [("gas", "density"), ("gas", "temperature")], width=(1.0,"Mpc"))
#slc.show()
slc.save()

sp = ds.sphere("c", (500.,"kpc"))

source_model = pyxsim.CIESourceModel("spex", 0.05, 11.0, 1000, 0.3, binscale='log')
#%%
xray_fields = source_model.make_source_fields(ds, 0.5, 7.0)
print(xray_fields)
#%%
print(sp["gas","xray_luminosity_0.5_7.0_keV"])
print(sp.sum(("gas","xray_luminosity_0.5_7.0_keV")))
#%%
prj = yt.ProjectionPlot(ds, "z", ("gas", "xray_emissivity_0.5_7.0_keV"), width=(1.0,"Mpc"))
#prj.show()
#%%
prj.save()
#%%
exp_time = (300., "ks") # exposure time
area = (1000.0, "cm**2") # collecting area
redshift = 0.05
#%%
n_photons, n_cells = pyxsim.make_photons("sloshing_photons", sp, redshift, area,
                                         exp_time, source_model)
#%%
n_events = pyxsim.project_photons("sloshing_photons", "sloshing_events",
                                  "z", (45.,30.), absorb_model="tbabs", nH=0.04)
#%%
events = pyxsim.EventList("sloshing_events.h5")
events.write_to_simput("sloshing", overwrite=True)
#%%
soxs.instrument_simulator("sloshing_simput.fits", "evt.fits", (100.0, "ks"),
                          "chandra_acisi_cy0", [45., 30.], overwrite=True)

#%%
soxs.write_image("evt.fits", "img.fits", emin=0.5, emax=2.0, overwrite=True)
#soxs.plot_image("img.fits", stretch='sqrt', cmap='arbre', vmin=0.0, vmax=10.0, width=0.2)
soxs.save()


print('done')
