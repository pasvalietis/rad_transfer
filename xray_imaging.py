import yt
import pyxsim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
#yt.toggle_interactivity()
from add_emissivity_field import ThermalBremsstrahlungModel
#%%
yt.enable_parallelism()
'''
run in console as  mpirun -n 6 python3 xray_imaging.py
to run on all processor threads: mpirun --use-hwthread-cpus python3 xray_imaging.py
'''

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    # "density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

ds = yt.load("datacubes/flarecs-id.0035.vtk", units_override=units_override, default_species_fields='ionized')
ds.field_list

#%%
emin = 6.0
emax = 6.1
# source_model = pyxsim.CIESourceModel(atomic_db_model_name, emin, emax, nbins, Zmet)
thermal_model = pyxsim.CIESourceModel("apec", emin, emax, 100, 0.2, binscale='log')
#%%
xray_fields = thermal_model.make_intensity_fields(ds, emin, emax, dist=(1.5e11, "m"))
print(xray_fields)
#%%
# Project and save image
#prj = yt.OffAxisProjectionPlot(ds, [0.0, 0.0, 1.0], xray_fields[-1],
#                           width=(1.5e8, "m"))#, north_vector=[-0.8, 0.6, 0.0])
"""
yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(data_source,
center, normal_vector, width, resolution, item, weight=None, volume=None, no_ghost=False, 
interpolated=False, north_vector=None, num_threads=1, method='integrate')
"""
#%%
# Resolution
N = 512
norm_vec = [0.0, 0.0, 1.0]
prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(ds,
                        [0.0, 0.5, 0.0],  # center position in code units
                        norm_vec,  # normal vector (z axis)
                        1.0,  # width in code units
                        N,  # image resolution
                        xray_fields[-1],  # respective field that is being projected
                        north_vector=[0.0, 1.0, 0.0]
                        )

Mm_len = ds.length_unit.to('Mm').value

X, Y = np.mgrid[-0.5*Mm_len:0.5*Mm_len:complex(0, N),
       0*Mm_len:Mm_len:complex(0, N)]

fig, ax = plt.subplots()
data_img = np.array(prji)
imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale

pcm = ax.pcolor(X, Y, imag,
                       #norm=colors.LogNorm(vmin=1e-1, vmax=imag.max()),
                       cmap='inferno', shading='auto')
int_units = str(prji.units)
fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
ax.set_xlabel('x, Mm')
ax.set_ylabel('y, Mm')

plt.show()
plt.savefig('mpl_img.png')


data_img = np.array(prji)
print(data_img.shape)
im = plt.imshow(np.rot90(data_img, k=1, axes=(0,1)), cmap='inferno', norm=colors.LogNorm())
plt.colorbar(im)
plt.show()
plt.imshow(np.rot90(data_img, k=1, axes=(0,1)), cmap='inferno')
#plt.show()
# prj.save()
plt.savefig('isometric_view.png')
#%%
# Plot temperature distribution
# prj_T = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(ds,
#                         [0.0, 0.5, 0.0],  # center position in code units
#                         [0.0, 0.0, 1.0],  # normal vector (z axis)
#                         1.0,  # width in code units
#                         N,  # image resolution
#                         ('gas', 'temperature'),  # respective field that is being projected
#                         )
#
# Mm_len = ds.length_unit.to('Mm').value
#
# X, Y = np.mgrid[-0.5*Mm_len:0.5*Mm_len:complex(0, N),
#        0*Mm_len:Mm_len:complex(0, N)]
#
# fig, ax = plt.subplots()
# imag = np.array(prj_T) #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale
#
# pcm = ax.pcolor(X, Y, imag,
#                        #norm=colors.LogNorm(vmin=1e-64, vmax=1e-62),
#                        cmap='inferno', shading='auto')
# int_units = str('K')
# fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
# ax.set_xlabel('x, Mm')
# ax.set_ylabel('y, Mm')
#
# plt.show()
#%%
# Produce EM-weighted temperature maps
# Add emission measure (EM) field (f1)

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
# Add EM*temperature field (f2)
def _em_scaled_temperature(field, data):
    return (
        data["gas", "temperature"] * data["gas", "density"]**2.
    )

ds.add_field(
    name=("gas", "em_scaled_temperature"),
    function=_em_scaled_temperature,
    sampling_type="local",
    units="g**2*K/cm**6",
    force_override=True
)
#%%
# Calculate the z projections of both fields
prj_EM = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(ds,
                        [0.0, 0.5, 0.0],  # center position in code units
                        norm_vec,  # normal vector (z axis)
                        1.0,  # width in code units
                        N,  # image resolution
                        ('gas', 'em_field'),  # respective field that is being projected
                        north_vector=[0.0, 1.0, 0.0]
                        )

prj_EMT = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(ds,
                        [0.0, 0.5, 0.0],  # center position in code units
                        norm_vec,  # normal vector (z axis)
                        1.0,  # width in code units
                        N,  # image resolution
                        ('gas', 'em_scaled_temperature'),  # respective field that is being projected
                        north_vector=[0.0, 1.0, 0.0]
                        )
#%%
# Find and plot the ratio
w_temp = prj_EMT / prj_EM
weighted_temp = np.array(w_temp)
#%%

fig, ax = plt.subplots()
imag_temp = weighted_temp #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale

# pcm = ax.pcolor(X, Y, imag_temp,
#                        #norm=colors.LogNorm(vmin=1e-64, vmax=1e-62),
#                        cmap='inferno', shading='auto')
# int_units = str('K')
# fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
# ax.set_xlabel('x, Mm')
# ax.set_ylabel('y, Mm')

# plt.show()
#plt.savefig('em_weighted_temp.pdf')

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

pcm = ax[0].pcolor(X, Y, imag,
                       #norm=colors.LogNorm(vmin=1e-1, vmax=imag.max()),
                        vmin=1e-1,
                        vmax=2,
                       cmap='inferno', shading='auto', rasterized=True)
int_units = str(prji.units)
fig.colorbar(pcm, ax=ax[0], extend='max', label='$'+int_units.replace("**", "^")+'$')
ax[0].set_xlabel('x, Mm')
ax[0].set_ylabel('y, Mm')

pcm = ax[1].pcolor(X, Y, imag_temp,
                       #norm=colors.LogNorm(vmin=1.e4, vmax=1.e8),
                       cmap='inferno', shading='auto', rasterized=True)
int_units = str('K')
fig.colorbar(pcm, ax=ax[1], extend='max', label='$'+int_units.replace("**", "^")+'$')
ax[1].set_xlabel('x, Mm')
ax[1].set_ylabel('y, Mm')

fig.subplots_adjust(top=0.94,
                           bottom=0.11,
                           left=0.07,
                           right=0.99,
                           hspace=0.2,
                           wspace=0.25)

#plt.show()
plt.savefig('plots_xray.png')
