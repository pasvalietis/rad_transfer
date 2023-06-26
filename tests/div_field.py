import yt
import numpy as np
import sys
import os.path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.units as u

import scipy.ndimage as ndimage

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.buffer import downsample
from rad_transfer.tests.xray_debug import proj_and_imag
from rad_transfer.emission_models import uv
from rad_transfer.visualization.colormaps import color_tables

'''
Plot gradient field of velocity in order to find the minimum of div(v)
'''

# Start with exporting the velocity field from a subsampled datacube
#%% Load input data
downs_factor = 3
original_file_path = '../datacubes/flarecs-id.0035.vtk'
downs_file_path = './subs_dataset_' + str(downs_factor) + '.h5'

L_0 = (1.5e10, "cm")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

if not os.path.isfile(downs_file_path):
    ds = yt.load(original_file_path, units_override=units_override,
                 default_species_fields='ionized', hint='AthenaDataset')
    # Specifying default_species_fields is required to produce emission_measure field so that PyXSIM thermal
    # emission model can be applied
    rad_buffer_obj = downsample(ds, rad_fields=True, n=downs_factor)
else:
    rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")

cut_box = rad_buffer_obj.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])
#%%
# Plot the projection of the velocity divergence

# add transparent colormap to further overlay it over the 93 A image
ncolors = 256
color_array = plt.get_cmap('cividis')(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name='inferno_alpha', colors=color_array)
plt.register_cmap(cmap=map_object)

color_array = plt.get_cmap('Reds')(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name='inferno_beta', colors=color_array)
plt.register_cmap(cmap=map_object)

channel = 'A131'
sdo_aia_model = uv.UVModel('temperature', 'density', channel)
sdo_aia_model.make_intensity_fields(rad_buffer_obj)

#cmap1 = matplotlib.colors.ListedColormap(['none', 'red'])
#%%
def _convergence(field, data):
    norm = yt.YTQuantity(1.0, "s")
    return (-norm*data["gas", "velocity_divergence"])

def _divergence(field, data):
    norm = yt.YTQuantity(1.0, "s")
    return (norm * data["gas", "velocity_divergence"])

rad_buffer_obj.add_field(
    name=('gas', 'convergence'),
    function=_convergence,
    sampling_type="local",
    units='dimensionless',
    force_override=True,
)

rad_buffer_obj.add_field(
    name=('gas', 'divergence'),
    function=_divergence,
    sampling_type="local",
    units='dimensionless',
    force_override=True,
)
#%%
# nframes = 5
# for frame in range(nframes):
#     imag = proj_and_imag(rad_buffer_obj, 'velocity_divergence',
#         norm_vec=[1.0 - frame*(1./nframes),
#                 0.0 + frame*(1./(nframes*3.)),
#                 0.0 + frame * (1./nframes)],
#         vmin=2.5e7, vmax=5e7,
#         cmap='inferno_alpha', figpath='../img/velocity_field/', logscale=False, frame=frame)

# #%%
# slc = yt.SlicePlot(rad_buffer_obj, "z", ("gas", "velocity_divergence"))
# slc.set_log(("gas", "velocity_divergence"), False)
# slc.save('div_slice.png')
#%%
#
# def local_minimum(field, data):
#     ftype = 'gas'
#     img2 = ndimage.minimum_filter(data[ftype, 'velocity_divergence'], size=(5, 5, 5))
#     #tr = np.minimum(np.minimum(t1, t2), t3)
#     return img2
#
# local_minimum(rad_buffer_obj, 'velocity_divergence')
#%%
N = 512
nframes = 1
imgcmap = color_tables.aia_color_table(131*u.angstrom)
plt.ioff()
for i in range(nframes):
    #norm_vec = [1.0 - i*(1./nframes), 0.0+i*(1./(nframes*3.)), i*(1./nframes)]
    norm_vec = [0.12, 0.1, 1.0]
    north_vector = [0.3, 0.7, 0.0]
    center_pos = [0.0, 0.45, 0.0]

    prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
                            cut_box,
                            center_pos,  # center position in code units
                            norm_vec,  # normal vector (z axis)
                            1.0,  # width in code units
                            N,  # image resolution
                            'aia_filter_band',  # respective field that is being projected
                            north_vector=north_vector)

    divvprj = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
                            cut_box,
                            center_pos,  # center position in code units
                            norm_vec,  # normal vector (z axis)
                            1.0,  # width in code units
                            N,  # image resolution
                            'convergence',  # respective field that is being projected
                            north_vector=north_vector)

    antidivprj = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
                            cut_box,
                            center_pos,  # center position in code units
                            norm_vec,  # normal vector (z axis)
                            1.0,  # width in code units
                            N,  # image resolution
                            'divergence',  # respective field that is being projected
                            north_vector=north_vector)

    Mm_len = 1 # ds.length_unit.to('Mm').value

    X, Y = np.mgrid[-0.5*150*Mm_len:0.5*150*Mm_len:complex(0, N),
           0*Mm_len:150*Mm_len:complex(0, N)]

    #%%
    fig, ax = plt.subplots()
    data_img = np.array(prji)
    imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale
    divvmap = np.array(divvprj)
    antidivvmap = np.array(antidivprj)

    vmin=1
    vmax=300
    imag[imag == 0] = vmin

    pcm = ax.pcolor(X, Y, imag,
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                            #vmin=1e-5,
                            #vmax=1.5e4,
                            cmap=imgcmap, shading='auto')

    ax.pcolor(X, Y, divvmap,
              #norm=colors.LogNorm(vmin=1e5, vmax=2.5e7),
              vmin=2.5e7,
              vmax=5e7,
              cmap='inferno_alpha', shading='auto')

    ax.pcolor(X, Y, antidivvmap,
              #norm=colors.LogNorm(vmin=1e5, vmax=2.5e7),
              vmin=2.5e7,
              vmax=5e7,
              cmap='inferno_beta', shading='auto')
    int_units = str(prji.units)
    #fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
    cbar = fig.colorbar(pcm, ax=ax, extend='max')
    cbar.set_label(label='DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
    ax.set_xlabel('x, Mm')
    ax.set_ylabel('y, Mm')

    #figpath = '../img/rad_tr_thermal_brem/'
    #plt.savefig(figpath + 'therm_brem_front_view_rad_buff.png')

    ax.set_title('Synthetic AIA '+channel[1:]+' Å')
    #plt.show()
    figpath = '../img/velocity_field/mov_'+channel+'/'
    plt.savefig(figpath + 'sdo_aia_'+channel+'_mov_'+str(i).zfill(3)+'.png')