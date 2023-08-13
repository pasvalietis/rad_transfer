import numpy as np
import sys
import os.path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.units as u
from astropy.coordinates import SkyCoord

import scipy.ndimage as ndimage
import sunpy.map

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.tests.xray_debug import proj_and_imag
from rad_transfer.emission_models import uv
from rad_transfer.visualization.colormaps import color_tables

# Import synthetic image manipulation tools
import yt
sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

export_pm = 'pgf'

if export_pm == 'pgf':
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'sans',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'axes.grid': False,
        'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 10,
        'font.size': 10,  # was 10
        # 'legend.fontsize': 6, # was 10
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'figure.figsize': [3.39, 2.10],
        'font.family': 'sans',
    })




'''
Plot gradient field of velocity in order to find the minimum of div(v)
'''

# Start with exporting the velocity field from a subsampled datacube
#%% Load input data
downs_factor = 3
original_file_path = '../datacubes/flarecs-id.0035.vtk'
#downs_file_path = './subs_dataset_' + str(downs_factor) + '.h5'
downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'

L_0 = (1.5e10, "cm")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

# if not os.path.isfile(downs_file_path):
#     ds = yt.load(original_file_path, units_override=units_override,
#                  default_species_fields='ionized', hint='AthenaDataset')
#     # Specifying default_species_fields is required to produce emission_measure field so that PyXSIM thermal
#     # emission model can be applied
#     rad_buffer_obj = downsample(ds, rad_fields=True, n=downs_factor)
# else:

rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")

cut_box = rad_buffer_obj.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

'''
Import AIA map
'''
aia_imag_lvl1 = sunpy.map.Map('./aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits')
aia_map = aia_imag_lvl1

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=200*u.arcsec, frame=aia_map.coordinate_frame)
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=550*u.arcsec, frame=aia_map.coordinate_frame)
cusp_submap = aia_map.submap(roi_bottom_left, top_right=roi_top_right)
cusp_submap.data[cusp_submap.data <= 0] = cusp_submap.data.min()+20

#%%
# Plot the projection of the velocity divergence
# add transparent colormap to further overlay it over the 93 A image

ncolors = 256
color_array = plt.get_cmap('RdPu_r')(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name='inferno_alpha', colors=color_array)
plt.register_cmap(cmap=map_object)

color_array = plt.get_cmap('Reds')(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name='inferno_beta', colors=color_array)
plt.register_cmap(cmap=map_object)

instr = 'aia'  # aia or xrt
channel = 131
sdo_aia_model = uv.UVModel('temperature', 'density', channel)
sdo_aia_model.make_intensity_fields(rad_buffer_obj)

timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

aia_synthetic = synt_img(cut_box, instr, channel)
samp_resolution = 584
obs_scale = [0.6, 0.6]*(u.arcsec/u.pixel)
reference_pixel = u.Quantity([833.5, -333.5], u.pixel)
reference_coord = cusp_submap.reference_coordinate

img_tilt = 23*u.deg

synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector': [-0.12, 0.05, 0.916],
                       'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}

'''
img_tilt = -23*u.deg

synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector': [0.12, 0.05, 0.916],
                       'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}
'''

aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                            view_settings=synth_view_settings,
                            image_shift=[-52, 105],
                            bkg_fill=10)

synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28',
                                             observer='earth',
                                             detector='Synthetic AIA',
                                             scale=obs_scale,
                                             reference_coord=reference_coord,
                                             reference_pixel=reference_pixel)  # .rotate(angle=0.0 * u.deg)

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
#for i in range(nframes):
#norm_vec = [1.0 - i*(1./nframes), 0.0+i*(1./(nframes*3.)), i*(1./nframes)]
norm_vec = synth_view_settings['normal_vector']  # [0.12, 0.1, 1.0]
north_vector = synth_view_settings['north_vector']  # [0.3, 0.7, 0.0]
center_pos = [0.0, 0.5, 0.0]

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
#fig, ax = plt.subplots()
xsize, ysize = 5.5, 4.5
fig, ax = plt.subplots(1, 1, figsize=(xsize, ysize), dpi=140)
data_img = np.array(prji)
imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale
divvmap = np.array(divvprj)
antidivvmap = np.array(antidivprj)

vmin=10
vmax=914.910
imag[imag == 0] = vmin

pcm = ax.pcolor(X, Y, imag,
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                        #vmin=1e-5,
                        #vmax=1.5e4,
                        cmap=imgcmap, shading='auto', rasterized=True)

ax.pcolor(X, Y, divvmap,
          #norm=colors.LogNorm(vmin=1e6, vmax=2.5e8),
          vmin=5e7,
          vmax=2e8,
          cmap='inferno_alpha', shading='auto', rasterized=True)

ax.pcolor(X, Y, antidivvmap,
          #norm=colors.LogNorm(vmin=1e6, vmax=2.5e8),
          vmin=3e7,
          vmax=7e7,
          cmap='inferno_beta', shading='auto', rasterized=True)

ax.annotate("Y Point, div v > 0",
            xy=(0, 70), xycoords='data', color='magenta',
            xytext=(20, 90), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='magenta'))

ax.annotate("X Point, div v > 0",
            xy=(-21, 145), xycoords='data', color='magenta',
            xytext=(20, 130), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='magenta'))

ax.annotate("TS, div v < 0",
            xy=(0, 85), xycoords='data', color='red',
            xytext=(20, 110), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'))

int_units = str(prji.units)
#fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
cbar = fig.colorbar(pcm, ax=ax, extend='max')
cbar.set_label(label='DN cm$^5$ pix$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
ax.set_xlabel('$x$, Mm')
ax.set_ylabel('$y$, Mm')
ax.set_aspect('equal')

#figpath = '../img/rad_tr_thermal_brem/'
#plt.savefig(figpath + 'therm_brem_front_view_rad_buff.png')

ax.set_title('$div(v)$') #'Synthetic AIA '+str(channel)+' Å')
#plt.show()
figpath = '/home/ivan/Study/Astro/solar/rad_transfer/tests/imag/velocity_field/'
plt.savefig(figpath + 'sdo_aia_'+str(channel)+'_mov_'+'.pgf', dpi=140)  # +str(i).zfill(3)