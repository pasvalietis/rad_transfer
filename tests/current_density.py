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

from yt.fields.vector_operations import create_vector_fields
from yt.fields.fluid_vector_fields import setup_fluid_vector_fields
sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.tests.xray_debug import proj_and_imag
from rad_transfer.emission_models import uv
from rad_transfer.visualization.colormaps import color_tables



# Import synthetic image manipulation tools
import yt
sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt
 #%%
downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")  # , hint='AthenaDataset', units_override=units_override)
cut_box = rad_buffer_obj.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
grad_fields_x = rad_buffer_obj.add_gradient_fields(("gas", "magnetic_field_x"))
grad_fields_y = rad_buffer_obj.add_gradient_fields(("gas", "magnetic_field_y"))
grad_fields_z = rad_buffer_obj.add_gradient_fields(("gas", "magnetic_field_z"))
#%%
def _current_density(field, data):
    norm = yt.YTQuantity(1.0, "s")
    j_x = data["gas", "magnetic_field_z_gradient_y"] - data["gas", "magnetic_field_y_gradient_z"]
    j_y = data["gas", "magnetic_field_x_gradient_z"] - data["gas", "magnetic_field_z_gradient_x"]
    j_z = data["gas", "magnetic_field_y_gradient_x"] - data["gas", "magnetic_field_x_gradient_y"]
    return 4 * np.pi * np.sqrt(j_x**2 + j_y**2 + j_z**2) / (yt.physical_constants.c * norm)
#%%
rad_buffer_obj.add_field(
    ("gas", "current_density"),
    function=_current_density,
    sampling_type="cell",
    units="G/cm**2",
    take_log=False,
    force_override=True
    # validators=[ValidateParameter(["center", "bulk_velocity"])],
)
#%%
'''
Import AIA map
'''
aia_imag_lvl1 = sunpy.map.Map('./aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits')
aia_map = aia_imag_lvl1

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=200*u.arcsec, frame=aia_map.coordinate_frame)
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=550*u.arcsec, frame=aia_map.coordinate_frame)
cusp_submap = aia_map.submap(roi_bottom_left, top_right=roi_top_right)

instr = 'aia'  # aia or xrt
channel = 131

aia_synthetic = synt_img(cut_box, instr, channel)
samp_resolution = 584
obs_scale = [0.6, 0.6]*(u.arcsec/u.pixel)
reference_pixel = u.Quantity([833.5, -333.5], u.pixel)
reference_coord = cusp_submap.reference_coordinate
img_tilt = 23*u.deg

synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector': [-0.12, 0.05, 0.916],
                       'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}
#%%
aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                            view_settings=synth_view_settings,
                            image_shift=[-52, 105],
                            bkg_fill=10)
#%%
synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28',
                                             observer='earth',
                                             detector='Synthetic AIA',
                                             scale=obs_scale,
                                             reference_coord=reference_coord,
                                             reference_pixel=reference_pixel)

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

j_proj = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
                        cut_box,
                        center_pos,  # center position in code units
                        norm_vec,  # normal vector (z axis)
                        1.0,  # width in code units
                        N,  # image resolution
                        'current_density',  # respective field that is being projected
                        north_vector=north_vector)

Mm_len = 1  # ds.length_unit.to('Mm').value

X, Y = np.mgrid[-0.5*150*Mm_len:0.5*150*Mm_len:complex(0, N),
       0*Mm_len:150*Mm_len:complex(0, N)]

#%%
#fig, ax = plt.subplots()
xsize, ysize = 5.5, 4.5
fig, ax = plt.subplots(1, 1, figsize=(xsize, ysize), dpi=140)
data_img = np.array(prji)
imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale
j_map = np.array(j_proj)

vmin=10
vmax=914.910
imag[imag == 0] = vmin

pcm = ax.pcolor(X, Y, imag,
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                        #vmin=1e-5,
                        #vmax=1.5e4,
                        cmap=imgcmap, shading='auto', rasterized=True)

ax.pcolor(X, Y, j_map,
          #norm=colors.LogNorm(vmin=1e6, vmax=2.5e8),
          #vmin=5e7,
          #vmax=2e8,
          cmap='inferno', shading='auto', rasterized=True)

cbar = fig.colorbar(pcm, ax=ax, extend='max')
cbar.set_label(label='DN cm$^5$ pix$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
ax.set_xlabel('$x$, Mm')
ax.set_ylabel('$y$, Mm')
ax.set_aspect('equal')

ax.set_title('Current density j')
figpath = '/home/ivan/Study/Astro/solar/rad_transfer/img/current_density/'
plt.savefig(figpath + 'sdo_aia_'+str(channel)+'.png', dpi=140)
