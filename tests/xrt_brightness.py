# Base imports
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Units
import astropy.units as u
from astropy.coordinates import SkyCoord

from pathlib import Path
import sunpy.map

# Import synthetic image manipulation tools
import yt
sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

export_pm = 'tex'
if export_pm == 'pgf':
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'sans',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'axes.grid': False,
        # 'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 14,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 14,
        'font.size': 14,  # was 10
        # 'legend.fontsize': 6, # was 10
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'text.usetex': True,
        'figure.figsize': [3.39, 2.10],
        'font.family': 'serif',
    })

#%%
if export_pm == 'tex':
    import matplotlib
    params = {
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        #'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 14, # fontsize for x and y labels (was 10)
        'axes.titlesize': 14,
        'font.size': 14, # was 10
        #'legend.fontsize': 6, # was 10
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'text.usetex': True,
        'figure.figsize': [3.39, 2.10],
        'font.family': 'sans',
    }
    matplotlib.rcParams.update(params)

#%%
'''
Import XRT image
'''

directory = "/home/ivan/Study/Astro/solar/rad_transfer/datasets/hinode/xrt/level1/"
data_file = Path(directory) / "L1_XRT20211102_033747.1.fits" # "L1_XRT20110307_180930.8.fits"

xrt_map = sunpy.map.Map(data_file, autoalign=True)
xrt_map.meta.update({'hgln_obs': 0})  # Add missing keywords to metadata https://community.openastronomy.org/t/sunpymetadatawarnings-when-using-hinode-xrt-data/393

xrt_map.data[xrt_map.data <= 0] = 1.

# roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=200*u.arcsec, frame=xrt_map.coordinate_frame)
# #width_Mm = (np.abs(-500+150)*asec2cm*u.cm).to(u.Mm).value
# roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=550*u.arcsec, frame=xrt_map.coordinate_frame)
# #height_Mm =  (np.abs(500-150)*asec2cm*u.cm).to(u.Mm).value
cusp_submap = xrt_map #.submap(roi_bottom_left, top_right=roi_top_right)

# fill blank values
cusp_submap.data[cusp_submap.data < 0] = cusp_submap.data.min()+20


#%%
'''
Import dataset
'''

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path)  # , hint='AthenaDataset', units_override=units_override)
cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
instr = 'xrt'
channel = 'Be-thin'
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep


# for i in range(1):

xrt_synthetic = synt_img(cut_box, instr, channel)

# Match parameters of the synthetic image to observed one
samp_resolution = cusp_submap.data.shape[0]
# Export physical size from the observations
scale = 1.0
obs_scale = [scale*cusp_submap.scale.axis1, scale*cusp_submap.scale.axis2] * (u.arcsec / u.pixel)
reference_pixel = u.Quantity([cusp_submap.reference_pixel[0].value,
                              cusp_submap.reference_pixel[1].value], u.pixel)
reference_coord = cusp_submap.reference_coordinate

img_tilt = 110 * u.deg

synth_plot_settings = {'resolution': samp_resolution}
synth_view_settings = {'normal_vector':  [0.1, 0.05, 0.92], #  [0.1*(i+1), 0.08, np.sqrt(1.-(0.1*(i+1)))], #[0.08, 0.08, 0.916],
                       'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}

xrt_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                            view_settings=synth_view_settings,
                            image_shift=[140, -40],
                            image_zoom=0.75,
                            bkg_fill=10)

# Import scale from an AIA image:
synth_map = xrt_synthetic.make_synthetic_map(obstime='2013-10-28',
                                             observer='earth',
                                             detector='Synthetic XRT',
                                             scale=obs_scale,
                                             reference_coord=reference_coord,
                                             reference_pixel=reference_pixel)  # .rotate(angle=0.0 * u.deg)

# Plot obs and synthetic image

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
fig.set_tight_layout(True)
plt.ioff()
plt.style.use('fast')
gs = fig.add_gridspec(3, 4)

# *****************
ax1 = fig.add_subplot(gs[:2, :2], projection=cusp_submap)
cusp_submap.plot_settings['norm'] = colors.LogNorm(10, cusp_submap.max())
# stonyhurst_grid = cusp_submap.draw_grid(axes=ax1, system='stonyhurst')
cusp_submap.plot(axes=ax1)
ax1.grid(False)
stonyhurst_grid = cusp_submap.draw_grid(axes=ax1, system='stonyhurst', annotate=False)
cusp_submap.draw_limb()

'''
Initialize slits
'''

line_coords = SkyCoord([850, 915], [-530, -550], unit=(u.arcsec, u.arcsec),
                       frame=cusp_submap.coordinate_frame)  # [x1, x2], [y1, y2]
intensity_coords = sunpy.map.pixelate_coord_path(cusp_submap, line_coords)
intensity = sunpy.map.sample_at_coords(cusp_submap, intensity_coords)

line_coords_ = SkyCoord([850, 915], [-530, -550], unit=(u.arcsec, u.arcsec),
                        frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
intensity_coords_ = sunpy.map.pixelate_coord_path(synth_map, line_coords_)
intensity_ = sunpy.map.sample_at_coords(synth_map, intensity_coords_)

line_coords2 = SkyCoord([860, 890], [-580, -480], unit=(u.arcsec, u.arcsec),
                       frame=cusp_submap.coordinate_frame)  # [x1, x2], [y1, y2]
intensity_coords2 = sunpy.map.pixelate_coord_path(cusp_submap, line_coords2)
intensity2 = sunpy.map.sample_at_coords(cusp_submap, intensity_coords2)

line_coords2_ = SkyCoord([860, 890], [-580, -480], unit=(u.arcsec, u.arcsec),
                        frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
intensity_coords2_ = sunpy.map.pixelate_coord_path(synth_map, line_coords2_)
intensity2_ = sunpy.map.sample_at_coords(synth_map, intensity_coords2_)

angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
angular_separation_ = intensity_coords_.separation(intensity_coords_[0]).to(u.arcsec)

angular_separation2 = intensity_coords2.separation(intensity_coords2[0]).to(u.arcsec)
angular_separation2_ = intensity_coords2_.separation(intensity_coords2_[0]).to(u.arcsec)
# Plot brightness profiles
ax1.plot_coord(intensity_coords, color='limegreen', linewidth=0.75, label='XRT')
ax1.plot_coord(intensity_coords2, color='blue', linewidth=0.75, label='XRT')

# *****************
ax2 = fig.add_subplot(gs[:2, 2:], projection=synth_map)
synth_map.plot_settings['norm'] = colors.LogNorm(10, cusp_submap.max())
synth_map.plot_settings['cmap'] = xrt_map.plot_settings['cmap']
synth_map.plot(axes=ax2)
ax2.grid(False)
stonyhurst_grid = synth_map.draw_grid(axes=ax2, system='stonyhurst', annotate=False)
synth_map.draw_limb()

ax2.text(120, 30,
         'norm: ' + str(list(float(i) for i in ["%.2f" % elem for elem in synth_view_settings['normal_vector']])) +
         '\n' + 'north: ' + str(
             list(float(i) for i in ["%.2f" % elem for elem in synth_view_settings['north_vector']])),
         style='italic', color='white')

ax2.plot_coord(intensity_coords_, color='limegreen', linewidth=0.75)
ax2.plot_coord(intensity_coords2_, color='blue', linewidth=0.75)

plt.colorbar()

# *****************
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(angular_separation, intensity, linewidth=0.65, color='limegreen')
ax3.plot(angular_separation_, intensity_, linewidth=0.65, color='limegreen', linestyle='--')
ax3.plot(angular_separation2, intensity2, linewidth=0.65, color='blue', label='XRT')
ax3.plot(angular_separation2_, intensity2_, linewidth=0.65, color='blue', linestyle='--', label='Synthetic')
ax3.set_ylabel('$I$, [DN cm$^5$ pix$^{-1}$ s$^{-1}$]')
ax3.set_xlabel("Angular distance along slit [arcsec]")
ax3.set_yscale('log')
ax3.legend(frameon=False, fontsize=10, loc ="upper right")
#ax3.legend(frameon=False)
#a3.text()


plt.savefig('xrt_brightness_profile_'+str(0)+'.pgf') #+str(export_pm))














