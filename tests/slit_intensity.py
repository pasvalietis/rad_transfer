
# Base imports
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Units
import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map

#PREP LEVEL 1.5 IMAGES
# from aiapy.calibrate import normalize_exposure, register, update_pointing

# Import synthetic image manipulation tools
import yt
sys.path.insert(0, '/home')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

export_pm = 'tex'

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
Import AIA image
'''

aia_imag_lvl1 = sunpy.map.Map('./aia_data/aia.lev1.131A_2011-03-07T180909.62Z.image_lev1.fits')

m_updated_pointing = update_pointing(aia_imag_lvl1)
m_registered = register(m_updated_pointing)
aia_imag = normalize_exposure(m_registered)

aia_map = aia_imag # sunpy.map.Map(aia_imag, autoalign=True)
aia_rotated = aia_map #.rotate(angle=0.0 * u.deg)

roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=200*u.arcsec, frame=aia_map.coordinate_frame)
#width_Mm = (np.abs(-500+150)*asec2cm*u.cm).to(u.Mm).value
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=550*u.arcsec, frame=aia_map.coordinate_frame)
#height_Mm =  (np.abs(500-150)*asec2cm*u.cm).to(u.Mm).value
cusp_submap = aia_map.submap(roi_bottom_left, top_right=roi_top_right)
cusp_submap.data[cusp_submap.data <= 0] = cusp_submap.data.min()+20

if __name__ == "__main__":
    #%%
    '''
    Import dataset
    '''

    # L_0 = (1.5e8, "m")
    # units_override = {
    #     "length_unit": L_0,
    #     "time_unit": (109.8, "s"),
    #     "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    #     "velocity_unit": (1.366e6, "m/s"),
    #     "temperature_unit": (1.13e8, "K"),
    # }

    downs_file_path = '/home/saber/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
    subs_ds = yt.load(downs_file_path)  # , hint='AthenaDataset', units_override=units_override)
    cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

    instr = 'aia'  # aia or xrt
    channel = 131
    timestep = downs_file_path[-7:-3]
    fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

    aia_synthetic = synt_img(cut_box, instr, channel)
    # Match parameters of the synthetic image to observed one
    samp_resolution = cusp_submap.data.shape[0]
    obs_scale = [cusp_submap.scale.axis1, cusp_submap.scale.axis2]*(u.arcsec/u.pixel)
    reference_pixel = u.Quantity([cusp_submap.reference_pixel[0].value,
                                  cusp_submap.reference_pixel[1].value], u.pixel)
    reference_coord = cusp_submap.reference_coordinate

    img_tilt = -23*u.deg

    synth_plot_settings = {'resolution': samp_resolution}
    synth_view_settings = {'normal_vector': [0.12, 0.05, 0.916],
                           'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}

    aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[-52, 105],
                                bkg_fill=10)

    #Import scale from an AIA image:
    synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28',
                                                 observer='earth',
                                                 detector='Synthetic AIA',
                                                 scale=obs_scale,
                                                 reference_coord=reference_coord,
                                                 reference_pixel=reference_pixel)  # .rotate(angle=0.0 * u.deg)


    #  Update physical scaling (only arcsec/pix right now)
    #synth_map.scale._replace(axis1=cusp_submap.scale.axis1, axis2=cusp_submap.scale.axis2)

    '''
    Plot obs and synthetic image
    '''

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #%%
    #fig = plt.figure(figsize=(10, 4))
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    fig.set_tight_layout(True)
    plt.ioff()
    plt.style.use('fast')
    gs = fig.add_gridspec(3, 4)

    #ax1 = fig.add_subplot(212, projection=cusp_submap)
    ax1 = fig.add_subplot(gs[:2, :2], projection=cusp_submap)
    cusp_submap.plot_settings['norm'] = colors.LogNorm(10, cusp_submap.max())
    #cusp_submap.plot_settings['cmap'] =
    cusp_submap.plot(axes=ax1)
    ax1.grid(False)
    stonyhurst_grid = cusp_submap.draw_grid(axes=ax1, system='stonyhurst', annotate=False)
    cusp_submap.draw_limb()


    line_coords = SkyCoord([-335, -380], [380, 480], unit=(u.arcsec, u.arcsec),
                           frame=cusp_submap.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords = sunpy.map.pixelate_coord_path(cusp_submap, line_coords)
    intensity = sunpy.map.sample_at_coords(cusp_submap, intensity_coords)

    line_coords_ = SkyCoord([-335, -380], [380, 480], unit=(u.arcsec, u.arcsec),
                           frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords_ = sunpy.map.pixelate_coord_path(synth_map, line_coords_)
    intensity_ = sunpy.map.sample_at_coords(synth_map, intensity_coords_)

    line_coords2 = SkyCoord([-390, -310], [400, 435], unit=(u.arcsec, u.arcsec),
                           frame=cusp_submap.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords2 = sunpy.map.pixelate_coord_path(cusp_submap, line_coords2)
    intensity2 = sunpy.map.sample_at_coords(cusp_submap, intensity_coords2)

    line_coords2_ = SkyCoord([-390, -310], [400, 435], unit=(u.arcsec, u.arcsec),
                           frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords2_ = sunpy.map.pixelate_coord_path(synth_map, line_coords2_)
    intensity2_ = sunpy.map.sample_at_coords(synth_map, intensity_coords2_)

    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    angular_separation_ = intensity_coords_.separation(intensity_coords_[0]).to(u.arcsec)

    angular_separation2 = intensity_coords2.separation(intensity_coords2[0]).to(u.arcsec)
    angular_separation2_ = intensity_coords2_.separation(intensity_coords2_[0]).to(u.arcsec)

    ax1.plot_coord(intensity_coords, color='magenta', linewidth=0.75)
    ax1.plot_coord(intensity_coords2, color='red', linewidth=0.75)

    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes('right', size='5%', pad=0.05)

    #ax2 = fig.add_subplot(221, projection=synth_map)
    ax2 = fig.add_subplot(gs[:2, 2:], projection=synth_map)
    #cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)
    synth_map.plot_settings['norm'] = colors.LogNorm(10, cusp_submap.max())
    synth_map.plot_settings['cmap'] = aia_map.plot_settings['cmap']

    synth_map.plot(axes=ax2)
    ax2.grid(False)
    stonyhurst_grid = synth_map.draw_grid(axes=ax2, system='stonyhurst', annotate=False)
    synth_map.draw_limb()

    ax2.text(200, 30,
        'norm: '+str(list(float(i) for i in ["%.2f" % elem for elem in synth_view_settings['normal_vector']])) +
        '\n'+'north: '+str(list(float(i) for i in ["%.2f" % elem for elem in synth_view_settings['north_vector']])),
        style='italic', color='white')

    ax2.plot_coord(intensity_coords_, color='magenta', linewidth=0.75)
    ax2.plot_coord(intensity_coords2_, color='red', linewidth=0.75)

    plt.colorbar()

    #ax3 = fig.add_subplot(222)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(angular_separation, intensity, linewidth=0.65, label='AIA', color='magenta')
    ax3.plot(angular_separation_, intensity_, linewidth=0.65, label='Synthetic', color='magenta', linestyle='--')

    ax3.plot(angular_separation2, intensity2, linewidth=0.65, color='red')
    ax3.plot(angular_separation2_, intensity2_, linewidth=0.65, color='red', linestyle='--')
    ax3.set_xlabel("Angular distance along slit [arcsec]")
    #ax3.set_ylabel(f"Intensity [{cusp_submap.unit}]")
    ax3.set_ylabel('$I$, [DN cm$^5$ pix$^{-1}$ s$^{-1}$]')
    ax3.set_yscale('log')
    ax3.legend(frameon=False, fontsize=10)
    #plt.tight_layout()
    plt.savefig('aia_slit_profiles.pgf')

    # # # sdoaia131 = matplotlib.colormaps['sdoaia131']
    # #
    # fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    # plt.ioff()
    # plt.style.use('fast')
    # #
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # gs = fig.add_gridspec(2, 4)
    #
    # ax1 = fig.add_subplot(gs[:2, :2], projection=cusp_submap)
    # #cusp_submap.plot(axes=ax1, clip_interval=(1, 99.99)*u.percent)
    # stonyhurst_grid = cusp_submap.draw_grid(axes=ax1, system='stonyhurst')
    #
    # ax1.axis('image')
    # ax1.set_title('AIA/SDO 131A 2011-03-07 18:09:09', pad=44)
    # ax1.set_xlabel('X, pix')
    # ax1.set_ylabel('Y, pix')
    #
    # ax2 = fig.add_subplot(gs[:2, 2:], projection=synth_map)
    # #divider = make_axes_locatable(ax2)
    # #cax = divider.append_axes('right', size='5%', pad=0.05)
    # ax2.set_title('Synthetic image')
    # ax2.set_xlabel('X, Pixel')
    # ax2.set_ylabel('Y, Pixel')
    #
    # # plt.ioff()
    # # plt.savefig('slit_profiles.png')
    # # plt.close()