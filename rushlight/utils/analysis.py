import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
import sunpy.map
import astropy.units as u

from utils.lso_align import synthmap_plot, calc_vect
import pickle


def slit_intensity(params_path, norm_slit, perp_slit, xmap_path=None, xmap=None, smap_path=None, smap=None, fig=None,
                   **kwargs):
    """
    Method to produce an intensity-along-slit plot for an observation compared with a synthetic X-ray projection

    @param smap_path: String path of cropped and pickled sunpy map
    @param params_path: String path of pickled CLB loop parameters
    @param norm_slit: 2D list containing arcsecond coordinates of the slit normal to the surface of the sun
                      (eg. [[x1,x2],[y1,y2]])
    @param perp_slit: 2D list containing arcsecond coordinates of the slit perpendicular to the norm_slit and
                      intersecting the y-region (eg. [[x1,x2],[y1,y2]])
    @param fig: OPTIONAL matplotlib fig object reference
    @return: N/A
    """

    # Generate synthetic map from provided observation map (smap) and coronal loop parameters (params_path)
    try:
        instr = kwargs.get('instr', 'xrt')
        channel = kwargs.get('channel', 'Ti-poly')
        if smap:
            synth_map = synthmap_plot(params_path, smap=smap, instr=instr, channel=channel)
        elif smap_path:
            synth_map = synthmap_plot(params_path, map_path=smap_path, instr=instr, channel=channel)
    except:
        print("\n\nHandled Exception:\n")
        raise Exception('Please provide either a map (smap) or path to map (smap_path)')

    # Retrieve xrt comparison image (ximg)
    try:
        if xmap:
            ximg = xmap
        else:
            try:
                with open(xmap_path, 'rb') as f:
                    ximg = pickle.load(f)
                    f.close()
            except:
                ximg = sunpy.map.Map(xmap_path)
    except:
        print("\n\nHandled Exception:\n")
        raise Exception('Please provide either a map (xmap) or path to map (xmap_path)')

    # Checks if matplotlib object has been targeted for plotting
    # If not, creates a new instance and plots there
    if not fig:
        import matplotlib.pyplot as plt
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))

    fig.set_tight_layout(True)
    plt.ioff()
    plt.style.use('fast')
    gs = fig.add_gridspec(3, 4)

    ax1 = fig.add_subplot(gs[:2, :2], projection=ximg)
    ximg.plot_settings['norm'] = colors.LogNorm(10, ximg.max())
    ximg.plot(axes=ax1)
    ax1.grid(False)
    ximg.draw_limb()

    # Normal observation slit
    line_coords = SkyCoord(norm_slit[0], norm_slit[1], unit=(u.arcsec, u.arcsec),
                           frame=ximg.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords = sunpy.map.pixelate_coord_path(ximg, line_coords)
    intensity = sunpy.map.sample_at_coords(ximg, intensity_coords)

    # Normal synthetic slit
    line_coords_ = SkyCoord(norm_slit[0], norm_slit[1], unit=(u.arcsec, u.arcsec),
                            frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords_ = sunpy.map.pixelate_coord_path(synth_map, line_coords_)
    intensity_ = sunpy.map.sample_at_coords(synth_map, intensity_coords_)

    # Perpendicular Observation Slit
    line_coords2 = SkyCoord(perp_slit[0], perp_slit[1], unit=(u.arcsec, u.arcsec),
                            frame=ximg.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords2 = sunpy.map.pixelate_coord_path(ximg, line_coords2)
    intensity2 = sunpy.map.sample_at_coords(ximg, intensity_coords2)

    # Perpendicular Synthetic Slit
    line_coords2_ = SkyCoord(perp_slit[0], perp_slit[1], unit=(u.arcsec, u.arcsec),
                             frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords2_ = sunpy.map.pixelate_coord_path(synth_map, line_coords2_)
    intensity2_ = sunpy.map.sample_at_coords(synth_map, intensity_coords2_)

    # Angular separation ranges
    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    angular_separation_ = intensity_coords_.separation(intensity_coords_[0]).to(u.arcsec)
    angular_separation2 = intensity_coords2.separation(intensity_coords2[0]).to(u.arcsec)
    angular_separation2_ = intensity_coords2_.separation(intensity_coords2_[0]).to(u.arcsec)

    # Plot slits
    ax1.plot_coord(intensity_coords, color='magenta', linewidth=0.75)
    ax1.plot_coord(intensity_coords2, color='red', linewidth=0.75)

    ax2 = fig.add_subplot(gs[:2, 2:], projection=synth_map)
    synth_map.plot_settings['norm'] = colors.LogNorm(10, ximg.max())
    synth_map.plot_settings['cmap'] = ximg.plot_settings['cmap']
    synth_map.plot(axes=ax2)
    ax2.grid(False)
    synth_map.draw_limb()

    # Retrieve normal and north vectors used for synthetic image alignment
    norm, north, _, _, _, _, _ = calc_vect(pkl=params_path)

    ax2.text(200, 30,
             'norm: ' + str(list(float(i) for i in ["%.2f" % elem for elem in norm])) +
             '\n' + 'north: ' + str(list(float(i) for i in ["%.2f" % elem for elem in north])),
             style='italic', color='white')

    ax2.plot_coord(intensity_coords_, color='magenta', linewidth=0.75)
    ax2.plot_coord(intensity_coords2_, color='red', linewidth=0.75)

    plt.colorbar()

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(angular_separation, intensity, linewidth=0.65, label='XRT', color='magenta')
    ax3.plot(angular_separation_, intensity_, linewidth=0.65, label='Synthetic', color='magenta', linestyle='--')

    ax3.plot(angular_separation2, intensity2, linewidth=0.65, color='red')
    ax3.plot(angular_separation2_, intensity2_, linewidth=0.65, color='red', linestyle='--')
    ax3.set_xlabel("Angular distance along slit [arcsec]")
    ax3.set_ylabel('$I$, [DN cm$^5$ pix$^{-1}$ s$^{-1}$]')
    ax3.set_yscale('log')
    ax3.legend(frameon=False, fontsize=10)

    if kwargs.get('clb', False):
        CLB_PATH = '/home/saber/CoronalLoopBuilder'
        import sys
        sys.path.insert(1, CLB_PATH)
        from CoronalLoopBuilder.builder import CoronalLoopBuilder #type: ignore

        coronal_loop1 = CoronalLoopBuilder(fig, [ax1, ax2], [ximg, synth_map], pkl=params_path)

    plt.show()
    plt.close()

    if kwargs.get('lp_sv', False):
        destination = kwargs.get('lp_dst', '2012/back_2012_testing.pkl')
        coronal_loop1.save_params_to_pickle(destination)
