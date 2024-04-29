import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
import sunpy.map
import astropy.units as u

from utils.lso_align import synthmap_plot, calc_vect
import pickle

def slit_intensity(params_path, norm_slit, perp_slit, map_path=None, map=None, fig=None):
    """
    Method to produce an intensity-along-slit plot for an observation compared with a synthetic X-ray projection

    @param map_path: String path of cropped and pickled sunpy map
    @param params_path: String path of pickled CLB loop parameters
    @param norm_slit: 2D list containing arcsecond coordinates of the slit normal to the surface of the sun
                      (eg. [[x1,x2],[y1,y2]])
    @param perp_slit: 2D list containing arcsecond coordinates of the slit perpendicular to the norm_slit and
                      intersecting the y-region (eg. [[x1,x2],[y1,y2]])
    @param fig: OPTIONAL matplotlib fig object reference
    @return: N/A
    """

    # Retrieve sunpy map object
    if map:
        img = map
    else:
        with open(map_path, 'rb') as f:
            img = pickle.load(f)
            f.close()

    # Calculate normal and north vectors for synthetic image alignment
    # Also retrieve lat, lon coords from loop params
    norm, north, lat, lon = calc_vect(pkl=params_path)

    if map_path:
        synth_map = synthmap_plot(params_path, map_path=map_path, instr='aia', channel=171)
    if map:
        synth_map = synthmap_plot(params_path, map=map, instr='aia', channel=171)

    if not fig:
        import matplotlib.pyplot as plt
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))

    fig.set_tight_layout(True)
    plt.ioff()
    plt.style.use('fast')
    gs = fig.add_gridspec(3, 4)

    ax1 = fig.add_subplot(gs[:2, :2], projection=img)
    img.plot_settings['norm'] = colors.LogNorm(10, img.max())
    img.plot(axes=ax1)
    ax1.grid(False)
    img.draw_limb()

    # Normal observation slit
    line_coords = SkyCoord(norm_slit[0], norm_slit[1], unit=(u.arcsec, u.arcsec),
                           frame=img.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords = sunpy.map.pixelate_coord_path(img, line_coords)
    intensity = sunpy.map.sample_at_coords(img, intensity_coords)

    # Normal synthetic slit
    line_coords_ = SkyCoord(norm_slit[0], norm_slit[1], unit=(u.arcsec, u.arcsec),
                            frame=synth_map.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords_ = sunpy.map.pixelate_coord_path(synth_map, line_coords_)
    intensity_ = sunpy.map.sample_at_coords(synth_map, intensity_coords_)

    # Perpendicular Observation Slit
    line_coords2 = SkyCoord(perp_slit[0], perp_slit[1], unit=(u.arcsec, u.arcsec),
                            frame=img.coordinate_frame)  # [x1, x2], [y1, y2]
    intensity_coords2 = sunpy.map.pixelate_coord_path(img, line_coords2)
    intensity2 = sunpy.map.sample_at_coords(img, intensity_coords2)

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
    synth_map.plot_settings['norm'] = colors.LogNorm(10, img.max())
    synth_map.plot_settings['cmap'] = img.plot_settings['cmap']
    synth_map.plot(axes=ax2)
    ax2.grid(False)
    synth_map.draw_limb()

    ax2.text(200, 30,
             'norm: ' + str(list(float(i) for i in ["%.2f" % elem for elem in norm])) +
             '\n' + 'north: ' + str(list(float(i) for i in ["%.2f" % elem for elem in north])),
             style='italic', color='white')

    ax2.plot_coord(intensity_coords_, color='magenta', linewidth=0.75)
    ax2.plot_coord(intensity_coords2_, color='red', linewidth=0.75)

    plt.colorbar()

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(angular_separation, intensity, linewidth=0.65, label='AIA', color='magenta')
    ax3.plot(angular_separation_, intensity_, linewidth=0.65, label='Synthetic', color='magenta', linestyle='--')

    ax3.plot(angular_separation2, intensity2, linewidth=0.65, color='red')
    ax3.plot(angular_separation2_, intensity2_, linewidth=0.65, color='red', linestyle='--')
    ax3.set_xlabel("Angular distance along slit [arcsec]")
    ax3.set_ylabel('$I$, [DN cm$^5$ pix$^{-1}$ s$^{-1}$]')
    ax3.set_yscale('log')
    ax3.legend(frameon=False, fontsize=10)

    plt.show()
    plt.close()
