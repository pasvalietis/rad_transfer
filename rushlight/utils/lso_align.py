import sys
sys.path.insert(1, '/home/saber/CoronalLoopBuilder')
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d # type: ignore

import yt
from utils.proj_imag import SyntheticFilterImage as synt_img
import astropy.units as u
import numpy as np
import sunpy.map
import astropy.constants as const
import pickle
from astropy.coordinates import SkyCoord, spherical_to_cartesian as stc
import matplotlib.colors as colors


# Method to create synthetic map of MHD data from rad_transfer
def synthmap_plot(params_path, smap_path=None, smap=None, fig=None, plot=None, **kwargs):
    '''

    @param params_path: string path to pickled Coronal Loop Builder loop parameters
    @param smap_path: string path to the reference map
    @param smap: sunpy reference map object
    @param fig: matplotlib figure object
    @param plot: string hint for kind of plot to produce
    @param kwargs: 'datacube' - path to simulation file; 'center', 'left_edge', 'right_edge' - bounds of simulation box
                   'instr' - simulated instrument (xrt or aia); 'channel' - simulated wavelength (in nm or eg. Ti-poly)
    @return: ax - axes object pointing to plotted synthetic image
    '''

    """

    @param img: Real image to project synthetic map onto
    @param fig: Matplotlib figure reference for plotting
    @param normvector: Vector indicating vector normal to the surface of the sun
    @param northvector: Vector indicating pointing direction for candleflame 'tip' of flare
    @param comp: 'True' for real + synthetic image, 'False' for just synthetic image
    @param fm_coord: Coordinate for the foot midpoint of the flare loop
    @param kwargs: 
    @return:
    """

    # Retrieve reference image (ref_img)
    try:
        if smap:
            ref_img = smap
        else:
            try:
                with open(smap_path, 'rb') as f:
                    ref_img = pickle.load(f)
                    f.close()
            except:
                ref_img = sunpy.map.Map(smap_path)
    except:
        print("\n\nHandled Exception:\n")
        raise Exception('Please provide either a map (xmap) or path to pickled map (xmap_path)')

    # Load subsampled 3D MHD file
    shen_datacube = '/home/saber/PycharmProjects/rad_transfer_/datacubes/subs_3_flarecs-id_0065.h5'
    downs_file_path = kwargs.get('datacube', shen_datacube)
    subs_ds = yt.load(downs_file_path)

    # Crop MHD file
    center = [0.0, 0.5, 0.0]
    left_edge = [-0.5, 0.016, -0.25]
    right_edge = [0.5, 1.0, 0.25]
    cut_box = subs_ds.region(center=kwargs.get('center', center), left_edge=kwargs.get('left_edge', left_edge),
                             right_edge=kwargs.get('right_edge', right_edge))

    # Instrument settings for synthetic image
    instr = kwargs.get('instr', 'xrt')  # keywords: 'aia' or 'xrt'
    channel = kwargs.get('channel', 'Ti-poly')
    # Prepare cropped MHD data for imaging
    xrt_synthetic = synt_img(cut_box, instr, channel)

    # Calculate normal and north vectors for synthetic image alignment
    # Also retrieve lat, lon coords from loop params
    normvector, northvector, lat, lon, radius, height, ifpd = calc_vect(pkl=params_path)

    # Match parameters of the synthetic image to observed one
    samp_resolution = ref_img.data.shape[0]
    s = 1   # unscaled
    # s = 1.5 # 2012
    # s = 3   # 2013
    obs_scale = [ref_img.scale.axis1/s, ref_img.scale.axis2/s] * (u.arcsec / u.pixel)

    # Dynamic synth plot settings
    synth_plot_settings = {'resolution': samp_resolution,
                           'vmin': 1e-15,
                           'vmax': 8e1,
                           'cmap': 'inferno',
                           'logscale': True}

    synth_view_settings = {'normal_vector': normvector,  # Line of sight - changes 'orientation' of projection
                           'north_vector': northvector}  # rotates projection in xy

    xrt_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[0, 0],  # move the bottom center of the flare in [x,y]
                                bkg_fill=np.min(ref_img.data))

    # define the heliographic sky coordinate of the midpoint of the loop
    hheight = 75 * u.Mm  # Half the height of the simulation box
    disp = hheight/s + height + radius

    fm = SkyCoord(lon=lon, lat=lat, radius=const.R_sun + disp, frame='heliographic_stonyhurst',
                  observer='earth', obstime=ref_img.reference_coordinate.obstime).transform_to(frame='helioprojective')
    
    # Import scale from an AIA image:
    synth_map = xrt_synthetic.make_synthetic_map(
                                                 obstime=ref_img.reference_coordinate.obstime,
                                                 observer='earth',
                                                 detector='Synthetic XRT',
                                                 scale=obs_scale,
                                                 reference_coord=fm,
                                                 )

    if fig:
        if plot == 'comp':
            comp = sunpy.map.Map(synth_map, ref_img, composite=True)
            comp.set_alpha(0, 0.50)
            ax = fig.add_subplot(projection=comp.get_map(0))
            comp.plot(axes=ax)
        elif plot == 'synth':
            ax = fig.add_subplot(projection=synth_map)
            # synth_map.plot(axes=ax, vmin=1e-5, vmax=8e1, cmap='inferno')
            synth_map.plot_settings['norm'] = colors.LogNorm(10, ref_img.max())
            synth_map.plot_settings['cmap'] = ref_img.plot_settings['cmap']
            synth_map.plot(axes=ax)
            ax.grid(False)
            synth_map.draw_limb()

            # Plotting key map points [debug purposes]
            center = SkyCoord(lon=lon, lat=lat, radius=1*u.cm, frame='heliographic_stonyhurst',
                  observer='earth', obstime=ref_img.reference_coordinate.obstime).transform_to(frame='helioprojective')
            # coord=synth_map.reference_coordinate
            # pixels = synth_map.wcs.world_to_pixel(coord)
            # coord_img=ref_img.reference_coordinate
            # pixels_img=ref_img.wcs.world_to_pixel(coord_img)
            # center_image_pix = [synth_map.data.shape[0] / 2., synth_map.data.shape[1] / 2.] * u.pix
            ax.plot_coord(fm, 'o', color='r')
            # ax.plot(pixels[0] * u.pix, pixels[1] * u.pix, 'x', color='w')
            ax.plot_coord(center, 'o', color='b')
            # ax.plot(pixels_img[0] * u.pix, pixels_img[1] * u.pix, 'x', color='w')
            # ax.plot(center_image_pix[0], center_image_pix[1], 'x', color='g')

        elif plot == 'obs':
            ax = fig.add_subplot(projection=ref_img)
            ref_img.plot(axes=ax)
        else:
            ax = fig.add_subplot(projection=synth_map)

        return ax, synth_map

    else:
        return synth_map


def calc_vect(radius=const.R_sun, height=10 * u.Mm, theta0=0 * u.deg, phi0=0 * u.deg, el=90 * u.deg, az=0 * u.deg,
              samples_num=100, **kwargs):

    DEFAULT_RADIUS = 10.0 * u.Mm
    DEFAULT_HEIGHT = 0.0 * u.Mm
    DEFAULT_PHI0 = 0.0 * u.deg
    DEFAULT_THETA0 = 0.0 * u.deg
    DEFAULT_EL = 90.0 * u.deg
    DEFAULT_AZ = 0.0 * u.deg

    if 'pkl' in kwargs:
        with open(kwargs.get('pkl'), 'rb') as f:
            dims = pickle.load(f)
            # print(f'Loop dimensions loaded:{dims}')
            radius = dims['radius']
            height = dims['height']
            phi0 = dims['phi0']
            theta0 = dims['theta0']
            el = dims['el'] # + 45*u.deg
            # az = dims['az'] + 180*u.deg
            az = 0*u.deg
            f.close()
    else:
        # Set the loop parameters using the provided values or default values
        radius = kwargs.get('radius', DEFAULT_RADIUS)
        height = kwargs.get('height', DEFAULT_HEIGHT)
        phi0 = kwargs.get('phi0', DEFAULT_PHI0)
        theta0 = kwargs.get('theta0', DEFAULT_THETA0)
        el = kwargs.get('el', DEFAULT_EL)
        az = kwargs.get('az', DEFAULT_AZ)

    r_1 = const.R_sun

    r0 = r_1 + height
    x0 = u.Quantity(0 * u.cm)
    y0 = u.Quantity(0 * u.cm)
    z0 = r0.to(u.cm)

    theta = el.to(u.rad).value  # np.pi / 2  # Elevation angle
    phi = az.to(u.rad).value #+ np.pi / 2 # np.pi / 4  # Azimuth angle
    t = np.linspace(0, 2 * np.pi, int(samples_num))  # Parameter t

    dx, dy, dz = circle_3d(0, 0, 0, radius, theta, phi, t)

    # Arrays of parametric coordinates
    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rdiff = r - r_1     # Array of radii minus radius of sun
    rsort = np.argmin(np.abs(rdiff))    # Minimum r of parametric point
    if rdiff[rsort] + rdiff[rsort + 1] < 0:
        rsort += 1
    r = np.roll(r, -rsort)
    x = np.roll(x, -rsort)
    y = np.roll(y, -rsort)
    z = np.roll(z, -rsort)

    i_r = np.where(r > r_1)

    x = x[i_r]
    y = y[i_r]
    z = z[i_r]

    # Define the vectors v1 and v2
    v1 = np.array([x[0].value, y[0].value, z[0].value]) * x[0].unit
    v2 = np.array([x[-1].value, y[-1].value, z[-1].value]) * x[0].unit

    # Inter-FootPoint distance
    ifpd = np.linalg.norm(v1-v2)

    # Use the cross product to determine the orientation
    # cross_product = np.cross(v1, v2) if az.value < 90 else -np.cross(v1, v2)
    cross_product = np.cross(v1, v2)

    nothing=[[1,0,0],
             [0,1,0],
             [0,0,1]]
    # +90 degree rotation around y axis
    yax90 = [[0,0,1],
             [0,1,0],
             [-1,0,0]]
    # +90 degree rotation around y axis inverted
    yax90i = [[0,0,-1],
             [0,1,0],
             [1,0,0]]
    # swap y and z
    y_z =   [[1,0,0],
             [0,0,1],
             [0,1,0]]
    # x = y, y = z, z = x
    custom =[[0,1,0],
             [0,0,1],
             [1,0,0]]
    # x = z, y = x, z = y
    customi=[[0,0,1],
             [1,0,0],
             [0,1,0]]
    
    transformation = yax90

    # Normal Vector
    norm0 = cross_product / np.linalg.norm(cross_product)
    # Transformation to MHD coordinate frame
    norm = [0, 0, 0]
    norm = np.dot(transformation, norm0)

    print("\nNorm:")
    print(norm)

    # Derive the cartesian coordinates of a normalized vector pointing in the direction
    # of the coronal loop's spherical coordinates (midpoint of footpoints)
    midptn_cart = stc(1, theta0, phi0)
    
    # North Vector
    north0 = [midptn_cart[0].value, midptn_cart[1].value, midptn_cart[2].value]
    # Transformation to MHD coordinate frame
    north = [0, 0, 0]
    north = np.dot(transformation, north0)

    # north = north0
    print("North:")
    print(north)
    print("\n")

    lat, lon = theta0, phi0
    # normal vector, north vector, latitude, longitude
    return norm, north, lat, lon, radius, height, ifpd
