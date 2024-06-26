# in-progress calc_vect (Jun 20)
from rushlight.config import config
from rushlight.utils.proj_imag import SyntheticFilterImage as synt_img
import sys
sys.path.insert(1, config.CLB_PATH)
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d, semi_circle_loop # type: ignore

import yt
import astropy.units as u
import numpy as np
import sunpy.map
import astropy.constants as const
import pickle
from astropy.coordinates import SkyCoord, spherical_to_cartesian as stc
from sunpy.coordinates import Heliocentric, frames
import matplotlib.colors as colors


# Method to create synthetic map of MHD data from rad_transfer
def synthmap_plot(params_path, smap_path=None, smap=None, fig=None, plot=None, **kwargs):
    '''
    Method for plotting sunpy map of synthetic projection aligned to CLB flare loop

    @param params_path: string path to pickled Coronal Loop Builder loop parameters
    @param smap_path: string path to the reference map
    @param smap: sunpy reference map object
    @param fig: matplotlib figure object
    @param plot: string hint for kind of plot to produce
    @param kwargs: 'datacube' - path to simulation file; 'center', 'left_edge', 'right_edge' - bounds of simulation box
                   'instr' - simulated instrument (xrt or aia); 'channel' - simulated wavelength (in nm or eg. Ti-poly)
    @return: ax - axes object pointing to plotted synthetic image
    '''

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
    shen_datacube_65 = config.SIMULATIONS['DATASET']
    downs_file_path = kwargs.get('datacube', shen_datacube_65)
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
    normvector, northvector, lat, lon, radius, height, ifpd = calc_vect(pkl=params_path, ref_img=ref_img)

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

    # NOTE - TESTING
    # normvector=  [0,0,1]
    # northvector= [1,0,0]
    #~~~~~~~~~~~~~~~

    synth_view_settings = {'normal_vector': normvector,  # Line of sight - changes 'orientation' of projection
                           'north_vector': northvector}  # rotates projection in xy

    xrt_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[0, 0],  # move the bottom center of the flare in [x,y]
                                bkg_fill=np.min(ref_img.data))

    # define the heliographic sky coordinate of the midpoint of the loop
    hheight = 75 * u.Mm  # Half the height of the simulation box
    disp = hheight/s + height + radius

    fm = SkyCoord(lon=lon, lat=lat, radius=const.R_sun + disp,
                  frame='heliographic_stonyhurst',
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
            # coord=synth_map.reference_coordinate
            # pixels = synth_map.wcs.world_to_pixel(coord)
            # coord_img=ref_img.reference_coordinate
            # pixels_img=ref_img.wcs.world_to_pixel(coord_img)
            # center_image_pix = [synth_map.data.shape[0] / 2., synth_map.data.shape[1] / 2.] * u.pix
            # ax.plot_coord(coord, 'o', color='r')
            # ax.plot(pixels[0] * u.pix, pixels[1] * u.pix, 'x', color='w')
            # ax.plot_coord(coord_img, 'o', color='b')
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

    if 'pkl' in kwargs:
        with open(kwargs.get('pkl'), 'rb') as f:
            dims = pickle.load(f)
            # print(f'Loop dimensions loaded:{dims}')
            radius = dims['radius']
            height = dims['height']
            phi0 = dims['phi0']
            theta0 = dims['theta0']
            el = dims['el']
            az = dims['az']
            f.close()
    else:
        # Set the loop parameters using the provided values or default values
        radius = kwargs.get('radius', 10.0 * u.Mm)
        height = kwargs.get('height', 0.0 * u.Mm)
        phi0 = kwargs.get('phi0', 0.0 * u.deg)
        theta0 = kwargs.get('theta0', 0.0 * u.deg)
        el = kwargs.get('el', 90.0 * u.deg)
        az = kwargs.get('az', 0.0 * u.deg)

    # Stonyhurst coordinates of line of sight can be converted to the MHD coord frame
    #los_vec = [0, 0, -1] # observer's coord frame
    ref_img = kwargs.get('ref_img', None)
    try:
        if ref_img is not None:
            ref_img = ref_img
    except:
        print("\n\nHandled Exception:\n")
        raise Exception('Please provide Observation time from the Reference Image')

    # Loop Coordinates in Heliographic Stonyhurst
    loop, _ = semi_circle_loop(radius, 0, 0, False, height, theta0, phi0, el, az, samples_num)

    obstime = ref_img.reference_coordinate.obstime
    obscoord = ref_img.observer_coordinate

    x  = loop.cartesian.x.to(u.Mm)
    y  = loop.cartesian.y.to(u.Mm)
    z  = loop.cartesian.z.to(u.Mm)

    mpt = int(np.floor(len(x) / 2))    # Index for top of the loop
    # v1, v2 - center of sun to footpoint | v3 - center of sun to top of loop
    v1 = np.array([x[0].value, y[0].value, z[0].value]) * u.Mm
    v2 = np.array([x[-1].value, y[-1].value, z[-1].value]) * u.Mm
    v3 = np.array([x[mpt].value, y[mpt].value, z[mpt].value]) * u.Mm
    # vectors going from footpoint to top of loop
    v1_loop = v3 - v1
    v2_loop = v3 - v2

    # Construct the coordinate basis
    v_12 = v1-v2 # x-direction in mhd frame
    ifpd = np.linalg.norm(v_12)  # Inter-FootPoint Distance
    cross_product = np.cross(v1_loop, v2_loop) # z-direction in mhd frame
    
    x_mhd = v_12 / ifpd
    z_mhd = cross_product / np.linalg.norm(cross_product)
    zx_cross = np.cross(z_mhd, x_mhd)
    y_mhd = zx_cross / np.linalg.norm(zx_cross)
    
    # Transformation matrix from stonyhurst to MHD coordinates
    mhd_in_stonyh = np.column_stack((x_mhd, y_mhd, z_mhd))
    stonyh_to_mhd = np.linalg.inv(mhd_in_stonyh)
            
    # Get observer's los to the center of the sun
    los_vector = ref_img.observer_coordinate
    los_vector_cart = np.array([los_vector.cartesian.x.value,
                                los_vector.cartesian.y.value,
                                los_vector.cartesian.z.value])
    
    # Transform observer's los to norm vector for mhd
    los_vec = los_vector_cart / np.linalg.norm(los_vector_cart)
    norm = np.dot(stonyh_to_mhd, los_vec)
    norm = norm / np.linalg.norm(norm)

    print("\nNorm:")
    print(norm)

    # Derive the cartesian coordinates of a normalized vector pointing in the direction
    # of the coronal loop's spherical coordinates (midpoint of footpoints)
    midptn_coords = (
            SkyCoord(lon=phi0, lat=theta0, radius=const.R_sun, frame='heliographic_stonyhurst', obstime=obstime))
    # midptn_cart = stc(1, theta0, phi0)
    mpx = midptn_coords.cartesian.x.value
    mpy = midptn_coords.cartesian.y.value
    mpz = midptn_coords.cartesian.z.value

    midptn_cart = [mpx, mpy, mpz]
    # n = v3-midptn_cart

    north0 = [midptn_cart[0], midptn_cart[1], midptn_cart[2]]
    # north0 = [n[0], n[1], n[2]]
    north = np.dot(stonyh_to_mhd, north0)

    default = True
    if default:
        north = [0,1,0]

    print("North:")
    print(north)
    print("\n")

    lat, lon = theta0, phi0
    return norm, north, lat, lon, radius, height, ifpd

def get_trsfm(keyword=None):

    if keyword:
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
        # swap x and y
        x_y =  [[0,1,0],
                [1,0,0],
                [0,0,1]]
        # x = y, y = z, z = x
        custom =[[0,1,0],
                [0,0,1],
                [1,0,0]]
        # x = z, y = x, z = y
        customi=[[0,0,1],
                [1,0,0],
                [0,1,0]]
        
        trsfms = {
            "yax90" : yax90,
            "yax90i" : yax90i,
            "y_z" : y_z,
            "x_y" : x_y,
            "custom" : custom,
            "customi": customi,
        }

        trsfm = trsfms[keyword]
    else:
        trsfm= [[1,0,0],
                [0,1,0],
                [0,0,1]]

    return trsfm
