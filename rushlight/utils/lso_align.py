#!/usr/bin/env python

"""Script to calculate normal and north directions to align synthetic image with observations.
Current version 7/6/24
"""

import sys

from rushlight.config import config
from rushlight.visualization.colormaps import color_tables
sys.path.insert(1, config.CLB_PATH)
from CoronalLoopBuilder.builder import CoronalLoopBuilder, semi_circle_loop, circle_3d # type: ignore

import yt
from rushlight.utils.proj_imag import SyntheticFilterImage as synt_img
import astropy.units as u
import numpy as np
import sunpy.map
import astropy.constants as const
import pickle
from astropy.coordinates import SkyCoord, CartesianRepresentation, spherical_to_cartesian as stc
from sunpy.coordinates import Heliocentric
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from astropy.units import Quantity


def synthmap_plot(params_path: str, smap_path: str=None, smap: sunpy.map.Map=None, 
                  fig: plt.figure=None, plot: str=None, params=None, **kwargs):
    """Method for plotting sunpy map of synthetic projection aligned to CLB flare loop

    :param params_path: path to pickled Coronal Loop Builder loop parameters
    :type params_path: string
    :param smap_path: path to the reference map, defaults to None
    :type smap_path: string, optional
    :param smap: sunpy reference map object, defaults to None
    :type smap: sunpy.map.Map, optional
    :param fig: Matplotlib Figure Instance, defaults to None
    :type fig: matplotlib.pyplot.fig, optional
    :param plot: string hint for kind of plot to produce, defaults to None
    :type plot: string, optional
    :raises Exception: NullReference
    :return: Tuple containing sunpy synthetic map and optionally matplotlib axes object
    :rtype: tuple (sunpy.map.Map , matplotlib.pyplot.axes)
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
    shen_datacube = config.SIMULATIONS['DATASET']
    downs_file_path = kwargs.get('datacube', shen_datacube)
    subs_ds = yt.load(downs_file_path)

    # Crop MHD file
    center = [0.0, 0.5, 0.0]
    left_edge = [-0.5, 0.016, -0.25]
    right_edge = [0.5, 1.0, 0.25]

    cut_box = subs_ds.region(center=kwargs.get('center', center), left_edge=kwargs.get('left_edge', left_edge),
                             right_edge=kwargs.get('right_edge', right_edge))

    instr = ref_img.instrument.split(' ')[0].lower()
    # Instrument settings for synthetic image
    instr = kwargs.get('instr', instr).lower()  # keywords: 'aia' or 'xrt'
    channel = kwargs.get('channel', "Ti-poly" if instr.lower() == 'xrt' else 171)
    # Prepare cropped MHD data for imaging
    xrt_synthetic = synt_img(cut_box, instr, channel)

    # Calculate normal and north vectors for synthetic image alignment
    # Also retrieve lat, lon coords from loop params
    normvector, northvector, lat, lon, radius, height, ifpd = calc_vect(pkl=params_path, ref_img = ref_img)

    # Match parameters of the synthetic image to observed one
    s = 1   # unscaled
    # s = 1.5 # 2012
    # s = 3   # 2013
    obs_scale = [ref_img.scale.axis1/s, ref_img.scale.axis2/s] * (u.arcsec / u.pixel)

    # Dynamic synth plot settings
    synth_plot_settings = {'resolution': ref_img.data.shape[0],
                           'vmin': 1e-15,
                           'vmax': 8e1,
                           'cmap': 'inferno',
                           'logscale': True}

    # normvector = [0,0,1]
    # northvector = [0,1,0]

    synth_view_settings = {'normal_vector': normvector,  # Line of sight - changes 'orientation' of projection
                           'north_vector': northvector}  # rotates projection in xy

    x, y = diff_roll(ref_img, lon, lat, **kwargs)

    xrt_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[x, y],  # move the bottom center of the flare in [x,y]
                                bkg_fill=np.min(ref_img.data))

    # define the heliographic sky coordinate of the midpoint of the loop
    hheight = 75 * u.Mm  # Half the height of the simulation box
    disp = hheight/s + height + radius
    # disp = hheight
    # disp = 0

    fm = SkyCoord(lon=lon, lat=lat, radius=const.R_sun + disp,
                  frame='heliographic_stonyhurst',
                  observer='earth', obstime=ref_img.reference_coordinate.obstime).transform_to(frame='helioprojective')

    kwargs = {'obstime': ref_img.reference_coordinate.obstime,
            #   'reference_coord': fm,
              'reference_coord': ref_img.reference_coordinate,
              'reference_pixel': u.Quantity(ref_img.reference_pixel), 
            #   'scale': obs_scale,
              'scale': u.Quantity(ref_img.scale),
              'telescope': ref_img.detector,
              'observatory': ref_img.observatory,
              'detector': ref_img.detector,
            #   'instrument': ref_img.instrument,
              'exposure': ref_img.exposure_time,
              'unit': ref_img.unit,
              }

    # Import scale from an AIA image:
    synth_map = xrt_synthetic.make_synthetic_map(**kwargs)

    if fig:
        if plot == 'comp':
            comp = sunpy.map.Map(synth_map, ref_img, composite=True)
            comp.set_alpha(0, 0.50)
            ax = fig.add_subplot(projection=comp.get_map(0))
            comp.plot(axes=ax)
        elif plot == 'synth':

            # synth_map.plot(axes=ax, vmin=1e-5, vmax=8e1, cmap='inferno')
            synth_map.plot_settings['norm'] = colors.LogNorm(10, ref_img.max())
            synth_map.plot_settings['cmap'] = color_tables.aia_color_table(int(131) * u.angstrom) # ref_img.plot_settings['cmap']
            ax = fig.add_subplot(projection=synth_map)
            synth_map.plot(axes=ax)
            ax.grid(False)
            synth_map.draw_limb()

            # coord=synth_map.reference_coordinate
            # coord_img=ref_img.reference_coordinate
            # ax.plot_coord(coord, 'o', color='r')
            # ax.plot_coord(coord_img, 'o', color='b')

            # Plotting key map points [debug purposes]
            # pixels = synth_map.wcs.world_to_pixel(coord)
            # pixels_img=ref_img.wcs.world_to_pixel(coord_img)
            # center_image_pix = [synth_map.data.shape[0] / 2., synth_map.data.shape[1] / 2.] * u.pix
            # ax.plot(pixels[0] * u.pix, pixels[1] * u.pix, 'x', color='w')

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

def calc_vect(radius: Quantity=const.R_sun, height: Quantity=10 * u.Mm, theta0: Quantity=0 * u.deg, phi0: Quantity=0 * u.deg, 
              el: Quantity=90 * u.deg, az: Quantity=0 * u.deg, samples_num: int=100, **kwargs):
    """Calculates the north and normal vectors for the synthetic image

    :param radius: radius of the CLB loop, defaults to const.R_sun
    :type radius: Quantity, optional
    :param height: height of the center of the CLB loop above solar surface, defaults to 10*u.Mm
    :type height: Quantity, optional
    :param theta0: longitude coordinate, defaults to 0*u.deg
    :type theta0: Quantity, optional
    :param phi0: latitude coordinate, defaults to 0*u.deg
    :type phi0: Quantity, optional
    :param el: angle of CLB loop relative to tangent plane of solar surface, defaults to 90*u.deg
    :type el: Quantity, optional
    :param az: rotation of CLB loop around vector normal to solar surface, defaults to 0*u.deg
    :type az: Quantity, optional
    :param samples_num: Number of points that make up the CLB loop, defaults to 100
    :type samples_num: int, optional
    :raises Exception: Null reference to kwargs member
    :return: norm, north, lat, lon, radius, height, ifpd
    :rtype: tuple (list, list, Quantity, Quantity, Quantity, Quantity, float)
    """    

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
            el = dims['el']
            az = dims['az']
            f.close()
    else:
        # Set the loop parameters using the provided values or default values
        radius = kwargs.get('radius', DEFAULT_RADIUS)
        height = kwargs.get('height', DEFAULT_HEIGHT)
        phi0 = kwargs.get('phi0', DEFAULT_PHI0)
        theta0 = kwargs.get('theta0', DEFAULT_THETA0)
        el = kwargs.get('el', DEFAULT_EL)
        az = kwargs.get('az', DEFAULT_AZ)

    # Define the vectors v1 and v2 (from center of sun to footpoints)
    
    try:
        loop_coords = semi_circle_loop(radius, 0, 0, False, height, theta0, phi0, el, az, samples_num=100)[0].cartesian
    except:
        print("Error handled: Your CLB does not support semicircles \n")
        loop_coords = semi_circle_loop(radius, height, theta0, phi0, el, az, samples_num=100)[0].cartesian

    v1 = np.array([loop_coords[0].x.value, 
                   loop_coords[0].y.value,
                   loop_coords[0].z.value])
    
    v2 = np.array([loop_coords[-1].x.value, 
                   loop_coords[-1].y.value,
                   loop_coords[-1].z.value])
    
    v3 = np.array([loop_coords[int(loop_coords.shape[0]/2.)].x.value, 
                   loop_coords[int(loop_coords.shape[0]/2.)].y.value,
                   loop_coords[int(loop_coords.shape[0]/2.)].z.value])

    # Inter-FootPoint distance
    v_12 = v1-v2  # x-direction in mhd frame
    ifpd = np.linalg.norm(v_12)

    # vectors going from footpoint to top of loop
    v1_loop = v3 - v1
    v2_loop = v3 - v2

    # Use the cross product to determine the orientation of the loop plane
    cross_product = np.cross(v1_loop, v2_loop) # z-direction in mhd frame

    # Normal Vector
    norm0 = cross_product / np.linalg.norm(cross_product)

    # Defining MHD base coordinate system
    z_mhd = norm0
    x_mhd = v_12 / np.linalg.norm(v_12)
    zx_cross = np.cross(z_mhd, x_mhd)
    y_mhd = zx_cross / np.linalg.norm(zx_cross)
    
    # Transformation matrix from stonyhurst to MHD coordinates

    mhd_in_stonyh = np.column_stack((x_mhd, y_mhd, z_mhd))
    stonyh_to_mhd = np.linalg.inv(mhd_in_stonyh)
    
    # Stonyhurst coordinates of line of sight can be converted to the MHD coord frame
    #los_vec = [0, 0, -1] # observer's coord frame
    ref_img = kwargs.get('ref_img', None)
    try:
        if ref_img is not None:
            ref_img = ref_img

    except:
        print("\n\nHandled Exception:\n")
        raise Exception('Please provide Observation time from the Reference Image')
        

    los_vector_obs = SkyCoord(CartesianRepresentation(0*u.Mm, 0*u.Mm, -1*u.Mm),
                          obstime=ref_img.coordinate_frame.obstime,
                          observer=ref_img.coordinate_frame.observer,
                          frame="heliocentric")
    
    imag_rot_matrix = ref_img.rotation_matrix
    
    cam_default = np.array([0, 1])
    cam_pt = np.dot(imag_rot_matrix, cam_default)  # camera pointing
    
    camera_north_obs = SkyCoord(CartesianRepresentation(cam_pt[0]*u.Mm, 
                                                        cam_pt[1]*u.Mm, 
                                                        0*u.Mm),
                          obstime=ref_img.coordinate_frame.obstime,
                          observer=ref_img.coordinate_frame.observer,
                          frame="heliocentric")
    
    los_vector = los_vector_obs.transform_to('heliographic_stonyhurst')
    camera_north = camera_north_obs.transform_to('heliographic_stonyhurst')

    los_vector_cart = np.array([los_vector.cartesian.x.value,
                                los_vector.cartesian.y.value,
                                los_vector.cartesian.z.value])
    
    camera_north_cart = np.array([camera_north.cartesian.x.value,
                                  camera_north.cartesian.y.value,
                                  camera_north.cartesian.z.value])
    
    los_vec = los_vector_cart / np.linalg.norm(los_vector_cart)
    camera_vec = camera_north_cart / np.linalg.norm(camera_north_cart)
    
    norm_vec = np.dot(stonyh_to_mhd, los_vec).T
    norm_vec = norm_vec / np.linalg.norm(norm_vec)
    
    north_vec = np.dot(stonyh_to_mhd, camera_vec).T
    north_vec = north_vec / np.linalg.norm(north_vec)

    # Inverting y component of the north vector in the MHD reference frame
    north_vec[1] = - north_vec[1]


    print("\nNorm:")
    print(norm_vec)

    # Derive the cartesian coordinates of a normalized vector pointing in the direction
    # of the coronal loop's spherical coordinates (midpoint of footpoints)
    midptn_cart = stc(1, theta0, phi0)
    
    # DEFAULT: CAMERA UP
    default = False
    if default:
        north = [0, 1., 0] 
        north_vec = np.array(north)
    
    print("North:")
    print(north_vec)
    print("\n")

    lat, lon = theta0, phi0
    return norm_vec, north_vec, lat, lon, radius, height, ifpd

def get_trsfm(keyword: str=None):
    """***DEPRECIATED*** 
    Returns preset transformation matrices corresponding to particular rotations

    :param keyword: Hint for kind of rotation to produce, defaults to None
    :type keyword: str, optional
    :return: 2D transformation matrix
    :rtype: list
    """

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

def diff_roll(ref_img: sunpy.map.Map, lon: Quantity, lat: Quantity, **kwargs):
    """Calculate amount to shift image by difference between observed foot midpoint
    and selected "shift origin"

    :param ref_img: Reference observed image for the synthetic map
    :type ref_img: sunpy.map.Map
    :param lon: Longitude coordinate for CLB foot midpoint (u.deg)
    :type lon: Quantity
    :param lat: Latitude coordinate for CLB foot midpoint (u.deg)
    :type lat: Quantity
    :return: Displacement vector x, y
    :rtype: tuple (int , int)
    """
    
    # Calculate amount to shift image by
    # Difference between foot midpoint and shift origin

    shift_origin = kwargs.get("sh_ori", "ref")
    if shift_origin == 'fpt0':
        # Manually Selected Synthetic Footpoint (2012-07-19 Event)
        fpt_coord = SkyCoord(630*u.arcsec, -250*u.arcsec, frame=ref_img.coordinate_frame)
        fpt_pix = ref_img.wcs.world_to_pixel(fpt_coord)

        ori_pix = fpt_pix
    else:
        # Reference Pixel (Center of View)
        ref = ref_img.reference_coordinate
        ref_pix = ref_img.wcs.world_to_pixel(ref)
        
        ori_pix = ref_pix


    # Foot Midpoint
    mpt = SkyCoord(lon=lon, lat=lat, radius=const.R_sun,
                frame='heliographic_stonyhurst',
                observer='earth', obstime=ref_img.reference_coordinate.obstime).transform_to(frame='helioprojective')
    mpt_pix = ref_img.wcs.world_to_pixel(mpt)

    
    x1 = float(mpt_pix[0])
    y1 = float(mpt_pix[1])

    x2 = float(ori_pix[0])
    y2 = float(ori_pix[1])

    x = int(x1-x2)
    y = int(y1-y2)

    if kwargs.get('noroll', False):
        x = 0
        y = 0

    return x, y


