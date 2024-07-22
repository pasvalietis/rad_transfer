#!/usr/bin/env python

"""Script to calculate normal and north directions to align synthetic image with observations.
Current version 7/6/24
"""

import os
import sys

import yt
from yt.utilities.orientation import Orientation
from rushlight.utils.proj_imag import SyntheticFilterImage as synt_img
from rushlight.config import config
from rushlight.visualization.colormaps import color_tables

sys.path.insert(1, config.CLB_PATH)
from CoronalLoopBuilder.builder import CoronalLoopBuilder, semi_circle_loop, circle_3d # type: ignore

from unyt import unyt_array
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time, TimeDelta
import astropy.constants as const

import numpy as np
import sunpy.map
from scipy import ndimage #, datasets

import pickle
from astropy.coordinates import SkyCoord, CartesianRepresentation, spherical_to_cartesian as stc
from sunpy.coordinates import Heliocentric
import matplotlib.colors as colors
import matplotlib.pyplot as plt


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
    :rtype: tuple (sunpy.map.Map , matplotlib.pyplot.axes, normvector, northvector)
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
    dataset = kwargs.get('datacube', None)
    if isinstance(dataset, str):
        downs_file_path = kwargs.get('datacube', shen_datacube)
        subs_ds = yt.load(downs_file_path)
    else:
        subs_ds = dataset

    # Crop MHD file
    center = [0.0, 0.5, 0.0]
    left_edge = [-0.5, 0.016, -0.25]
    right_edge = [0.5, 1.0, 0.25]

    cut_box = subs_ds.region(center=kwargs.get('center', center),
                             left_edge=kwargs.get('left_edge', left_edge),
                             right_edge=kwargs.get('right_edge', right_edge))

    # Instrument settings for synthetic image
    instr = ref_img.instrument.split(' ')[0].lower()
    instr = kwargs.get('instr', instr).lower()  # keywords: 'aia' or 'xrt'
    channel = kwargs.get('channel', "Ti-poly" if instr.lower() == 'xrt' else 171)

    # Prepare cropped MHD data for imaging
    synth_imag = synt_img(cut_box, instr, channel)

    # Calculate normal and north vectors for synthetic image alignment
    # Also retrieve lat, lon coords from loop params
    normvector, northvector, lat, lon, radius, height, ifpd = calc_vect(pkl=params_path, ref_img=ref_img)

    # Match parameters of the synthetic image to observed one
    s = 1   # unscaled
    # s = 1.5 # 2012
    # s = 3   # 2013
    obs_scale = [ref_img.scale.axis1/s, ref_img.scale.axis2/s] * (u.arcsec / u.pixel)

    # Dynamic synth plot settings
    synth_plot_settings = {'resolution': ref_img.data.shape[0],
                           'vmin': kwargs.get('vmin', 1.),
                           'vmax': kwargs.get('vmax', 8.e2),
                           'norm': colors.LogNorm(kwargs.get('vmin', 1.), kwargs.get('vmax', 8.e2)),
                           'cmap': 'inferno',
                           'logscale': True}

    # normvector = [0,0,1]
    # northvector = [0,1,0]

    synth_view_settings = {'normal_vector': normvector,  # Line of sight - changes 'orientation' of projection
                           'north_vector': northvector}  # rotates projection in xy

    x, y = diff_roll(ref_img, lon, lat, normvector, northvector, subs_ds, **kwargs,)

    zoom = kwargs.get('zoom', None)
    bkg_fill = kwargs.get('bkg_fill', 5.e-1)
    
    synth_imag.proj_and_imag(plot_settings=synth_plot_settings,
                             view_settings=synth_view_settings,
                             image_shift=[x, y],  # move the bottom center of the flare in [x,y]
                             image_zoom=zoom,
                             bkg_fill=bkg_fill) #np.min(ref_img.data))

    # define the heliographic sky coordinate of the midpoint of the loop
    hheight = 75 * u.Mm  # Half the height of the simulation box
    disp = hheight/s + height + radius
        
    # disp = hheight
    # disp = 0
    
    timescale = kwargs.get('timescale', 291.89)
    
    timestep = subs_ds.current_time.value.item()
    timediff = TimeDelta(10 * timestep * timescale * u.s)
    # factor 10 to convert timestep to slice number
    
    start_time = Time(ref_img.reference_coordinate.obstime, scale='utc', format='isot')
    synth_obs_time = start_time + timediff
    
    print('obstime:', synth_obs_time)
    
    ref_coord = SkyCoord(ref_img.reference_coordinate.Tx, 
                         ref_img.reference_coordinate.Ty,
                        obstime=synth_obs_time,
                        observer=ref_img.reference_coordinate.observer,  # Temporarily 1 AU away
                        frame='helioprojective')#ref_img.reference_coordinate.frame) #ref_img.reference_coordinate
    
    
    map_kwargs = {'obstime': synth_obs_time,
              'reference_coord': ref_coord,
              'reference_pixel': u.Quantity(ref_img.reference_pixel), 
              'scale': u.Quantity(ref_img.scale),
              'telescope': ref_img.detector,
              'observatory': ref_img.observatory,
              'detector': ref_img.detector,
              'exposure': ref_img.exposure_time,
              'unit': ref_img.unit,
              'wavelength': kwargs.get('wavelength', ref_img.wavelength),
              'poisson': kwargs.get('poisson', False),
              }

    # Import scale from an AIA image:
    synth_map = synth_imag.make_synthetic_map(**map_kwargs)

    if fig:
        if plot == 'comp':
            comp = sunpy.map.Map(synth_map, ref_img, composite=True)
            comp.set_alpha(0, 0.50)
            ax = fig.add_subplot(projection=comp.get_map(0))
            comp.plot(axes=ax)
        elif plot == 'synth':
            # synth_map.plot_settings['norm'] = colors.LogNorm(kwargs.get('vmin', 1.), kwargs.get('vmax', 8.e2)) #colors.LogNorm(10, ref_img.max())
            synth_map.plot_settings['norm'] = colors.LogNorm(10, ref_img.max())
            synth_map.plot_settings['cmap'] = color_tables.aia_color_table(int(131) * u.angstrom) # ref_img.plot_settings['cmap']
            ax = fig.add_subplot(projection=synth_map)
            synth_map.plot(axes=ax)
            ax.grid(False)
            synth_map.draw_limb()
            ax.autoscale(False)

        elif plot == 'obs':
            ax = fig.add_subplot(projection=ref_img)
            ref_img.plot(axes=ax)
        else:
            ax = fig.add_subplot(projection=synth_map)
        
        shift = (x,y)
        return ax, synth_map, normvector, northvector, shift

    else:
        shift = (x,y)
        return synth_map, normvector, northvector, shift

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
        if isinstance(kwargs.get('pkl') , dict):
            dims = kwargs.get('pkl')
            radius = dims['radius']
            height = dims['height']
            phi0 = dims['phi0']
            theta0 = dims['theta0']
            el = dims['el']
            az = dims['az']
        else:
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

def coord_projection(coord, dataset, orientation=None, **kwargs):
    """
    Reproduces yt plot_modifications _project_coords functionality
    """
    # coord_copy should be a unyt array in code_units
    coord_copy = coord
    coord_vectors = coord_copy.transpose() - (dataset.domain_center.v * dataset.domain_center.uq)

    # orientation object is computed from norm and north vectors
    if orientation:
        unit_vectors = orientation.unit_vectors
    else:
        if 'norm_vector' in kwargs:
            norm_vector = kwargs['norm_vector']
            norm_vec = unyt_array(norm_vector) * dataset.domain_center.uq
        if 'north_vector' in kwargs:
            north_vector = kwargs['north_vector']
            north_vec = unyt_array(north_vector) * dataset.domain_center.uq
        if 'north_vector' and 'norm_vector' in kwargs:
            orientation = Orientation(norm_vec, north_vector=north_vec)
            unit_vectors = orientation.unit_vectors

    # Default image extents [-0.5:0.5, 0:1] imposes vertical shift
    y = np.dot(coord_vectors, unit_vectors[1]) + dataset.domain_center.value[1]
    x = np.dot(coord_vectors, unit_vectors[0])  # * dataset.domain_center.uq

    ret_coord = (x, y) # (y, x)

    return ret_coord

def code_coords_to_arcsec(code_coord, ref_image):
    """
    assume x axis extents in code units are [-.5 to .5] and y axis is changing from 0 to 1.
    """
    # acquire x and y extents of the reference_image
    # image center:
    center_x = ref_image.center.Tx
    center_y = ref_image.center.Ty

    x_code_coord, y_code_coord = code_coord[0], code_coord[1]

    resolution = ref_image.data.shape
    scale = ref_image.scale

    x_asec = center_x + resolution[0] * scale[0] * x_code_coord * u.pix
    y_asec = center_y + resolution[1] * scale[1] * (y_code_coord - 0.5) * u.pix

    return SkyCoord(x_asec, y_asec, frame=ref_image.coordinate_frame) #(x_asec, y_asec)

def diff_roll(ref_img: sunpy.map.Map, lon: Quantity, lat: Quantity, norm: list, north: list,
              dataset, **kwargs):
    """Calculate amount to shift image by difference between observed foot midpoint
    and selected "shift origin"

    :param ref_img: Reference observed image for the synthetic map
    :type ref_img: sunpy.map.Map
    :param lon: Longitude coordinate for CLB foot midpoint (u.deg)
    :type lon: Quantity
    :param lat: Latitude coordinate for CLB foot midpoint (u.deg)
    :type lat: Quantity
    :param norm: Normal LOS to the dataset 
    :type norm: list
    :param north: Vector directing camera rotation of projection
    :type north: list
    :return: Displacement vector x, y
    :rtype: tuple (int , int)
    """

    zoom = kwargs.get('zoom', None)

    # Synthetic Foot Midpoint (0,0,0 in code_units)
    north_q = unyt_array(north, dataset.units.code_length)
    norm_q = unyt_array(norm, dataset.units.code_length)

    ds_orientation = Orientation(norm_q, north_vector=north_q)
    synth_fpt_2d = coord_projection(unyt_array([0,0,0], dataset.units.code_length), 
                                   dataset, orientation=ds_orientation)
    synth_fpt_asec = code_coords_to_arcsec(synth_fpt_2d, ref_img)
    ori_pix = ref_img.wcs.world_to_pixel(synth_fpt_asec)

    if zoom and zoom < 1:
        # Find coordinates of bottom corner of "zoom area"
        if zoom >= 1:
                raise ValueError("Scale parameter has to be lower than 1")
        zoomed_img = ndimage.zoom(ref_img.data, zoom)  # scale<1
        y, x = ref_img.data.shape
        cropx = (zoomed_img.shape[0])
        cropy = (zoomed_img.shape[1])
        startx = (x - cropx) // 2
        starty = (y - cropy) // 2
    else:
        startx = 0
        starty = 0
        zoom = 1

    # Foot Midpoint from CLB
    mpt = SkyCoord(lon=lon, lat=lat, radius=const.R_sun,
                frame='heliographic_stonyhurst',
                observer='earth', obstime=ref_img.reference_coordinate.obstime).transform_to(frame='helioprojective')
    mpt_pix = ref_img.wcs.world_to_pixel(mpt)

    # Find difference between pixel positions
    x1 = float(mpt_pix[0])
    y1 = float(mpt_pix[1])
    # Shift and scale the synthetic coords by zoom
    x2 = float(ori_pix[0]*zoom + startx)
    y2 = float(ori_pix[1]*zoom + starty)

    x = int((x1-x2))
    y = int((y1-y2))

    # 'noroll' for debugging purposes
    if kwargs.get('noroll', False):
        x = 0
        y = 0

    # Return shift amount
    return x, y

