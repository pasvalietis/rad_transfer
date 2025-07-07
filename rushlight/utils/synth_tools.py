import numpy as np

from rushlight.config import config
import sys
sys.path.insert(1, config.CLB_PATH)
from CoronalLoopBuilder.builder import semi_circle_loop # type: ignore

import pickle
import sunpy
from unyt import unyt_array
import numpy as np

import astropy
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u

from yt.utilities.orientation import Orientation

###############################################################

def calc_vect(loop_coords: np.ndarray, ref_img: astropy.nddata.NDData, **kwargs):
    """Calculates the north and normal vectors for the synthetic image

    #NOTE Change to accept 3 coordinates: of fpt1, of fpt2, and of loop apex
    

    :param loop_coords: Coordinates of the CLB loop as an array of
                        `astropy.coordinates.SkyCoord` objects.
    :type loop_coords: astropy.coordinates.SkyCoord
    :param ref_img: A reference FITS image or `astropy.nddata.NDData` object
                    containing coordinate information (observer, obstime, rotation).
    :type ref_img: astropy.nddata.NDData
    :param default: If True, sets the north vector in the MHD frame to [0, 1, 0],
                    overriding the calculation based on the reference image.
                    Defaults to False.
    :type default: bool, optional
    :return: A tuple containing:
            **normvector** (:class:`numpy.ndarray`): The normal vector of the loop plane
            in the MHD coordinate system,
            **northvector** (:class:`numpy.ndarray`): The north vector of the camera
            in the MHD coordinate system,
            **ifpd** (:class:`float`): The inter-footpoint distance of the loop in Mm
    :rtype: tuple (numpy.ndarray, numpy.ndarray, float)

    .. note::
        The MHD coordinate system is defined such that:
        - The x-axis points from one footpoint of the loop to the other.
        - The z-axis is normal to the plane defined by the three loop coordinates
        (v1, v2, v3).
        - The y-axis completes the right-handed coordinate system.

        - The north vector is derived from the camera pointing information in the
        reference image and transformed into the MHD frame. The y-component of
        the north vector in the MHD frame is inverted by convention.

        - Setting `default=True` forces the north vector to be aligned with the
        y-axis of the MHD frame, which can be useful for specific alignment purposes.
    """

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
    
    # DEFAULT: CAMERA UP
    default = kwargs.get('default', False)
    if default:
        north = [0, 1., 0] 
        north_vec = np.array(north)
    
    northvector = north_vec
    normvector = norm_vec

    return (normvector, northvector, ifpd)

def get_loop_coords(loop_params):
    """
    Generates Cartesian coordinates for a semi-circular loop based on the provided parameters.

    :param loop_params: A dictionary containing the parameters defining the loop.
                        Required keys include at least one of 'radius' (for circular loops)
                        or 'majax' and 'minax' (for elliptical loops). Other optional keys are 'height', 'phi0',
                        'theta0', 'el', 'az', and 'samples_num'.
    :type loop_params: dict
    :return: An `astropy.coordinates.CartesianRepresentation` object containing the
             x, y, and z coordinates of the points defining the semi-circular loop.
    :rtype: astropy.coordinates.CartesianRepresentation
    """

    radius = loop_params.get("radius", 10 * u.Mm)
    height = loop_params.get("height", 0 * u.Mm)
    phi0 = loop_params.get("phi0", 0 * u.deg)
    theta0 = loop_params.get("theta0", 0 * u.deg)
    el = loop_params.get("el", 90 * u.deg)
    az = loop_params.get("az", 0 * u.deg)
    samples_num = loop_params.get("samples_num", 100)

    try:
        loop_coords = semi_circle_loop(radius, 0, 0, False, height, theta0, phi0, el, az, samples_num)[0].cartesian
    except:
        print("Error handled: Your CLB does not support semicircles \n")
        loop_coords = semi_circle_loop(radius, height, theta0, phi0, el, az, samples_num)[0].cartesian

    return loop_coords

def get_loop_params(loop_params, **kwargs):
    """Loads or initializes a dictionary of loop parameters.

    This function attempts to load loop parameters from either a provided dictionary
    or a pickle file. If neither is provided or if there's an error loading, it
    initializes the parameters using keyword arguments or default values.

    :param loop_params: Either a dictionary containing loop parameters or a string
                        path to a pickle file containing such a dictionary. If None
                        or loading fails, parameters are initialized from ``kwargs``.
    :type loop_params: dict or str or None
    :param kwargs: Keyword arguments to override default loop parameters.
        Valid keys include:

        - ``'radius'`` (:class:`astropy.units.quantity.Quantity`): Radius of the loop. Default: 10.0 Mm.
        - ``'majax'`` (:class:`astropy.units.quantity.Quantity`): Semi-major axis of the loop. Default: 0.0 Mm.
        - ``'minax'`` (:class:`astropy.units.quantity.Quantity`): Semi-minor axis of the loop. Default: 0.0 Mm.
        - ``'height'`` (:class:`astropy.units.quantity.Quantity`): Height of the loop. Default: 0.0 Mm.
        - ``'phi0'`` (:class:`astropy.units.quantity.Quantity`): Initial azimuthal angle. Default: 0.0 deg.
        - ``'theta0'`` (:class:`astropy.units.quantity.Quantity`): Initial polar angle. Default: 0.0 deg.
        - ``'el'`` (:class:`astropy.units.quantity.Quantity`): Elevation angle. Default: 90.0 deg.
        - ``'az'`` (:class:`astropy.units.quantity.Quantity`): Azimuthal angle. Default: 0.0 deg.
        - ``'samples_num'`` (:class:`int`): Number of sampling points. Default: 100.

    :raises FileNotFoundError: If ``loop_params`` is a string path to a pickle file
                               and the file does not exist.
    :raises pickle.PickleError: If there is an error during unpickling the
                                 ``loop_params`` file.
    :returns: A dictionary containing the loop parameters. The keys are:
              ``'majax'``, ``'minax'``, ``'radius'``, ``'height'``, ``'phi0'``,
              ``'theta0'``, ``'el'``, ``'az'``, and ``'samples_num'``.
    :rtype: dict

    .. note::
        If ``loop_params`` is a dictionary, it is used directly, provided it contains
        the key ``'phi0'`` as a basic check. If it's a string, it's assumed to be a
        path to a pickle file which is loaded. If ``loop_params`` is None or if
        loading the pickle file fails, the function falls back to using the
        keyword arguments provided in ``kwargs`` or their default values.

        The function prioritizes loading from ``loop_params`` if it's a valid
        dictionary or a readable pickle file. Only if this fails are the ``kwargs``
        and default values used.
    """
    
    try:
        if isinstance(loop_params , dict):
            loop_params_dict = loop_params
            loop_params_dict['phi0']
        else:
            with open(loop_params, 'rb') as f:
                loop_params_dict = pickle.load(f)
                loop_params_dict['phi0']
                f.close()
    except:
        print("No loop coord object provided! Using kwarg (or default) values... \n")

        default_radius = 10.0 * u.Mm
        default_majax = 0.0 * u.Mm
        default_minax = 0.0 * u.Mm
        default_height = 0.0 * u.Mm
        default_phi0 = 0.0 * u.deg
        default_theta0 = 0.0 * u.deg
        default_el = 90.0 * u.deg
        default_az = 0.0 * u.deg
        default_samples_num = 100

        # Set the loop parameters using the provided values or default values
        radius = kwargs.get('radius', default_radius)
        majax = kwargs.get('majax', default_majax)
        minax = kwargs.get('minax', default_minax)
        height = kwargs.get('height', default_height)
        phi0 = kwargs.get('phi0', default_phi0)
        theta0 = kwargs.get('theta0', default_theta0)
        el = kwargs.get('el', default_el)
        az = kwargs.get('az', default_az)
        samples_num = kwargs.get('samples_num', default_samples_num)

        loop_params_dict = {
            'majax':        majax, 
            'minax':        minax,
            'radius':       radius,
            'height':       height,
            'phi0':         phi0,
            'theta0':       theta0,
            'el':           el,
            'az':           az,
            'samples_num':  samples_num
        }

    

    return loop_params_dict

def get_reference_image(smap_path: str = None, smap=None, **kwargs):
    """Loads a reference SunPy Map object.

    :param smap_path: Path to a pickle file containing a SunPy Map object or a path to a
                      standard solar data file (e.g., FITS) that SunPy can read.
    :type smap_path: str, optional
    :param smap: A pre-loaded SunPy Map object. If provided, this takes precedence over
                 `smap_path`.
    :type smap: sunpy.map.Map, optional
    :param kwargs: Keyword arguments passed to :class:`proj_imag_classified.ReferenceImage`
                       if a default reference image needs to be generated.
    :raises FileNotFoundError: If `smap_path` is provided but the file does not exist.
    :raises pickle.PickleError: If `smap_path` points to a pickle file but there is an error during unpickling.
    :raises sunpy.io.header.FileError: If `smap_path` points to a file that SunPy cannot recognize or read.
    :returns: The loaded or generated SunPy Map object.
    :rtype: sunpy.map.Map
    """
    try:
        # If a SunPy Map object is directly provided, use it
        if smap:
            ref_img = smap
        else:
            # Try to load from a pickle file
            try:
                if smap_path:
                    with open(smap_path, 'rb') as f:
                        ref_img = pickle.load(f)
                        f.close()
                else:
                    raise ValueError("No smap_path provided for pickle loading.")
            except:
                # If pickle loading fails, try to load using SunPy's Map function
                if smap_path:
                    ref_img = sunpy.map.Map(smap_path)
                else:
                    raise ValueError("No smap_path provided for SunPy Map loading.")
    except:
        # If all loading attempts fail, generate a default reference image
        print("No reference image provided or loading failed, generating default\n")
        
        from rushlight.utils import proj_imag_classified as prim
        ref_img = prim.ReferenceImage(**kwargs).map

    return ref_img

def code_coords_to_arcsec(code_coord: unyt_array, ref_img: astropy.nddata.NDData = None, **kwargs):
    """Converts coordinates in simulated datcube into arcsecond coordinates from the 
    reference image observer. Assumes that x axis extents in code units are [-.5 to .5] 
    and y axis is changing from 0 to 1.

    :param code_coord: Projected 2D coordinates of synthetic footpoint
    :type code_coord: unyt_array
    :param smap: Sunpy map object
    :type smap: astropy.nddata.NDData
    :return: Arcsecond coordinates in observer's frame of reference
    :rtype: SkyCoord
    """    
    if ref_img == None:
        ref_img = get_reference_image()

    # Take the center coordinates of the base image in arcseconds
    center_x = kwargs.get('center_x', ref_img.center.Tx)
    center_y = kwargs.get('center_y', ref_img.center.Ty)

    # Take the resolution, scale, and coordinate frame from the base image
    # NOTE: Synthetic Image and Reference Image scales should be identical at this point
    resolution = kwargs.get('resolution', ref_img.data.shape)
    scale = kwargs.get('scale', ref_img.scale)
    frame = kwargs.get('frame', ref_img.coordinate_frame)

    # Store the code coordinate units for the anchor point of the image
    # (Should be the foot midpoint) Must be normalized!
    x_code_coord, y_code_coord = code_coord[0], code_coord[1]

    box = kwargs.get('box')
    # NOTE Add an exception for center property (center / domain center)
    # NOTE Take into account scaling of bbox?
    # NOTE subtracting anything from y_code_coord needs to be in code_units!
    x_asec = center_x + (resolution[0] * u.pix * scale[0]) * (x_code_coord - box.domain_center.value[0])
    # x_asec = center_x + resolution[0] * scale[0] * (x_code_coord - 0.5) * u.pix
    # y_asec = center_y + resolution[1] * scale[1] * y_code_coord * u.pix
    # y_asec = center_y + (resolution[1] * u.pix * scale[1]) * (y_code_coord - 0.5)
    y_asec = center_y + (resolution[1] * u.pix * scale[1]) * (y_code_coord - box.domain_center.value[1])

    print(box.domain_center.value[0])
    print(box.domain_center.value[1])


    asec_coords = SkyCoord(x_asec, y_asec, frame=frame) #(x_asec, y_asec)

    return asec_coords

def coord_projection(data, coord: unyt_array, orientation: Orientation=None, **kwargs):
        """Reproduces yt plot_modifications _project_coords functionality

        :param coord: Coordinates of the point in the datacube domain
        :type coord: unyt_array
        :param orientation: Orientation object calculated from norm / north vector, defaults to None
        :type orientation: Orientation, optional
        :return: Cooordinates of the projected point from the viewing camera perspective
        :rtype: tuple
        """     

        # coord_copy should be a unyt array in code_units
        # NOTE coord_copy.transpose() seems to do nothing to coord copy [Generic Dataset]
        # Also, domain_center.v = [0,0,0], so adding does nothing
        coord_copy = coord
        coord_vectors = coord_copy.transpose() - (data.domain_center.v * data.domain_center.uq)

        # orientation object is computed from norm and north vectors
        if orientation:
            unit_vectors = orientation.unit_vectors
        else:
            if 'norm_vector' in kwargs:
                norm_vector = kwargs['norm_vector']
                norm_vec = unyt_array(norm_vector) * data.domain_center.uq
            if 'north_vector' in kwargs:
                north_vector = kwargs['north_vector']
                north_vec = unyt_array(north_vector) * data.domain_center.uq
            if 'north_vector' and 'norm_vector' in kwargs:
                orientation = Orientation(norm_vec, north_vector=north_vec)
                unit_vectors = orientation.unit_vectors

        # NOTE if self.data.domain_center is [0,0,0], then this does nothing
        # Default image extents [-0.5:0.5, 0:1] imposes vertical shift
        y = np.dot(coord_vectors, unit_vectors[1]) + data.domain_center.value[1]
        x = np.dot(coord_vectors, unit_vectors[0])  # data.domain_center.uq

        ret_coord = (x, y) # (y, x)

        return ret_coord



