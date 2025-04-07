import numpy as np

from rushlight.config import config

import sys
sys.path.insert(1, config.CLB_PATH)
from CoronalLoopBuilder.builder import semi_circle_loop # type: ignore

from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u

def calc_vect(loop_coords, ref_img, **kwargs):
        """Calculates the north and normal vectors for the synthetic image

        :raises Exception: Null reference to kwargs member
        :return: norm, north, lat, lon, radius, height, ifpd
        :rtype: tuple (list, list, Quantity, Quantity, Quantity, Quantity, float)
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

    radius = loop_params.get("radius", 10 * u.Mm)
    height = loop_params.get("height", 0 * u.Mm)
    phi0 = loop_params.get("phi0", 0 * u.deg)
    theta0 = loop_params.get("theta0", 0 * u.deg)
    el = loop_params.get("el", 90 * u.deg)
    az = loop_params.get("az", 0 * u.deg)
    samples_num = loop_params.get("samples_num", 100)

    # Define the vectors v1 and v2 (from center of sun to footpoints)
    try:
        loop_coords = semi_circle_loop(radius, 0, 0, False, height, theta0, phi0, el, az, samples_num)[0].cartesian
    except:
        print("Error handled: Your CLB does not support semicircles \n")
        loop_coords = semi_circle_loop(radius, height, theta0, phi0, el, az, samples_num)[0].cartesian

    return loop_coords
