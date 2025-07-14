#!/usr/bin/env python
# This script holds the up-to-date versions of all image projection algorithms, made accessible
# through rushlight class objects

from datetime import datetime
import numpy as np
from scipy import ndimage

import yt
from yt.data_objects.selection_objects.region import YTRegion
from yt.utilities.orientation import Orientation
yt.set_log_level(50)

from rushlight.config import config
from rushlight.emission_models import uv, xrt, xray_bremsstrahlung
from rushlight.visualization.colormaps import color_tables
from rushlight.utils import synth_tools as st

from skimage.util import random_noise

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sunpy.map
from sunpy.map.map_factory import MapFactory
from sunpy.coordinates import frames
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.coordinates.sun import _radius_from_angular_radius

from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.constants as const

import pickle
import textwrap
import os
from unyt import unyt_array

from dataclasses import dataclass
from abc import ABC

###############################################################
# Filter Images Classes

#TODO: Write a parent class for Synthetic images that both SyntheticBandImage and SyntheticFilterImage can inherit from,
# so you don't need to describe the same input parameters, such as *dataset*, or *view_settings* and avoid code repetition.
# Get back to this when you will start working on nonthermal emission models, and further on gyrosynchrotron.
# IMPORTANT: proj_and_imag can be inherited from this parent class as well, however exact methods are to be redefined
# Or just inherit from SyntheticFilterImage (?)

@dataclass
class SyntheticImage(ABC):

    """
    Parent class for generating synthetic images
    """

    def __init__(self, dataset = None, smap_path: str=None, smap=None, **kwargs):
        """Object to contain all of the elements of the synthetic image and simulated flare

        :param dataset: Either PATH to the local simulated dataset or a loaded yt object
        :type dataset: _string, yt dataset
        :param smap_path: PATH to the local reference map, defaults to None
        :type smap_path: str, optional
        :param smap: Sunpy map object of the reference map, defaults to None
        :type smap: sunpy.map.Map, optional
        :raises Exception: _description_
        """        

        # Attributes properties of the synthetic loop to the Synthetic Object 
        self.radius, self.majax, self.minax, self.height, self.phi0, \
        self.theta0, self.el, self.az, self.samples_num, self.lat, self.lon \
        = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.dims = {}
        self.set_loop_params(**kwargs)

        # Initializes self.ref_img as either a provided sunpy map
        # or as a generated default map
        self.ref_img = st.get_reference_image(smap_path, smap, **kwargs)

        # Properties extracted from the header metadata of the reference image
        instr = self.ref_img.instrument.split(' ')[0].lower()
        self.instr = kwargs.get('instr', instr).lower()  # keywords: 'aia' or 'xrt'
        if self.instr == 'aia' or self.instr == 'secchi':
            channel_from_meta = self.ref_img.meta.get('wavelnth', 171)  # Default channel for AIA
            self.channel = kwargs.get('channel', channel_from_meta)
        elif self.instr == 'xrt' or self.instr == 'defaultinstrument':
            self.channel = kwargs.get('channel', 'Ti-poly')
        if self.instr == 'secchi' and self.channel == 195:
            self.channel = 193 # Exception for STEREO not having 193 channel
        self.obs = kwargs.get('obs', "DefaultInstrument")  # Name of the observatory

        # Calculation of the CLB loop properties, including the normvector and northvector
        # used to align the MHD box
        self.loop_coords, self.ifpd, self.normvector, self.northvector = (None, None, None, None)
        self.loop_coords = st.get_loop_coords(self.dims)
        self.normvector, self.northvector, self.ifpd = st.calc_vect(self.loop_coords, self.ref_img, default=False)
        
        # Group the normal and north vectors in self.view_settings
        self.view_settings = {'normal_vector': self.normvector,
                              'north_vector': self.northvector}

        # Initialize the 3D MHD file to be used for synthetic image
        shen_datacube = config.SIMULATIONS['DATASET']   # Default datacube TODO make this generic
        if dataset:
            if isinstance(dataset, YTRegion):
                self.box = dataset
                self.data = self.box.ds
                self.domain_width = np.abs(self.box.right_edge - self.box.left_edge).in_units('cm').to_astropy()
            else:
                if isinstance(dataset, str):
                    self.data = yt.load(dataset)
                    self.box = self.data
                else:
                    try:
                        dataset.field_list
                        self.data = dataset
                        self.box = self.data
                    except:
                        print('Invalid datacube provided! Using default datacube... \n')
                        self.data = yt.load(shen_datacube)
                        self.box = self.data
                self.domain_width = np.abs(self.data.domain_right_edge - self.data.domain_left_edge).in_units('cm').to_astropy()
        else:
            print('No datacube provided! Using default datacube... \n')
            self.data = yt.load(shen_datacube)

        # Determine synthetic observation time with respect to observation time
        self.timescale = kwargs.get('timescale', 109.8)
        timestep = self.data.current_time.value.item()
        timediff = TimeDelta(timestep * self.timescale * u.s)
        start_time = Time(self.ref_img.reference_coordinate.obstime, scale='utc', format='isot')
        self.synth_obs_time = start_time + timediff
        self.obstime = kwargs.get('obstime', self.synth_obs_time)  # Can manually specify synthetic box observation time

        # Determines the number of pixels required to shift the synthetic image
        # to align MHD origin with loop foot midpoint. Additionally, determines
        # the lower left pixel of the synthetic image, relative to the lower left pixel
        # of the ref_image.
        # self.zoom, self.image_shift = (None, None)
        # self.diff_roll(**kwargs)

        # Aesthetic settings for the creation of the synthetic image
        self.plot_settings = {'resolution': self.ref_img.data.shape[0],
                              'vmin': kwargs.get('vmin', 1e-15),
                              'vmax': kwargs.get('vmax', 1e6),
                              'norm': colors.LogNorm(kwargs.get('vmin', 1e-15), kwargs.get('vmax', 1e6)),
                              'cmap': 'inferno',
                              'logscale': True,
                              'figpath': './prj_plt.png',
                              'frame': None,
                              'label': None}


        self.imag_field, self.image = (None, None)
        self.proj_and_imag(**kwargs)
        self.make_synthetic_map(**kwargs)
    
    def set_loop_params(self, **kwargs):
        '''
        Initializes the properties of the CLB loop required to position and orient the MHD cube.

        :param pkl: Path to a pickle file containing loop parameters. If provided, other keyword arguments are ignored.
        :type pkl: str, optional
        :param radius: Radius of the circular loop. Used if the loop is circular.
        :type radius: float, optional
        :param majax: Semi-major axis of the elliptical loop. Used if the loop is elliptical.
        :type majax: float, optional
        :param minax: Semi-minor axis of the elliptical loop. Used if the loop is elliptical.
        :type minax: float, optional
        :param height: Height of the loop above the reference plane.
        :type height: float
        :param phi0: Initial azimuthal angle (longitude) of the loop in degrees.
        :type phi0: float
        :param theta0: Initial polar angle (latitude) of the loop in degrees.
        :type theta0: float
        :param el: Elevation angle of the loop's normal vector in degrees.
        :type el: float
        :param az: Azimuthal angle of the loop's normal vector in degrees.
        :type az: float
        :param samples_num: Number of discrete points used to represent the loop.
        :type samples_num: int
        :raises KeyError: If required parameters are missing and no pickle file is provided.
        '''

        # Initialize self.dims (dictionary of loop_params)
        loop_params = kwargs.get('pkl', None)
        self.dims = st.get_loop_params(loop_params, **kwargs)

        # Make each element of self.dims accessible as object property
        try:
            self.radius = self.dims['radius']
        except:
            self.majax = self.dims['majax']
            self.minax = self.dims['minax']
        self.height = self.dims['height']
        self.phi0 = self.dims['phi0']
        self.theta0 = self.dims['theta0']
        self.el = self.dims['el']
        self.az = self.dims['az']
        self.samples_num = self.dims['samples_num']

        # Establish aliases for lat, lon coordinates
        self.lat = self.theta0
        self.lon = self.phi0

    def diff_roll(self, **kwargs):
        """Calculate amount to shift image by difference between observed foot midpoint
        and selected "shift origin"

        :return: Displacement vector x, y
        :rtype: tuple (int , int)
        """

        self.zoom = kwargs.get('zoom', self.scale_factor())

        # Synthetic Foot Midpoint (0,0,0 in code_units)
        north_q = unyt_array(self.northvector, self.data.units.code_length)
        norm_q = unyt_array(self.normvector, self.data.units.code_length)

        ds_orientation = Orientation(norm_q, north_vector=north_q)

        # NOTE synth origin needs to be provided by user
        origin = kwargs.get('origin', [0,0,0])
        synthbox_origin = unyt_array(origin, self.data.units.code_length)
        
        synth_fpt_2d = st.coord_projection(self.data, synthbox_origin, ds_orientation)
        synth_fpt_asec = st.code_coords_to_arcsec(synth_fpt_2d, self.ref_img, box=self.box)
        ori_pix = self.ref_img.wcs.world_to_pixel(synth_fpt_asec)

        if self.zoom and self.zoom < 1:
            # Find coordinates of bottom left corner of "zoom area"
            zoomed_img = ndimage.zoom(self.ref_img.data, self.zoom)  # scale<1
            y, x = self.ref_img.data.shape
            cropx = (zoomed_img.shape[0])
            cropy = (zoomed_img.shape[1])
            startx = (x - cropx) // 2
            starty = (y - cropy) // 2
        else:
            print("Scale parameter has to be lower than 1! Defaulting to self.zoom = 1... \n")
            startx = 0
            starty = 0
            self.zoom = 1

        # Foot Midpoint from CLB
        mpt = SkyCoord(lon=self.lon, lat=self.lat, radius=const.R_sun,
                    frame='heliographic_stonyhurst',
                    observer='earth', obstime=self.obstime).transform_to(frame='helioprojective')
        mpt_pix = self.ref_img.wcs.world_to_pixel(mpt)

        # Find difference between pixel positions
        x1 = float(mpt_pix[0])
        y1 = float(mpt_pix[1])
        # Shift and scale the synthetic coords by zoom
        x2 = float(ori_pix[0]*self.zoom + startx)
        y2 = float(ori_pix[1]*self.zoom + starty)

        x = int((x1-x2))
        y = int((y1-y2))

        # 'noroll' for debugging purposes
        if kwargs.get('noroll', False):
            x = 0
            y = 0

        self.image_shift = (x,y)
        self.start_pix = (startx, starty)

    def make_filter_image_field(self, **kwargs):
        """Selects and applies the correct filter image field to the synthetic dataset

        :raises ValueError: Raised if filter instrument is unrecognized
        :raises ValueError: Raised if AIA wavelength is not from valid selection 
                            (1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335)
        """

        cmap = {}

        print('DefaultInstrument used... Generating xrt intensity_field; self.instr = \'xrt\' \n')
        self.instr = 'xrt'
        imaging_model = xrt.XRTModel("temperature", "density", self.channel)
        cmap['xrt'] = color_tables.xrt_color_table()

        imaging_model.make_intensity_fields(self.data)

        field = str(self.instr) + '_filter_band'
        self.imag_field = field

        if self.plot_settings:
            self.plot_settings['cmap'] = cmap[self.instr]

    class ImageProcessor:
        def __init__(self, image, image_shift):
            self.image = image
            self.image_shift = image_shift

        def roll_and_crop(self):
            # Create the new array filled with the minimum value of the original image
            new_arr = np.ones_like(self.image) * self.image.min()

            xshift = self.image_shift[0]
            yshift = self.image_shift[1]

            # Get the dimensions of the image
            img_height, img_width = self.image.shape

            # Determine the slice boundaries for the original image
            # and the insertion points in new_arr

            # X-direction (columns)
            if xshift > 0:
                # We want the right portion of the original image
                # and place it starting from xshift in new_arr
                src_x_slice = slice(0, img_width - xshift)
                dest_x_slice = slice(xshift, img_width)
            elif xshift < 0:
                # We want the left portion of the original image
                # and place it ending at img_width + xshift in new_arr
                src_x_slice = slice(-xshift, img_width)
                dest_x_slice = slice(0, img_width + xshift)
            else: # xshift == 0
                src_x_slice = slice(0, img_width)
                dest_x_slice = slice(0, img_width)

            # Y-direction (rows)
            if yshift > 0:
                # We want the bottom portion of the original image
                # and place it starting from yshift in new_arr
                src_y_slice = slice(0, img_height - yshift)
                dest_y_slice = slice(yshift, img_height)
            elif yshift < 0:
                # We want the top portion of the original image
                # and place it ending at img_height + yshift in new_arr
                src_y_slice = slice(-yshift, img_height)
                dest_y_slice = slice(0, img_height + yshift)
            else: # yshift == 0
                src_y_slice = slice(0, img_height)
                dest_y_slice = slice(0, img_height)

            # Extract the relevant part from the original image
            sliced_portion = self.image[src_y_slice, src_x_slice]

            # Insert the sliced portion into new_arr at the shifted position
            new_arr[dest_y_slice, dest_x_slice] = sliced_portion

            self.image = new_arr # Update self.image with the new, cropped array
            return self.image

    def proj_and_imag(self, **kwargs):
        """Projects the synthetic dataset and applies image zoom and shift

        :param bkg_fill: Value to fill the background (where image values are less than or equal to 0).
            If None, the background is not filled.
        :type bkg_fill: float, optional

        :notes: The center position is offset by 0.5 in the y-axis, which is dataset-dependent.
            Transposes the synthetic image to swap axes for `imshow`.
            If `self.zoom` is set and not equal to 1, the image is zoomed out.
            If `self.image_shift` is set, the image is shifted by the specified amounts
            along the y and x axes, respectively.
        """

        self.make_filter_image_field()  # Create emission fields

        # NOTE Why is center position offset by 0.5 in y axis? This is dataset-dependent
        prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
            self.box,
            # [0.0, 0.5, 0.0],  # center position in code units
            self.box.domain_center.value,
            normal_vector=self.view_settings['normal_vector'],  # normal vector (z axis)
            width=self.data.domain_width[0].value,  # width in code units
            resolution=self.plot_settings['resolution'],  # image resolution
            item=self.imag_field,  # respective field that is being projected
            north_vector=self.view_settings['north_vector'])

        # transpose synthetic image (swap axes for imshow)
        self.image = np.array(prji).T

        # Determines the number of pixels required to shift the synthetic image
        # to align MHD origin with loop foot midpoint. Additionally, determines
        # the lower left pixel of the synthetic image, relative to the lower left pixel
        # of the ref_image.
        self.zoom, self.image_shift = (None, None)
        self.diff_roll(**kwargs)

        if self.zoom and not (self.zoom == 1):
            self.image = self.zoom_out(self.image, self.zoom)

        if self.image_shift:
            processor1 = self.ImageProcessor(self.image, (self.image_shift[0], self.image_shift[1])) # Shift right by 2, down by 1
            self.image = processor1.roll_and_crop()

            # new_arr = np.ones_like(self.image) * self.image.min()

            # xshift = self.image_shift[0]
            # yshift = self.image_shift[1]
            # self.image = np.roll(self.image, (self.image_shift[0],
            #                                   self.image_shift[1]), axis=(1, 0))
            # if xshift > 0:
            #     self.image = self.image[xshift::, :]


        # Fill background
        self.bkg_fill = kwargs.get('bkg_fill', None)
        if self.bkg_fill: self.image[self.image <= 0] = self.bkg_fill

    def scale_factor(self):
        """
        This function will determine a scale factor to use with zoom_out method based on
        the ratio between the provided observable window scale and the scale of the 
        3D dataset.
        """

        # Base calculations off of x-dimension
        img_pix = self.ref_img.dimensions[0]    # Number of pixels across
        img_scale = self.ref_img.scale[0]       # Number of arcseconds per pixel
        img_as = img_pix * img_scale            # Number of arcseconds across

        synth_lu = self.data.length_unit            # Data length unit
        synth_dw = self.data.domain_width[0].value  # Data domain width
        synth_ludw = synth_lu * synth_dw            # Length of domain (should be km)
        synth_ludw = synth_ludw.to('km')            # Ensure length in km

        synth_as = synth_ludw / (737)           # Assumption 737 km/as for solar feature
        
        factor = float(synth_as / img_as)

        return factor

    def zoom_out(self, img, scale):
        """Move the virtual observer away from from the projected dataset

        :param img: Array-like object containing pixel brightness values
        :type img: numpy.ndarray, other
        :param scale: Value indicating the amount of zoom to apply (<1)
        :type scale: float
        :raises ValueError: Raised if the zoom parameter is not less than one (can only zoom in)
        :return: Zoomed and cropped image array
        :rtype: numpy.ndarray, other
        """

        new_arr = np.ones_like(img) * img.min()
        if scale >= 1:
            raise ValueError("Scale parameter has to be lower than 1")
        zoomed_img = ndimage.zoom(img, scale)  # scale<1

        # fill the central part of the new image
        y, x = new_arr.shape
        cropx = (zoomed_img.shape[0])
        cropy = (zoomed_img.shape[1])
        startx = (x - cropx) // 2
        starty = (y - cropy) // 2
        new_arr[starty:starty + cropy, startx:startx + cropx] = zoomed_img
        return new_arr

    def make_synthetic_map(self, **kwargs):
        """
        Creates a synthetic map object that can be loaded/edited with sunpy

        :return: Synthetic sunpy map created with projected dataset and specified header data
        :rtype: sunpy.map.Map
        """
        # Define header parameters for the synthetic image

        # Coordinates can be passed from sunpy maps that comparisons are made width
        self.reference_coord = self.ref_img.reference_coordinate
        self.reference_pixel = u.Quantity(self.ref_img.reference_pixel)

        asec2cm = _radius_from_angular_radius(1. * u.arcsec, 1 * u.AU).to(u.cm)  # centimeters per arcsecond at 1 AU
        resolution = self.plot_settings['resolution']
        domain_size = self.domain_width.max()
        len_asec = (domain_size/asec2cm).value
        scale_ = [len_asec/resolution, len_asec/resolution]

        self.scale = kwargs.get('scale', u.Quantity(self.ref_img.scale))
        self.telescope = kwargs.get('telescope', self.ref_img.detector)
        self.observatory = kwargs.get('observatory', self.ref_img.observatory)
        self.detector = kwargs.get('detector', self.ref_img.detector)
        self.instrument = kwargs.get('instrument', None)
        self.wavelength = kwargs.get('wavelength', self.ref_img.wavelength)
        self.exposure = kwargs.get('exposure', self.ref_img.exposure_time)
        self.unit = kwargs.get('unit', self.ref_img.unit)
        self.poisson = kwargs.get('poisson', None)

        if self.poisson:
            self.image = 0.5*np.max(self.image) * random_noise(self.image / (0.5*np.max(self.image)), mode='poisson')
        
        # Creating header using sunpy
        header = make_fitswcs_header(self.image,
                                     coordinate=self.reference_coord,
                                     reference_pixel=self.reference_pixel,
                                     scale=self.scale,
                                     telescope=self.telescope,
                                     detector=self.detector,
                                     instrument=self.instrument,
                                     observatory=self.observatory,
                                     wavelength=self.wavelength,
                                     exposure=self.exposure,
                                     unit=self.unit)

        self.synth_map = sunpy.map.Map(self.image, header)

        self.synth_map.plot_settings['norm'] = colors.LogNorm(0.1, self.ref_img.max())
        self.synth_map.plot_settings['cmap'] = self.plot_settings['cmap']

        return self.synth_map

    def project_point(self, y_points):
        """Identify pixels where three dimensional points from the original dataset are projected
        on the image plane

        :param dataset: Dataset containing 3d coordinates for ypoints, defaults to None
        :type dataset: YTGridDataset, optional

        :return: x, y -- pixels on which the point inside synthetic datacube projects to
        """
        # Orientation of synthetic flare from CLB
        north_q = unyt_array(self.northvector, self.data.units.code_length)
        norm_q = unyt_array(self.normvector, self.data.units.code_length)
        ds_orientation = Orientation(norm_q, north_vector=north_q)

        # Sun Center to bottom left pixel displacement
        sc = SkyCoord(lon=0*u.deg, lat=0*u.deg, radius=1*u.cm,
            frame='heliographic_stonyhurst',
            observer='earth', obstime=self.obstime).transform_to(frame='helioprojective')
        sc_pix = self.ref_img.wcs.world_to_pixel(sc)

        sc2bl_x = float(0 - sc_pix[0])
        sc2bl_y = float(0 - sc_pix[1])

        map_ypoints_coords = []
        for ypt in y_points['coordinates']:
            ypt_2d_code = st.coord_projection(self.data, ypt, ds_orientation)
            ypt_2d_asec = st.code_coords_to_arcsec(ypt_2d_code, self.ref_img)
            ypt_coord_pix = self.ref_img.wcs.world_to_pixel(ypt_2d_asec)

            x_shifted = (
                         + float(ypt_coord_pix[0]) * self.zoom
                         + sc2bl_x
                         + self.image_shift[0]
                         + self.start_pix[0]
                         )
            y_shifted = (
                         + float(ypt_coord_pix[1]) * self.zoom
                         + sc2bl_y
                         + self.image_shift[1]
                         + self.start_pix[1]
                         )

            # Save the shifted coords
            shifted = [x_shifted, y_shifted]
            map_ypoints_coords.append(shifted)

        return map_ypoints_coords

    def update_dir(self, norm: unyt_array=None, north: unyt_array=None):
        """Updates the normal and north vectors for the view settings and regenerates the image.

        :param norm: The new normal vector. If not provided, the current normal vector is retained.
        :type norm: unyt_array, optional
        :param north: The new north vector. If not provided, the current north vector is retained.
        :type north: unyt_array, optional
        :returns: A tuple containing the updated normal vector and north vector.
        :rtype: tuple[unyt_array, unyt_array]
        """
        
        change = False  # Initialize a flag to track if changes occurred
        if not np.array_equal(norm, self.normvector):
            self.normvector = norm  # Update the normal vector
            change = True  # Set change flag to True
        if not np.array_equal(north, self.northvector):
            self.northvector = north  # Update the north vector
            change = True  # Set change flag to True
        if change:
            # Update view settings with new vectors
            self.view_settings = {'normal_vector': self.normvector,
                                  'north_vector': self.northvector}
            self.proj_and_imag()  # Re-project the image with new settings
            self.make_synthetic_map()  # Recreate the synthetic map

        return norm, north  # Return the updated vectors

    def save_synthobj(self):
        """Saves relevant synthetic object parameters into a dictionary.

        This method compiles key information about the synthetic object, including FITS header data
        from the reference image, loop parameters, and view vectors, into a dictionary.
        This dictionary is structured for easy identification using a key composed of the
        telescope and observation date.

        :returns: A dictionary containing the synthetic object's data,
            keyed by a string combining the telescope and observation date.
        :rtype: dict
        """
        event_dict = {}  # Initialize an empty dictionary to store event-related data
        event_dict['header'] = self.ref_img.fits_header  # Store the FITS header from the reference image
        event_dict['loop_params'] = self.dims  # Store the loop parameters (dimensions)
        event_dict['norm_vector'] = self.normvector  # Store the normal vector
        event_dict['north_vector'] = self.northvector  # Store the north vector 

        telescope = event_dict['header']['TELESCOP']  # Extract the telescope name from the FITS header
        dateobs = event_dict['header']['DATE-OBS']  # Extract the observation date from the FITS header
        event_key = f'{telescope}|{dateobs}'  # Create a unique key for the event
        synthobj = {event_key: event_dict}  # Create the final dictionary with the event key and data

        return synthobj  # Return the structured synthetic object dictionary
    
    def append_synthobj(self, target=None):
        """Appends the current synthetic object's data to an existing dictionary or a new one,
        then saves it to a file if a file path is provided or creates a new file.

        :param target: The target to append to. Can be a file path (str) to a pickled dictionary,
            an existing dictionary (dir), or None to start with an empty dictionary.
        :type target: str or dict, optional
        :returns: A tuple containing the updated synthetic object dictionary and the target file path.
        :rtype: tuple[dict, str]
        :raises TypeError: If `target` is not a string, a dictionary, or None.
        """
        
        synthobj = {}  # Initialize an empty dictionary for the synthetic object data

        # Load existing data if a target is provided
        if target:
            if isinstance(target, str):
                # If target is a string, assume it's a file path and load the pickled dictionary
                try:
                    with open(target, 'rb') as f:
                        synthobj = pickle.load(f)
                except FileNotFoundError:
                    print(f"File not found: {target}. Creating a new dictionary.")
                    synthobj = {}
                except Exception as e:
                    print(f"Error loading pickle file: {e}. Creating an empty dictionary.")
                    synthobj = {}
            elif isinstance(target, dict):
                # If target is already a dictionary, use it directly
                synthobj = target
            else:
                # Handle invalid target types
                print('Invalid target type! Creating an empty dictionary.')
                synthobj = {}
        else:
            # If no target is provided, start with an empty dictionary
            print('No target provided! Creating an empty dictionary.')
            synthobj = {}

        # Get the current synthetic object's information
        this_synthobj = self.save_synthobj()
        # Extract the key and value from the current synthetic object's data
        key = list(this_synthobj.keys())[0]
        value = this_synthobj[key]
        # Append or update the synthetic object dictionary with the current object's data
        synthobj[key] = value

        # Save the updated synthetic object dictionary
        if isinstance(target, str):
            # If the original target was a string (file path), save back to that file
            with open(target, 'wb') as file:
                pickle.dump(synthobj, file)
        else:
            # If no file path was provided initially, create a new one in 'loop_parameters/'
            now = datetime.datetime.now()  # Get current timestamp
            loop_dir = './loop_parameters/'  # Define the directory for saving
            if not os.path.exists(loop_dir):
                os.makedirs(loop_dir)  # Create the directory if it doesn't exist

            # Generate a unique filename using the timestamp
            fname = f'{now}.pkl'.replace(' ', '_').replace(':', '-').replace('.', '_')
            target = f'{loop_dir}{fname}'  # Construct the full file path
            with open(target, 'wb') as file:
                pickle.dump(synthobj, file)

        return synthobj, target  # Return the updated dictionary and the final target path

    def __str__(self):
        return f"{self._text_summary()}\n{self.data.__repr__()}"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"

    def _text_summary(self):
        # similarly to sunpy.map.mapbase GenericMap(NDData)
        return textwrap.dedent("""\
        Synthetic image from the yt dataset
        ---------
        Instrument:\t\t {inst}
        Wavelength:\t\t {wave}
        """).format(inst=self.instr,
                    wave=self.channel)

@dataclass
class SyntheticFilterImage(SyntheticImage):
    """ For UV and Soft Xrays """
    def __init__(self, dataset=None, smap_path: str=None, smap=None, **kwargs):
        super().__init__(dataset, smap_path, smap, **kwargs)

    def make_filter_image_field(self):
        """Selects and applies the correct filter image field to the synthetic dataset

        :raises ValueError: Raised if filter instrument is unrecognized
        :raises ValueError: Raised if AIA wavelength is not from valid selection 
                            (1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335)
        """

        cmap = {}
        imaging_model = None
        instr_list = ['xrt', 'aia', 'secchi', 'defaultinstrument']

        if self.instr not in instr_list:
            raise ValueError("instr should be in the instrument list: ", instr_list)

        if self.instr == 'xrt':
            imaging_model = xrt.XRTModel("temperature", "density", self.channel)
            cmap['xrt'] = color_tables.xrt_color_table()
        elif self.instr == 'aia':
            imaging_model = uv.UVModel("temperature", "density", self.channel)
            try:
                cmap['aia'] = color_tables.aia_color_table(int(self.channel) * u.angstrom)
            except ValueError:
                raise ValueError("AIA wavelength should be one of the following:"
                                 "1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335.")
        elif self.instr == 'secchi':
            self.instr = 'aia'  # Band-aid for lack of different UV model
            imaging_model = uv.UVModel("temperature", "density", self.channel)
            try:
                cmap['aia'] = color_tables.euvi_color_table(int(self.channel) * u.angstrom)
            except ValueError:
                raise ValueError("AIA wavelength should be one of the following:"
                                 "1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335.")
        elif self.instr == 'defaultinstrument':
            print('DefaultInstrument used... Generating xrt intensity_field; self.instr = \'xrt\' \n')
            self.instr = 'xrt'
            imaging_model = xrt.XRTModel("temperature", "density", self.channel)
            cmap['xrt'] = color_tables.xrt_color_table()

        # Adds intensity fields to the self-contained dataset
        imaging_model.make_intensity_fields(self.data)

        field = str(self.instr) + '_filter_band'
        self.imag_field = field

        if self.plot_settings:
            self.plot_settings['cmap'] = cmap[self.instr]

@dataclass
class SyntheticEnergyRangeImage(SyntheticImage):
    """ For Non-thermal emission, like PyXsim """
    pass

@dataclass
class SyntheticInterferometricImage(SyntheticImage):
    """ For Radio images (CASA)"""
    pass

@dataclass
class SyntheticBandImage():
    """
    Class to store synthetic X-ray images generated in a given *energy band*, such as ones from RHESSI.
    """

    def __init__(self, dataset, emin, emax, nbins, emission_model, hint=None, units_override=None,
                 view_settings=None, plot_settings=None, **kwargs):

        """
        :param dataset: Path of the downsampled dataset or a dataset itself
        :param emin: low energy limit in keV
        :param emax: high energy limit in keV
        :param nbins: number of energy bins to compute synthetic spectra and images
        """

        self.emin = emin
        self.emax = emax
        self.nbins = nbins
        self.emission_model = emission_model  # Thermal / Non-thermal

        self.box = None  # Importing a region within an initial dataset

        if isinstance(dataset, str):
            self.data = yt.load(dataset, units_override=units_override, hint=hint)
        elif isinstance(dataset, yt.data_objects.static_output.Dataset):
            self.data = dataset
        elif isinstance(dataset, yt.data_objects.selection_objects.region.YTRegion):
            self.data = dataset.ds
            self.box = dataset

        self.obs = kwargs.get('obs', "DefaultInstrument")  # Name of the observatory
        self.instr = kwargs.get('instr', 'RHESSI')
        self.binscale = kwargs.get('binscale', 'linear')
        self.obstime = kwargs.get('obstime', '2017-09-10')  # Observation time

        self.view_settings = {'normal_vector': (0.0, 0.0, 1.0),  # pass vectors as mutable arguments
                              'north_vector': (-0.7, -0.3, 0.0)}
        self.imag_field = None
        self.image = None

        if self.box:
            self.domain_width = np.abs(self.box.right_edge - self.box.left_edge).in_units('cm').to_astropy()
        else:
            self.domain_width = self.data.domain_width.in_units("cm").to_astropy()  # convert unyt to astropy.units

    def make_band_image_field(self, **kwargs):

        cmap = {}
        imaging_model = None

        if self.emission_model == 'Thermal':
            imaging_model = xray_bremsstrahlung.ThermalBremsstrahlungModel

        imaging_model.make_intensity_fields(self.data)
        field = 'xray_' + str(self.emin) + '_' + str(self.emax) + '_keV_band'
        self.imag_field = field

###############################################
# Reference Image Classes

@dataclass
class ReferenceImage(ABC, MapFactory):
    """
    Default object for reference image types
    """

    def __init__(self, ref_img_path: str = None, **kwargs):
        """Constructor for the default reference image object

        :param ref_img_path: Path to the reference image .fits file, defaults to None
        :type ref_img_path: str, optional
        """
        reference_image = None

        if ref_img_path:
            m = sunpy.map.Map(ref_img_path) 
        else: 
            import datetime

            # Create an empty dataset
            resolution = 1000
            # data = np.full((resolution, resolution), np.random.randint(100))
            data = np.random.randint(0, 1e6, size=(resolution, resolution)) 

            obstime = datetime.datetime(2000,1,1,0,0,0)
            # Define a reference coordinate and create a header using sunpy.map.make_fitswcs_header
            skycoord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=obstime,
                                observer='earth', frame=frames.Helioprojective)
            # Scale set to the following for solar limb to be in the field of view
            # scale = 220 # Changes bounds of the resulting helioprojective view
            scale = kwargs.get('refmap_scale', 1)
            
            instr = kwargs.get('instrument', 'DefaultInstrument')
            self.instrument = instr

            header_kwargs = {
                'scale': [scale, scale]*u.arcsec/u.pixel,
                'telescope': instr,
                'detector': instr,
                'instrument': instr,
                'observatory': instr,
                'exposure': 0.01 * u.s,
                'unit': u.Mm
            }

            header = make_fitswcs_header(data, skycoord, **header_kwargs)
            default_kwargs = {'data': data, 'header': header}
            m = sunpy.map.Map(data, header)        

        self.map = m

@dataclass
class XRTReferenceImage(ReferenceImage):
    """
    XRT instrument variant of default reference image object
    """

    def __init__(self, ref_img_path: str = None):
        super().__init__(ref_img_path, instrument='Xrt')

@dataclass
class AIAReferenceImage(ReferenceImage):
    """
    AIA instrument variant of default reference image object
    """

    def __init__(self, ref_img_path):
        super().__init__(ref_img_path)