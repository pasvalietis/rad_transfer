#!/usr/bin/env python
# This script holds the up-to-date versions of all image projection algorithms, made accessible
# through rushlight class objects

import numpy as np
from scipy import ndimage

import yt
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

from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.time import Time, TimeDelta
import astropy.constants as const

import pickle
import textwrap
import os
import sys
sys.path.insert(1, config.CLB_PATH)
from CoronalLoopBuilder.builder import semi_circle_loop # type: ignore
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

        #self.loop_coords, self.ifpd, self.normvector, self.northvector = (None, None, None, None)
        #self.loop_coords = st.get_loop_coords(self.dims)
        #self.normvector, self.northvector, self.ifpd = st.calc_vect(self.loop_coords, self.ref_img, default=False)

        self.loop_coords = st.get_loop_coords(self.dims)

        if 'normvector' in kwargs and 'northvector' in kwargs:
            self.normvector, self.northvector = kwargs['normvector'], kwargs['northvector']
        else:
            self.loop_coords, self.ifpd, self.normvector, self.northvector = (None, None, None, None)
            self.normvector, self.northvector, self.ifpd = st.calc_vect(self.loop_coords, self.ref_img, default=False)

        # Group the normal and north vectors in self.view_settings
        self.view_settings = {'normal_vector': self.normvector,
                              'north_vector': self.northvector}

        # Initialize the 3D MHD file to be used for synthetic image
        shen_datacube = config.SIMULATIONS['DATASET']   # Default datacube TODO make this generic
        if dataset:
            if isinstance(dataset, str):
                    self.data = yt.load(dataset)
            else:
                try:
                    dataset.field_list
                    self.data = dataset
                except:
                    print('Invalid datacube provided! Using default datacube... \n')
                    self.data = yt.load(shen_datacube)
        else:
            print('No datacube provided! Using default datacube... \n')
            self.data = yt.load(shen_datacube)

        # TODO Remove this part of code (Crops bottom slice of box out by default)
        center = [0.0, 0.5, 0.0]
        left_edge = [-0.5, 0.005, -0.25]
        right_edge = [0.5, 1.0, 0.25]
        self.box = self.data.region(center=kwargs.get('center', center),
                                left_edge=kwargs.get('left_edge', left_edge),
                                right_edge=kwargs.get('right_edge', right_edge))

        # Define self.domain_width for later reference
        self.domain_width = np.abs(self.box.right_edge - self.box.left_edge).in_units('cm').to_astropy()

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
        self.zoom, self.image_shift = (None, None)
        self.diff_roll(**kwargs)

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


        self.__imag_field, self.image = (None, None)
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

        self.zoom = kwargs.get('zoom', None)

        # Synthetic Foot Midpoint (0,0,0 in code_units)
        north_q = unyt_array(self.northvector, self.data.units.code_length)
        norm_q = unyt_array(self.normvector, self.data.units.code_length)

        ds_orientation = Orientation(norm_q, north_vector=north_q)
        synthbox_origin = unyt_array([0,0,0], self.data.units.code_length)
        synth_fpt_2d = self.coord_projection(synthbox_origin, ds_orientation)
        synth_fpt_asec = st.code_coords_to_arcsec(synth_fpt_2d, self.ref_img, box=self.data)
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

    def synthmap_plot(self, fig: plt.figure=None, plot: str=None, **kwargs):
        """Plot the generated synthetic map in different configurations

        :param fig: matplotlib figure object, defaults to None
        :type fig: plt.figure, optional
        :param plot: Hint for what type of plot to produce ['comp', 'synth', 'obs'], defaults to None
        :type plot: str, optional
        :return: Synthetic map object, normvector, northvector, and image shift
        :rtype: tuple
        """
        self.synth_map.plot_settings['norm'] = colors.LogNorm(0.1, self.ref_img.max())
        self.synth_map.plot_settings['cmap'] = self.plot_settings['cmap']

        if fig:
            if plot == 'comp':
                comp_map = sunpy.map.Map(self.ref_img, self.synth_map, composite=True)

                alpha = kwargs.get('alpha', 0.5)
                comp_map.set_alpha(1, alpha)
                ax = fig.add_subplot(projection=comp_map.get_map(0))
                comp_map.plot(axes=ax)
            elif plot == 'synth':
                ax = fig.add_subplot(projection=self.synth_map)
                ax.grid(False)
                self.synth_map.plot(axes=ax)

                debug = kwargs.get('debug', False)
                if debug:
                    # Plot Synthetic Footpont
                    north_q = unyt_array(self.northvector, self.data.units.code_length)
                    norm_q = unyt_array(self.normvector, self.data.units.code_length)
                    ds_orientation = Orientation(norm_q, north_vector=north_q)

                    synthbox_origin = unyt_array([0,0,0], self.data.units.code_length)
                    synth_fpt_2d = self.coord_projection(synthbox_origin, ds_orientation)
                    synth_fpt_asec = st.code_coords_to_arcsec(synth_fpt_2d, self.ref_img)
                    ori_pix = self.ref_img.wcs.world_to_pixel(synth_fpt_asec)
                    ax.plot(ori_pix[0] * self.zoom, ori_pix[1] * self.zoom, 'og')
                    ax.text(ori_pix[0] * self.zoom, ori_pix[1] * self.zoom, 'origin', color='g')

                    # Plot Observed Footpoint
                    mpt = SkyCoord(lon=self.lon, lat=self.lat, radius=const.R_sun,
                        frame='heliographic_stonyhurst',
                        observer='earth', obstime=self.obstime).transform_to(frame='helioprojective')
                    mpt_pix = self.ref_img.wcs.world_to_pixel(mpt)
                    ax.plot(mpt_pix[0], mpt_pix[1], 'or')
                    ax.text(mpt_pix[0], mpt_pix[1], 'midpoint', color='r')

                    # Plot Sun center
                    sc = SkyCoord(lon=0*u.deg, lat=0*u.deg, radius=1*u.cm,
                        frame='heliographic_stonyhurst',
                        observer='earth', obstime=self.obstime).transform_to(frame='helioprojective')
                    sc_pix = self.ref_img.wcs.world_to_pixel(sc)
                    ax.plot(sc_pix[0], sc_pix[1], 'oc')
                    ax.text(sc_pix[0], sc_pix[1], 'sun center', color='c')

                    # Plot (0,0) [correlates to bl of image]
                    ax.plot(0,0,'oy')
                    ax.text(0,0, 'bottom left', color='y')
            elif plot == 'obs':
                ax = fig.add_subplot(projection=self.ref_img)
                self.ref_img.plot(axes=ax)
            else:
                ax = fig.add_subplot(projection=self.synth_map)

            return ax, self.synth_map, self.normvector, self.northvector, self.image_shift

        else:

            return self.synth_map, self.normvector, self.northvector, self.image_shift

    def make_filter_image_field(self, **kwargs):
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
            imaging_model = xrt.XRTModel("temperature", "number_density", self.channel)
            cmap['xrt'] = color_tables.xrt_color_table()
        elif self.instr == 'aia':
            imaging_model = uv.UVModel("temperature", "number_density", self.channel)
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
        self.__imag_field = field

        if self.plot_settings:
            self.plot_settings['cmap'] = cmap[self.instr]

    def proj_and_imag(self, **kwargs):
        """Projects the synthetic dataset and applies image zoom and shift"""

        self.make_filter_image_field()  # Create emission fields

        prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
            self.box,
            [0.0, 0.5, 0.0],  # center position in code units
            normal_vector=self.view_settings['normal_vector'],  # normal vector (z axis)
            width=self.data.domain_width[0].value,  # width in code units
            resolution=self.plot_settings['resolution'],  # image resolution
            item=self.__imag_field,  # respective field that is being projected
            north_vector=self.view_settings['north_vector'])

        # transpose synthetic image (swap axes for imshow)
        self.image = np.array(prji).T

        if self.zoom and not (self.zoom == 1):
            self.image = self.zoom_out(self.image, self.zoom)

        # return self.image
        if self.image_shift:
            self.image = np.roll(self.image, (self.image_shift[0],
                                              self.image_shift[1]), axis=(1, 0))

        # Fill background
        self.bkg_fill = kwargs.get('bkg_fill', None)
        if self.bkg_fill: self.image[self.image <= 0] = self.bkg_fill

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
            ypt_2d_code = self.coord_projection(ypt, ds_orientation)
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

    def coord_projection(self, coord: unyt_array, orientation: Orientation=None, **kwargs):
        """Reproduces yt plot_modifications _project_coords functionality

        :param coord: Coordinates of the point in the datacube domain
        :type coord: unyt_array
        :param orientation: Orientation object calculated from norm / north vector, defaults to None
        :type orientation: Orientation, optional
        :return: Cooordinates of the projected point from the viewing camera perspective
        :rtype: tuple
        """

        # coord_copy should be a unyt array in code_units
        coord_copy = coord
        coord_vectors = coord_copy.transpose() - (self.data.domain_center.v * self.data.domain_center.uq)

        # orientation object is computed from norm and north vectors
        if orientation:
            unit_vectors = orientation.unit_vectors
        else:
            if 'norm_vector' in kwargs:
                norm_vector = kwargs['norm_vector']
                norm_vec = unyt_array(norm_vector) * self.data.domain_center.uq
            if 'north_vector' in kwargs:
                north_vector = kwargs['north_vector']
                north_vec = unyt_array(north_vector) * self.data.domain_center.uq
            if 'north_vector' and 'norm_vector' in kwargs:
                orientation = Orientation(norm_vec, north_vector=north_vec)
                unit_vectors = orientation.unit_vectors

        # Default image extents [-0.5:0.5, 0:1] imposes vertical shift
        y = np.dot(coord_vectors.value, unit_vectors[1].value) + self.data.domain_center.value[1]
        x = np.dot(coord_vectors.value, unit_vectors[0].value)  # * self.data.domain_center.uq

        ret_coord = (x, y) # (y, x)

        return ret_coord

    def save_synthobj(self):
        event_dict = {}
        event_dict['header'] = self.ref_img.fits_header
        event_dict['loop_params'] = self.dims
        event_dict['norm_vector'] = self.normvector
        event_dict['norm_vector'] = self.northvector

        telescope = event_dict['header']['TELESCOP']
        dateobs = event_dict['header']['DATE-OBS']
        event_key = f'{telescope}|{dateobs}'
        synthobj = {event_key: event_dict}

        return synthobj

    def append_synthobj(self, target=None):
        if target:
            if isinstance(target, str):
                with open(target, 'rb') as f:
                    synthobj = pickle.load(f)
                    f.close()
            elif isinstance(target, dir):
                synthobj = target
            else:
                print('Invalid target! Creating empty dict...\n')
                synthobj = {}
        else:
            print('No target! Creating empty dict...\n')
            synthobj = {}

        # Appends this sfi object's information to the provided dictionary
        this_synthobj = self.save_synthobj()
        key = list(this_synthobj.keys())[0]
        value = this_synthobj[key]
        synthobj[key] = value

        if isinstance(target, str):
            with open(target, 'wb') as file:
                pickle.dump(synthobj, file)
                file.close()
        else:
            import datetime
            now = datetime.datetime.now()

            loop_dir = './loop_parameters/'
            if not os.path.exists(loop_dir):
                os.makedirs(loop_dir)

            fname = f'{now}.pkl'.replace(' ', '_')
            target = f'{loop_dir}{fname}'
            with open(target, 'wb') as file:
                pickle.dump(synthobj, file)
                file.close()

        return synthobj, target

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
        self.__imag_field = None
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
        self.__imag_field = field

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
            resolution = 194
            # data = np.full((resolution, resolution), np.random.randint(100))
            data = np.random.randint(0, 1e6, size=(resolution, resolution))

            obstime = datetime.datetime(2000, 1, 1, 0, 0, 0)
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