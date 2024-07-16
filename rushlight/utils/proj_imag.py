import yt
import os
import sys
import numpy as np
from scipy import ndimage #, datasets

from skimage.util import random_noise

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from rushlight.emission_models import uv, xrt, xray_bremsstrahlung
from rushlight.visualization.colormaps import color_tables

# Importing sunpy dependencies for a synthetic map
# See creating custom maps: https://docs.sunpy.org/en/stable/how_to/create_custom_map.html
import sunpy.map
from sunpy.map.mapbase import GenericMap, SpatialPair
import textwrap
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.coordinates.sun import _radius_from_angular_radius

class SyntheticImage():

    """
    Template for a parent class for the Synthetic Images
    """
    # TODO: Should this be an abstract class? So that abstract methods can be explicitly defined in each of the
    #  inheriting classes

    def __init__(self):
        pass


class SyntheticFilterImage():

    """
    Load a yt readable dataset and plot synthetic image having given the instrument name and wavelength
    """

    def __init__(self, dataset, instr, channel, hint=None, units_override=None,
                 view_settings=None, plot_settings=None, **kwargs):
        """
        :param dataset: Path of the downsampled dataset or a dataset itself
        """

        self.box = None  # Importing a region within an initial dataset

        if isinstance(dataset, str):
            self.data = yt.load(dataset, units_override=units_override, hint=hint)
        elif isinstance(dataset, yt.data_objects.static_output.Dataset):
            self.data = dataset
        elif isinstance(dataset, yt.data_objects.selection_objects.region.YTRegion):
            self.data = dataset.ds
            self.box = dataset

        self.instr = instr
        self.obs = kwargs.get('obs', "DefaultInstrument")  # Name of the observatory
        self.obstime = kwargs.get('obstime', '2017-09-10')  # Observation time
        self.channel = channel
        self.view_settings = {'normal_vector': (0.0, 0.0, 1.0),  # pass vectors as mutable arguments
                              'north_vector': (-0.7, -0.3, 0.0)}
        self.__imag_field = None
        self.image = None

        if self.box:
            self.domain_width = np.abs(self.box.right_edge - self.box.left_edge).in_units('cm').to_astropy()
        else:
            self.domain_width = self.data.domain_width.in_units("cm").to_astropy() #convert unyt to astropy.units

        self.plot_settings = {'resolution': 512,
                              'vmin': 1e-15,
                              'vmax': 1e6,
                              'cmap': 'inferno',
                              'logscale': True,
                              'figpath': './prj_plt.png',
                              'frame': None,
                              'label': None}

    def make_filter_image_field(self, **kwargs):

        cmap = {}
        imaging_model = None
        instr_list = ['xrt', 'aia']

        if self.instr not in instr_list:
            raise ValueError("instr should be in the instrument list: ", instr_list)

        if self.instr == 'xrt':
            imaging_model = xrt.XRTModel("temperature", "density", self.channel)
            cmap['xrt'] = color_tables.xrt_color_table()
        if self.instr == 'aia':
            imaging_model = uv.UVModel("temperature", "density", self.channel)
            try:
                cmap['aia'] = color_tables.aia_color_table(int(self.channel) * u.angstrom)
            except ValueError:
                raise ValueError("AIA wavelength should be one of the following:"
                                 "1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335.")

        imaging_model.make_intensity_fields(self.data)

        field = str(self.instr) + '_filter_band'
        self.__imag_field = field

        if self.plot_settings:
            self.plot_settings['cmap'] = cmap[self.instr]


    def proj_and_imag(self, plot_settings=None, view_settings=None, **kwargs):

        self.make_filter_image_field()  # Create emission fields

        if plot_settings:
            self.plot_settings = plot_settings

        if view_settings:
            self.view_settings = view_settings

        if self.box:
            region = self.box
        else:
            region = self.data

        prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
            region,
            [0.0, 0.5, 0.0],  # center position in code units
            normal_vector=self.view_settings['normal_vector'],  # normal vector (z axis)
            width=self.data.domain_width[0].value,  # width in code units
            resolution=self.plot_settings['resolution'],  # image resolution
            item=self.__imag_field,  # respective field that is being projected
            north_vector=self.view_settings['north_vector'])

        # transpose synthetic image (swap axes for imshow)
        self.image = np.array(prji).T

        # Fill background
        self.bkg_fill = kwargs.get('bkg_fill', None)
        
        # return self.image
        self.image_shift = kwargs.get('image_shift', None)  # (xshift, yshift)
        if self.image_shift:
            self.image = np.roll(self.image, (self.image_shift[0],
                                              self.image_shift[1]), axis=(1, 0))
        
        
        self.image_zoom = kwargs.get('image_zoom', None)
        if self.image_zoom:
            self.image = self.zoom_out(self.image, self.image_zoom, self.bkg_fill)
            
        if self.bkg_fill: self.image[self.image <= 0] = self.bkg_fill

    def zoom_out(self, img, scale, bkg_fill=None):
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
        if bkg_fill: new_arr[new_arr <= 0] = bkg_fill
        return new_arr

    def make_synthetic_map(self, **kwargs):

        """
        Creates a synthetic map object that can be loaded/edited with sunpy
        :return:
        """
        data = self.image
        
        
        self.obstime = kwargs.get('obstime')

        # Define header parameters for the synthetic image

        # Coordinates can be passed from sunpy maps that comparisons are made width
        self.reference_coord = kwargs.get('reference_coord', SkyCoord(0*u.arcsec, 0*u.arcsec,
                                   obstime=self.obstime,
                                   observer='earth',  # Temporarily 1 AU away
                                   frame=frames.Helioprojective))

        self.reference_pixel = kwargs.get('reference_pixel', u.Quantity([(data.shape[1] - 1)/2.,
            (data.shape[0] - 1)/2.], u.pixel))  # Reference pixel along each axis: Defaults to the center of data array

        asec2cm = _radius_from_angular_radius(1. * u.arcsec, 1 * u.AU).to(u.cm)  # centimeters per arcsecond at 1 AU
        resolution = self.plot_settings['resolution']
        domain_size = self.domain_width.max()
        len_asec = (domain_size/asec2cm).value
        scale_ = [len_asec/resolution, len_asec/resolution]

        self.scale = kwargs.get('scale', u.Quantity(scale_, u.arcsec/u.pixel))
        self.telescope = kwargs.get('telescope', 'EIT')
        self.observatory = kwargs.get('observatory', 'SOHO')
        self.detector = kwargs.get('detector', 'Synthetic')
        self.instrument = kwargs.get('instrument', None)
        if isinstance(self.channel, int):
            self.wavelength = kwargs.get('wavelength', int(self.channel) * u.angstrom)
        else:
            self.wavelength = kwargs.get('wavelength', None)
        self.exposure = kwargs.get('exposure', None)
        self.unit = kwargs.get('unit', None)
        
        self.poisson = kwargs.get('poisson', None)
        if self.poisson:
            data = 0.5*np.max(self.image) * random_noise(self.image / (0.5*np.max(self.image)), mode='poisson')
            
        if self.bkg_fill: 
            data[data <= 0] = self.bkg_fill
            data[np.isnan(data)] = self.bkg_fill
        
        # Creating header using sunpy
        header = make_fitswcs_header(data,
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

        self.synth_map = sunpy.map.Map(data, header)
        return self.synth_map

    def project_points(self, dataset=None, image=None):
        """
        Identify pixels where three dimensional points from the original dataset are projected
        on the image plane
        TODO: Add markers on the synthetic image object
        :return: x, y -- pixels on which the point inside synthetic datacube projects to
        """
        if dataset is None:
            dataset = self.data #self.region
        if image is None:
            image = self.image

        # Create a dummy PlotMPL plot that will take data and coord system from the initial dataset

        # Use yt function sanitize_coord_system to export x, y values of the point in the image plane



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

#TODO: Write a parent class for Synthetic images that both SyntheticBandImage and SyntheticFilterImage can inherit from,
# so you don't need to describe the same input parameters, such as *dataset*, or *view_settings* and avoid code repetition.
# Get back to this when you will start working on nonthermal emission models, and further on gyrosynchrotron.
# IMPORTANT: proj_and_imag can be inherited from this parent class as well, however exact methods are to be redefined
# Or just inherit from SyntheticFilterImage (?)

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
        field = 'xray_' + str(emin) + '_' + str(emax) + '_keV_band'
        self.__imag_field = field




