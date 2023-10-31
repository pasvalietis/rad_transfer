from astropy.coordinates import SkyCoord
import astropy.units as u

import sunpy
from sunpy.coordinates import frames
from sunpy.coordinates.sun import _radius_from_angular_radius
from sunpy.map.header_helper import make_fitswcs_header

self = {'image': None,
        'plotSetting_resolution': None,
        'domainWidth_max': None,
        'channel': None,
        'reference_coord': None}

kgs = {'reference_coord': None,
       'reference_pixel': None,
       'scale': None,
       'telescope': None,
       'observatory': None,
       'detector': None,
       'instrument': None,
       'exposure': None,
       'unit': None}

def make_synthetic_map(self, **kwargs):
    """
    Creates a synthetic map object that can be loaded/edited with sunpy
    :return:
    """
    data = self.image

    # Define header parameters for the synthetic image

    # Coordinates can be passed from sunpy maps that comparisons are made width
    self.c = kwargs.get('reference_coord', SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                                                                  obstime='2013-10-28',
                                                                  observer='earth',  # Temporarily 1 AU away
                                                                  frame=frames.Helioprojective))

    self.reference_pixel = kwargs.get('reference_pixel', u.Quantity([(data.shape[1] - 1) / 2.,
                                                                     (data.shape[0] - 1) / 2.],
                                                                    u.pixel))  # Reference pixel along each axis: Defaults to the center of data array

    asec2cm = _radius_from_angular_radius(1. * u.arcsec, 1 * u.AU).to(u.cm)  # centimeters per arcsecond at 1 AU
    resolution = self.plot_settings['resolution']
    domain_size = self.domain_width.max()
    len_asec = (domain_size / asec2cm).value
    scale_ = [len_asec / resolution, len_asec / resolution]

    self.scale = kwargs.get('scale', u.Quantity(scale_, u.arcsec / u.pixel))
    self.telescope = kwargs.get('telescope', 'EIT')
    self.observatory = kwargs.get('observatory', 'SOHO')
    self.detector = kwargs.get('detector', 'Synthetic')
    self.instrument = kwargs.get('instrument', None)
    if isinstance(self.channel, int):
        self.wavelength = int(self.channel) * u.angstrom
    else:
        self.wavelength = None
    self.exposure = kwargs.get('exposure', None)
    self.unit = kwargs.get('unit', None)

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


