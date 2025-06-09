from sunpy.net import Fido
from sunpy.net import attrs as a
import astropy.units as u
from astropy.time import Time
from rushlight.config import config

'''
This script provides functionality to download FITS data and apply calibrations to the images
provided by different solar observatories, including SDO/AIA, Hinode/XRT
'''


def get_sdo_aia_data(start_time=Time('2011-03-07T18:06:04', scale='utc', format='isot'),
                     channel: int = 171,
                     duration: int = 24,
                     jsoc_email=None,
                     **kwargs):

    """
    Downloads SDO/AIA data from JSOC database
    :param start_time: Observations start time
    :param channel: AIA channel
    :param duration: Observations duration in seconds
    :param jsoc_email: email address to access JSOC database
    :param kwargs: Optional keyword arguments
    :return:
    """

    if jsoc_email is None:
        jsoc_email = config['JSOC_EMAIL']

    wavelength = channel*u.angstrom
    query = Fido.search(
        a.Time(start_time - 0*u.h, start_time + duration*u.s),
        a.Wavelength(wavelength),
        a.Sample(12*u.s),
        a.jsoc.Series.aia_lev1_euv_12s,
        a.jsoc.Notify(jsoc_email),
        #a.jsoc.Segment.image,
        #cutout,
    )

    print(query)
    # by default files wil be downloaded to ~/sunpy/data folder
    files = Fido.fetch(query)
    files.sort()

def get_hinode_xrt_data(**kwargs):
    pass
