import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import frames, Helioprojective, HeliographicStonyhurst, Heliocentric, get_earth
import sunpy.data.sample
from sunpy.map import make_fitswcs_header

from aiapy.calibrate import update_pointing, register, correct_degradation, normalize_exposure

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as const

from ipywidgets import *
import matplotlib.pyplot as plt
import numpy as np

from CoronalLoopBuilder.builder import CoronalLoopBuilder # type: ignore
from rushlight.utils.proj_imag_classified import SyntheticFilterImage as sfi
from rushlight.utils.proj_imag_classified import XRTReferenceImage
from rushlight.config import config

import itertools
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

# If the provided params path points to a nested dictionary, unpack those values.
def extract_loop_params(file_path):
  """
  Extracts 'loop_params' dictionaries from a nested dictionary in a pickle file.

  Args:
    file_path: Path to the pickle file.

  Returns:
    A dictionary where keys are the keys from the original dictionary 
    and values are the corresponding 'loop_params' dictionaries.
  """
  try:
    with open(file_path, 'rb') as f:
      data = pickle.load(f)

    loop_params_dict = {}
    for key, value in data.items():
      if isinstance(value, dict) and 'loop_params' in value:
        loop_params_dict[key] = value['loop_params']

    return loop_params_dict

  except FileNotFoundError:
    print(f"File not found: {file_path}")
    return {}
  except Exception as e:
    print(f"An error occurred: {e}")
    return {}
  
# If the provided params path points to a nested dictionary, select one of the loop_parameters for display.
def select_param_year(loop_params_dict: dict, year: str = '', hour: str = ''):
  """ 
  Search the nested dictionary structure for the specified year and hour

  :param loop_params_dict: Nested loop parameters dictionary
  :type loop_params_dict: dict
  :param year: Year of the saved loop parameters
  :type year: str, int, optional
  :param hour: Hour of the saved loop parameters
  :type hour: str, int, optional
  :return: Matching entry from the nested dictionary
  :rtype: dict
  """
  
  for n, key in enumerate(loop_params_dict.keys()):
    if (str(year) in key) and (f"T{hour:02}" in key):
      return list(loop_params_dict.values())[n]
  return None

# Ensures that the user enters an input
def get_user_input(prompt):
  """
  Prompts the user for input and returns the sanitized input.
  """
  while True:
    user_input = input(prompt)
    print(user_input)
    if user_input:
      return user_input.strip()
    else:
      print('No Input!')

def apply_time_window(result, minutes=None):
  if minutes:
    h = round(minutes / 60)
    m = minutes % 60
    # if h < 10: h = f'0{h}'
    # if m < 10: m = f'0{m}'
    # window = f'{h}:{m}'
    window = f'{h:02d}:{m:02d}'
  else:
    window = get_user_input("Please enter a time range around the STEREO image that you would like to search (HH:MM):")
    
  start = str(result[0]["Start Time"][0])
  date = start.split(" ")[0]
  tstring = start.split(" ")[1]

  w_hour = int(window.split(':')[0])
  w_min  = int(window.split(':')[1])

  t_hour = int(tstring.split(':')[0])
  t_min  = int(tstring.split(':')[1])

  # Calculate new start time with hour adjustments
  new_start_hour = t_hour - w_hour
  new_start_min = t_min - w_min

  # Adjust for minutes going below 0
  if new_start_min < 0:
      new_start_hour -= 1
      new_start_min += 60

  # Adjust for minutes going above 60
  elif new_start_min >= 60:
      new_start_hour += 1
      new_start_min -= 60

  # Ensure hour is within 0-23 range
  new_start_hour = (new_start_hour + 24) % 24 

  new_start = f'{date} {new_start_hour:02d}:{new_start_min:02d}' 

  # Calculate new end time with hour adjustments
  new_end_hour = t_hour + w_hour
  new_end_min = t_min + w_min

  # Adjust for minutes going below 0
  if new_end_min < 0:
      new_end_hour -= 1
      new_end_min += 60

  # Adjust for minutes going above 60
  elif new_end_min >= 60:
      new_end_hour += 1
      new_end_min -= 60

  # Ensure hour is within 0-23 range
  new_end_hour = (new_end_hour + 24) % 24

  new_end = f'{date} {new_end_hour:02d}:{new_end_min:02d}'

  time = a.Time(new_start, new_end)

  return time

def approx_stereo(stereo_result, wav):
  n_aia = 0
  mins = 0
  while n_aia == 0:
    mins += 10
    print(f'No AIA in stereo range! \nAttempting search in {mins}-minute window...\n')
    time = apply_time_window(stereo_result, mins)
    aia_result = Fido.search(time, a.Instrument.aia, a.Sample(1*u.minute), a.Physobs('Intensity'), wav)
    n_aia = aia_result.__dict__['_numfile']
  
  return aia_result

# Finds pairs of FITS files with different instruments but equal filter wavelengths
def find_matching_fits(directory):
  """
  Finds pairs of FITS files with different instruments but equal filter wavelengths.

  Args:
    directory: Path to the directory containing the FITS files.

  Returns:
    A list of tuples, where each tuple contains the paths to two FITS files 
    with different instruments but equal filter wavelengths.
  """

  fits_files = glob.glob(os.path.join(directory, '*.fits')) + glob.glob(os.path.join(directory, '*.fts'))
  fits_maps = [sunpy.map.Map(f) for f in fits_files]

  matches = []
  for i, map1 in enumerate(fits_maps):
    for j, map2 in enumerate(fits_maps):
      different_instruments = map1.meta['TELESCOP'] != map2.meta['TELESCOP']
      same_wavelength = (map1.meta['WAVELNTH'] == map2.meta['WAVELNTH'])
      almost = (map1.meta['WAVELNTH'] == 195 and map2.meta['WAVELNTH'] == 193) \
            or (map1.meta['WAVELNTH'] == 193 and map2.meta['WAVELNTH'] == 195)
      if i < j and different_instruments and (same_wavelength or almost):
        matches.append((fits_files[i], fits_files[j]))

  return matches

def select_pair_by_wavelength(matching_pairs, target_wavelength):
  """
  Selects the pair of FITS files from the given list that corresponds to the specified target wavelength.

  Args:
    matching_pairs: A list of tuples, where each tuple contains the paths to two FITS files 
                    with different instruments but equal filter wavelengths.
    target_wavelength: The target wavelength to search for.

  Returns:
    The first pair of FITS files found with the specified target wavelength, 
    or None if no such pair exists.
  """
  for pair in matching_pairs:
    map1 = sunpy.map.Map(pair[0])
    match_target = map1.meta['WAVELNTH'] == target_wavelength
    almost = map1.meta['WAVELNTH'] == 195 and target_wavelength == 193 \
          or map1.meta['WAVELNTH'] == 193 and target_wavelength == 195
    if match_target or almost: 
      return pair  # Only need to check one map in the pair
  return None

# Crop the maps to ROI
def crop_map(m = None, blox: float=0, bloy: float=0, trox: float=0, troy: float=0):
    """
    Crops a given map to the specified rectangular region defined by bottom-left and top-right coordinates.
    Parameters:
    m (Map): The map to be cropped. It should be an instance of a map object with a coordinate frame.
    blox (float): The x-coordinate of the bottom-left corner in arcseconds. Default is 0.
    bloy (float): The y-coordinate of the bottom-left corner in arcseconds. Default is 0.
    trox (float): The x-coordinate of the top-right corner in arcseconds. Default is 0.
    troy (float): The y-coordinate of the top-right corner in arcseconds. Default is 0.
    Returns:
    Map: A new map object that is the cropped version of the input map.
    """

    blo = [blox * u.arcsec, bloy * u.arcsec]
    tro = [trox * u.arcsec, troy * u.arcsec]

    bottom_left = SkyCoord(blo[0],blo[1], frame=m.coordinate_frame)
    top_right = SkyCoord(tro[0],tro[1], frame=m.coordinate_frame)
    
    cropped_map = m.submap(bottom_left=bottom_left, top_right=top_right)
    
    return cropped_map

def calc_lims(crop_lims, roi_map):
    """
    Calculate pixel limits for display.
    Parameters:
    crop_lims (dict): A dictionary containing the crop limits with keys 'blox', 'trox', 'bloy', and 'troy'.
                      These values represent the bottom-left and top-right coordinates in arcseconds.
    roi_map (object): A map object that contains the region of interest and its coordinate frame.
    Returns:
    tuple: A tuple containing the pixel coordinates corresponding to the specified limits in the region of interest map.
    """

    xlim = [crop_lims['blox'], 
            crop_lims['trox']] * u.arcsec
    ylim = [crop_lims['bloy'], 
            crop_lims['troy']] * u.arcsec
    lims = SkyCoord(Tx=xlim, Ty=ylim, frame=roi_map.coordinate_frame)

    return roi_map.wcs.world_to_pixel(lims)

def plot_aia_los(ax, aia_map, **kwargs):
    
    # Extracting data from the map
    time = aia_map.reference_coordinate.obstime         # Time of Observation
    observer = aia_map.reference_coordinate.observer    # Observer location (HGS)

    # Define various frames
    hgs_frame = frames.HeliographicStonyhurst(obstime=time)     # Heliographic Stonyhurst frame
    hpj_frame = frames.Helioprojective(obstime=time,            # Helioprojective frame
                                    observer=observer)

    # Define loop coordinates in hgs frame
    r_1 = const.R_sun   # Solar radius

    # Setting CLB loop (if provided)
    dummyParms = {'theta0': 0 * u.deg, 'phi0': 0 * u.deg, 'height': 30 * u.Mm}
    loop_params = kwargs.get('loop_params', dummyParms)
    lat = loop_params.get('theta0', 0 * u.deg) 
    lon = loop_params.get('phi0', 0 * u.deg) 
    height = loop_params.get('height', 30 * u.Mm) 

    # Determining los target
    target = kwargs.get("target", "bottom")
    if target == "bottom":
      hgs_coord = SkyCoord(lon=lon, lat=lat, radius=r_1, frame=hgs_frame)  # longitude, latitude, and distance from the origin
    else:
      # First attempt to use sfiObj, then loop_params as available
      sfiObj = kwargs.get('sfiObj', None)
      if sfiObj:
        # 1. Define observer and box parameters
        # observer = get_earth(time)
        dom_width = sfiObj.domain_width * (1-sfiObj.zoom)
        box_dims = u.Quantity([
            dom_width[2],
            dom_width[1],
            dom_width[0],
        ])
        box_origin = SkyCoord(lon=sfiObj.lon, lat=sfiObj.lat, radius=r_1,
                        frame='heliographic_stonyhurst',
                        observer=observer, obstime=time)
        frame_hcc = Heliocentric(observer=box_origin, obstime=time)
        box_origin_hcc = box_origin.transform_to(frame_hcc)
        hgs_coord = SkyCoord(x=box_origin_hcc.x,
                                y=box_origin_hcc.y,
                                z=box_origin_hcc.z + box_dims[2],
                                frame=box_origin_hcc.frame).transform_to(hgs_frame)
      else:
        hgs_coord = SkyCoord(lon=lon, lat=lat, radius=r_1 + height, frame=hgs_frame)  # longitude, latitude, and distance from the origin
    
    # Renaming
    flare_center_los_hgs = hgs_coord

    # Manually define the position of the loop in helioprojective plane using transformation from HGS to HPJ frames
    flare_los_prj = hgs_coord.transform_to(hpj_frame)
    flare_aia_point = frames.Helioprojective(Tx=flare_los_prj.Tx,
                                            Ty=flare_los_prj.Ty,
                                            distance=r_1,
                                            obstime=time,
                                            observer=observer)
    flare_aia_hgs = SkyCoord(lon=flare_aia_point.transform_to(hgs_frame).lon,
                            lat=flare_aia_point.transform_to(hgs_frame).lat,
                            radius=observer.radius,
                            frame=hgs_frame)

    # Get cartesian representation of loop coordinate in heliographic stonyhurst frame
    hgs_coord_xyz = SkyCoord(flare_center_los_hgs, representation_type='cartesian')

    aia_coord_flare = flare_aia_hgs 

    # Convert helioprojective 0,0 coordinate to SkyCoord accounting for observer's position
    aia_coord_flare_xyz = SkyCoord(aia_coord_flare, representation_type='cartesian')

    # Combine the coordinate components into lists that define the rays to be plotted

    x = [aia_coord_flare_xyz.x, hgs_coord_xyz.x]
    y = [aia_coord_flare_xyz.y, hgs_coord_xyz.y]
    z = [aia_coord_flare_xyz.z, hgs_coord_xyz.z]

    # Find the equation of the line to extend the ray past the loop
    # <x, y, z> = <x0, y0, z0> + t * <mx, my, mz>
    # <mx, my, mz> = (<x, y, z> - <x0, y0, z0>) / t     # t can be any number, e.g. 1
    # <mx, my, mz> = <x-x0, y-y0, z-z0> 

    dx = x[1] - x[0]        # Components of the derivative
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    t = kwargs.get('t', 2)                   # Change the length of the ray with any factor   
    px = x[0] + t * dx      # Components of the new point
    py = y[0] + t * dy
    pz = z[0] + t * dz

    x2 = [x[0], px]         # Construct the ray using origin (AIA) and new point
    y2 = [y[0], py]
    z2 = [z[0], pz]

    # Pass the los ray values into SkyCoord object
    los = SkyCoord(
        x=[x2[0].value, x2[1].value] * u.m,
        y=[y2[0].value, y2[1].value] * u.m,
        z=[z2[0].value, z2[1].value] * u.m,
        representation_type='cartesian',
        frame=hgs_frame
    ).transform_to(hpj_frame)

    ax.plot_coord(los, linestyle='--', color='k', lw=1)

    return los

def color_slice(ax, m, los1, los2, **kwargs):
    los1_pix = m.wcs.world_to_pixel(los1)
    los2_pix = m.wcs.world_to_pixel(los2)

    m1 = (los1_pix[1][1] - los1_pix[1][0]) / (los1_pix[0][1] - los1_pix[0][0]) 
    m2 = (los2_pix[1][1] - los2_pix[1][0]) / (los2_pix[0][1] - los2_pix[0][0]) 

    x = [0, m.dimensions[0].value]
    y1 = [0, 0]
    y2 = [0, 0]

    y1[0] = m1 * (x[0]-los1_pix[0][0]) + los1_pix[1][0]
    y1[1] = m1 * (x[1]-los1_pix[0][0]) + los1_pix[1][0]

    y2[0] = m2 * (x[0]-los2_pix[0][0]) + los2_pix[1][0]
    y2[1] = m2 * (x[1]-los2_pix[0][0]) + los2_pix[1][0]

    ax.fill_between(x, y1, y2, **kwargs)

    return ax

def plot_ribbon(ax, pair_matches, **kwargs):

    # Get 304 image
    pr = select_pair_by_wavelength(pair_matches, 304)
    pr_maps = [sunpy.map.Map(item) for item in pr]

    # Load AIA and SDO maps
    a_map = None
    s_map = None
    for ma in pr_maps:
        if ma.meta['TELESCOP'] == 'SDO/AIA':
            a_map = ma
        elif ma.meta['TELESCOP'] == 'STEREO':
            s_map = ma
        else:
            print('Pair contains maps from non AIA / STEREO instruments')

    # select between aia or stereo contour
    instr = kwargs.get('instr', 'STEREO')
    if instr == 'STEREO':
        s_crop_lims = kwargs.get('cl', None)
        s_map_roi = crop_map(s_map, **s_crop_lims)
        s_map_roi = s_map_roi.rotate()

        ma = s_map_roi
    else:
        a_crop_lims = kwargs.get('cl', None)
        a_map_roi = crop_map(a_map, **a_crop_lims)
        a_map = a_map_roi

        ma = a_map

    # Define brightness limits of map
    map_max = ma.data.max()
    map_min = ma.data.min()
    map_range = map_max - map_min

    # Calculate contour level
    unit = ma.unit
    percentile = kwargs.get('p', 95) / 100
    ct = map_range * percentile * unit


    # Find and plot contours
    contours = ma.contour(ct)
    for contour in contours:
        con_pix = ma.wcs.world_to_pixel(contour)
        x = con_pix[0]
        y = con_pix[1]

        ax.plot(x, y, color = 'c')

    return contours

def plot_bars(ax, m, cl, los, **kwargs):
  # Get footpoint coordinates in pix
  fpt1 = cl.loop_coords[0]
  fpt2 = cl.loop_coords[-1]
  fpt1 = m.wcs.world_to_pixel(fpt1)
  fpt2 = m.wcs.world_to_pixel(fpt2)

  # Find difference between fpt pix coords
  pix_diff = (fpt2[0]-fpt1[0], fpt2[1]-fpt1[1])
  pix_diff_fmpt = [p/2 for p in pix_diff]
  pix_diff_fmpt = tuple(pix_diff_fmpt)

  # Plot footpoint coordinates
  ax.plot(fpt1[0], fpt1[1], color = 'c', marker='o', ms=5)
  ax.plot(fpt2[0], fpt2[1], color = 'm', marker='o', ms=5)
  ax.plot([fpt1[0], fpt2[0]], [fpt1[1], fpt2[1]], color = 'k')

  # Get perpendicular coordinates in pix
  l = 50
  xrange = [(fpt1[0] - l), (fpt2[0] + l)]
  per1, per2 = get_perp(fpt1, fpt2, xrange)

  # Plot perpendicular coordinates
  ax.plot(per1[0], per1[1], color = 'c', marker='o', ms=5)
  ax.plot(per2[0], per2[1], color = 'm', marker='o', ms=5)
  ax.plot([per1[0], per2[0]], [per1[1], per2[1]], color = 'k')

  # At other leg
  ax.plot(per1[0] + pix_diff[0], per1[1] + pix_diff[1], color = 'c', marker='o', ms=5)
  ax.plot(per2[0] + pix_diff[0], per2[1] + pix_diff[1], color = 'm', marker='o', ms=5)
  ax.plot([per1[0] + pix_diff[0], per2[0] + pix_diff[0]], 
           [per1[1] + pix_diff[1], per2[1] + pix_diff[1]], color = 'k')

  # At fmpt
  ax.plot(per1[0] + pix_diff_fmpt[0], per1[1] + pix_diff_fmpt[1], color = 'c', marker='o', ms=5)
  ax.plot(per2[0] + pix_diff_fmpt[0], per2[1] + pix_diff_fmpt[1], color = 'm', marker='o', ms=5)
  ax.plot([per1[0] + pix_diff_fmpt[0], per2[0] + pix_diff_fmpt[0]], 
           [per1[1] + pix_diff_fmpt[1], per2[1] + pix_diff_fmpt[1]], color = 'k')
  
  # Get los coordinates in pix
  los_pix = m.wcs.world_to_pixel(los)
  lpt1 = (los_pix[0][0], los_pix[0][1])
  lpt2 = (los_pix[1][0], los_pix[1][1])

  lpix_diff= (lpt2[0]-lpt1[0], lpt2[1]-lpt1[1])
  los_m = lpix_diff[1] / lpix_diff[0]
  los_x1 = per1[0]
  los_y1 = los_m*los_x1 + per1[1]
  los_x2 = per2[0]
  los_y2 = los_m*los_x2 + per1[1]

  los1 = (los_x1, los_y1)
  los2 = (los_x2, los_y2)

  # Plot portion of los
  ax.plot(los1[0], los1[1], color = 'c', marker='o', ms=5)
  ax.plot(los2[0], los2[1], color = 'm', marker='o', ms=5)
  ax.plot([los1[0], los2[0]], [los1[1], los2[1]], color = 'k')

  # Angle between los and per
  theta = get_angle(per1, per2, lpt1, lpt2)
  theta_deg = theta * (180 / np.pi)
  print(theta_deg)

def example_map_annotation(ax2):
    # Annotating the maps
    elements = []
    fs = 14

    elements.append(ax2.text(120,250,'AIA limb',fontsize=fs, color='w'))

    rib_x = [0, 250]
    rib_y = [190, 160]
    elements.append(ax2.plot(rib_x, rib_y, color='c', linestyle='--', lw=2))

    elements.append(ax2.text(40 , 190,
                            f'304 Å {p}% contours',
                            fontsize=fs, 
                            color='c', 
                            #  weight='bold',                         
                            ))

    elements.append(ax2.text(90 , 90,
                            f'Loop of interest',
                            fontsize=fs, 
                            color='r', 
                            #  weight='bold',                         
                            ))

    elements.append(ax2.text(40 , 140,
                            f'AIA los',
                            fontsize=fs, 
                            color='k', 
                            #  weight='bold',                         
                            ))
    
def get_perp(pt1: list, pt2: list, xrange: list):
    """
    Use point-slope form to create a perpendicular line from pixel coordinates

    :param: pt1 - First point in pixel coords
    :param: pt2 - Second point in pixel coords
    :param: xrange - range of x pixels for which the line should be drawn
    :return: Nested list containing coordinates of 2 points forming perpendicular [[x1, y1], [x2, y2]]
    """

    # First Coordinate
    x1 = pt1[0]
    y1 = pt1[1]
    # Second coordinate
    x2 = pt2[0]
    y2 = pt2[1]

    # Calculate Slope
    m = (y2 - y1) / (x2 - x1)

    # Two x values, find corresponding y-values
    x = xrange
    y = [0, 0]

    # Point-slope form (y = mx + b)
    y[0] = -(1/m) * (x[0]-x1) + y1
    y[1] = -(1/m) * (x[1]-x1) + y1

    # Coordinates of 2 pixel points forming perpendicular line
    return (x[0], y[0]) , (x[1], y[1])

def get_angle(pta1, pta2, ptb1, ptb2):

    # Slopes of the two lines, a and b
    ma = (pta2[1] - pta1[1]) / (pta2[0] - pta1[0])
    mb = (ptb2[1] - ptb1[1]) / (ptb2[0] - ptb1[0])

    # Angle between the two co-planar lines
    theta = np.arctan( np.abs( (ma-mb) / (1 + ma*mb) ))

    return theta

class Box:
    """
    Represents a 3D box in solar or observer coordinates defined by its origin, center, dimensions, and resolution.

    This class calculates and stores the coordinates of the box's edges, differentiating between bottom edges and other edges.
    It is designed to integrate with solar physics data analysis frameworks such as SunPy and Astropy.

    :param frame_obs: The observer's frame of reference as a `SkyCoord` object.
    :param box_origin: The origin point of the box in the specified coordinate frame as a `SkyCoord`.
    :param box_center: The geometric center of the box as a `SkyCoord`.
    :param box_dims: The dimensions of the box specified as an `astropy.units.Quantity` array-like in the order (x, y, z).
    :param box_res: The resolution of the box, given as an `astropy.units.Quantity` typically in units of megameters.

    Attributes
    ----------
    corners : list of tuple
        List containing tuples representing the corner points of the box in the specified units.
    edges : list of tuple
        List containing tuples that represent the edges of the box by connecting the corners.
    bottom_edges : list of `SkyCoord`
        A list containing the bottom edges of the box calculated based on the minimum z-coordinate value.
    non_bottom_edges : list of `SkyCoord`
        A list containing all edges of the box that are not classified as bottom edges.

    Methods
    -------
    bl_tr_coords(pad_frac=0.0)
        Calculates and returns the bottom left and top right coordinates of the box in the observer frame.
        Optionally applies a padding factor to expand the box dimensions symmetrically.

    Example
    -------
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> time = Time('2024-05-09T17:12:00')
    >>> box_origin = SkyCoord(450 * u.arcsec, -256 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
    >>> box_center = SkyCoord(500 * u.arcsec, -200 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
    >>> box_dims = u.Quantity([100, 100, 50], u.Mm)
    >>> box_res = 1.4 * u.Mm
    >>> box = Box(frame_obs=box_origin.frame, box_origin=box_origin, box_center=box_center, box_dims=box_dims, box_res=box_res)
    >>> print(box.bounds_coords_bl_tr())
    """

    def __init__(self, frame_obs, box_origin, box_center, box_dims, box_res):
        '''
        Initializes the Box instance with origin, dimensions, and computes the corners and edges.

        :param box_center: SkyCoord, the origin point of the box in a given coordinate frame.
        :param box_dims: u.Quantity, the dimensions of the box (x, y, z) in specified units. x and y are in the solar frame, z is the height above the solar surface.
        '''
        self._frame_obs = frame_obs
        with Helioprojective.assume_spherical_screen(frame_obs.observer):
            self._origin = box_origin
            self._center = box_center
        self._dims = box_dims
        self._res = box_res
        self._dims_pix = np.int_(np.round(self._dims / self._res.to(self._dims.unit)))
        # Generate corner points based on the dimensions
        self.corners = list(itertools.product(self._dims[0] / 2 * [-1, 1],
                                                self._dims[1] / 2 * [-1, 1],
                                                self._dims[2] / 2 * [-1, 1]))

        # Identify edges as pairs of corners differing by exactly one dimension
        self.edges = [edge for edge in itertools.combinations(self.corners, 2)
                        if np.count_nonzero(u.Quantity(edge[0]) - u.Quantity(edge[1])) == 1]
        # Initialize properties to store categorized edges
        self._bottom_edges = None
        self._non_bottom_edges = None
        self._calculate_edge_types()  # Categorize edges upon initialization
        self.b3dtype = ['lfff', 'nlfff']
        self.b3d = {b3dtype: None for b3dtype in self.b3dtype}

    @property
    def dims_pix(self):
        return self._dims_pix

    @property
    def grid_coords(self):
        return self._get_grid_coords(self._center)

    def _get_grid_coords(self, grid_center):
        grid_coords = {}
        grid_coords['x'] = np.linspace(grid_center.x.to(self._dims.unit) - self._dims[0] / 2,
                                        grid_center.x.to(self._dims.unit) + self._dims[0] / 2, self._dims_pix[0])
        grid_coords['y'] = np.linspace(grid_center.y.to(self._dims.unit) - self._dims[1] / 2,
                                        grid_center.y.to(self._dims.unit) + self._dims[1] / 2, self._dims_pix[1])
        grid_coords['z'] = np.linspace(grid_center.z.to(self._dims.unit) - self._dims[2] / 2,
                                        grid_center.z.to(self._dims.unit) + self._dims[2] / 2, self._dims_pix[2])
        grid_coords['frame'] = self._frame_obs
        return grid_coords

    def _get_edge_coords(self, edges, box_center):
        """
        Translates edge corner points to their corresponding SkyCoord based on the box's origin.

        :param edges: List of tuples, each tuple contains two corner points defining an edge.
        :type edges: list of tuple
        :param box_center: The origin point of the box in the specified coordinate frame as a `SkyCoord`.
        :type box_center: `~astropy.coordinates.SkyCoord`
        :return: List of `SkyCoord` coordinates of edges in the box's frame.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return [SkyCoord(x=box_center.x + u.Quantity([edge[0][0], edge[1][0]]),
                         y=box_center.y + u.Quantity([edge[0][1], edge[1][1]]),
                         z=box_center.z + u.Quantity([edge[0][2], edge[1][2]]),
                         frame=box_center.frame) for edge in edges]

    # def _get_bottom_bl_tr_coords(self,box_center):
    #    return [SkyCoord(x=box_center.x - self._box_dims[0] / 2,
    def _get_bottom_cea_header(self):
        """
        Generates a CEA header for the bottom of the box.

        :return: The FITS WCS header for the bottom of the box.
        :rtype: dict
        """
        origin = self._origin.transform_to(HeliographicStonyhurst)
        shape = self._dims[:-1][::-1] / self._res.to(self._dims.unit)
        shape = list(shape.value)
        shape = [int(np.ceil(s)) for s in shape]
        rsun = origin.rsun.to(self._res.unit)
        scale = np.arcsin(self._res / rsun).to(u.deg) / u.pix
        scale = u.Quantity((scale, scale))
        # bottom_cea_header = make_fitswcs_header(shape, origin,
        #                                          scale=scale, observatory=self._origin.observer, projection_code='CEA')
        bottom_cea_header = make_fitswcs_header(shape, origin,
                                                 scale=scale, projection_code='CEA')
        bottom_cea_header['OBSRVTRY'] = str(origin.observer)
        return bottom_cea_header

    def _calculate_edge_types(self):
        """
        Separates the box's edges into bottom edges and non-bottom edges. This is done in a single pass to improve efficiency.
        """
        min_z = min(corner[2] for corner in self.corners)
        bottom_edges, non_bottom_edges = [], []
        for edge in self.edges:
            if edge[0][2] == min_z and edge[1][2] == min_z:
                bottom_edges.append(edge)
            else:
                non_bottom_edges.append(edge)
        self._bottom_edges = self._get_edge_coords(bottom_edges, self._center)
        self._non_bottom_edges = self._get_edge_coords(non_bottom_edges, self._center)

    def _get_bounds_coords(self, edges, bltr=False, pad_frac=0.0):
        """
        Provides the bounding box of the edges in solar x and y.

        :param edges: List of tuples, each tuple contains two corner points defining an edge.
        :type edges: list of tuple
        :param bltr: If True, returns bottom left and top right coordinates, otherwise returns minimum and maximum coordinates.
        :type bltr: bool, optional
        :param pad_frac: Fractional padding applied to each side of the box, expressed as a decimal, defaults to 0.0.
        :type pad_frac: float, optional

        :return: Coordinates of the box's bounds.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        xx = []
        yy = []
        for edge in edges:
            xx.append(edge.transform_to(self._frame_obs).Tx)
            yy.append(edge.transform_to(self._frame_obs).Ty)
        unit = xx[0][0].unit
        min_x = np.min(xx)
        max_x = np.max(xx)
        min_y = np.min(yy)
        max_y = np.max(yy)
        if pad_frac > 0:
            _pad = pad_frac * np.max([max_x - min_x, max_y - min_y, 20])
            min_x -= _pad
            max_x += _pad
            min_y -= _pad
            max_y += _pad
        if bltr:
            bottom_left = SkyCoord(min_x * unit, min_y * unit, frame=self._frame_obs)
            top_right = SkyCoord(max_x * unit, max_y * unit, frame=self._frame_obs)
            return [bottom_left, top_right]
        else:
            coords = SkyCoord(Tx=[min_x, max_x] * unit, Ty=[min_y, max_y] * unit,
                                frame=self._frame_obs)
            return coords

    def bounds_coords_bl_tr(self, pad_frac=0.0):
        """
        Calculates and returns the bottom left and top right coordinates of the box in the observer frame.
        Optionally applies a padding factor to expand the box dimensions symmetrically.

        :param pad_frac: Fractional padding applied to each side of the box, expressed as a decimal, defaults to 0.0.
        :type pad_frac: float, optional
        :return: Bottom left and top right coordinates of the box in the observer frame.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._get_bounds_coords(self.all_edges, bltr=True, pad_frac=pad_frac)

    @property
    def bounds_coords(self):
        """
        Provides access to the box's bounds in the observer frame.

        :return: Coordinates of the box's bounds.
        :rtype: `~astropy.coordinates.SkyCoord`
        """
        return self._get_bounds_coords(self.all_edges)

    @property
    def bottom_bounds_coords(self):
        """
        Provides access to the box's bottom bounds in the observer frame.

        :return: Coordinates of the box's bottom bounds.
        :rtype: `~astropy.coordinates.SkyCoord`
        """
        return self._get_bounds_coords(self.bottom_edges)

    @property
    def bottom_cea_header(self):
        """
        Provides access to the box's bottom WCS CEA header.

        :return: The WCS CEA header for the box's bottom.
        :rtype: dict
        """
        return self._get_bottom_cea_header()

    @property
    def bottom_edges(self):
        """
        Provides access to the box's bottom edge coordinates.

        :return: Coordinates of the box's bottom edges.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._bottom_edges

    @property
    def non_bottom_edges(self):
        """
        Provides access to the box's non-bottom edge coordinates.

        :return: Coordinates of the box's non-bottom edges.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._non_bottom_edges

    @property
    def all_edges(self):
        """
        Provides access to all the edge coordinates of the box, combining both bottom and non-bottom edges.

        :return: Coordinates of all the edges of the box.
        :rtype: list of `~astropy.coordinates.SkyCoord`
        """
        return self._bottom_edges + self._non_bottom_edges

    @property
    def box_origin(self):
        """
        Provides read-only access to the box's origin coordinates.

        :return: The origin of the box in the specified frame.
        :rtype: `~astropy.coordinates.SkyCoord`
        """
        return self._center

    @property
    def box_dims(self):
        """
        Provides read-only access to the box's dimensions.

        :return: The dimensions of the box (length, width, height) in specified units.
        :rtype: `~astropy.units.Quantity`
        """
        return self._dims

def plot_edges(ax, m, sfiObj, **kwargs):
    # 1. Define observer and box parameters
    time = m.reference_coordinate.obstime
    observer = get_earth(time)
    dom_width = sfiObj.domain_width * (1-sfiObj.zoom)
    box_dims = u.Quantity([
        dom_width[2],
        dom_width[1],
        dom_width[0],
    ])
    box_res = 1.4 * u.Mm

    frame_obs = Helioprojective(observer=observer, obstime=time)
    box_origin = SkyCoord(lon=93 * u.deg, lat=-14 * u.deg, radius=696 * u.Mm, #TODO get from sfiObj
                    frame='heliographic_stonyhurst',
                    observer=observer, obstime=time)
    frame_hcc = Heliocentric(observer=box_origin, obstime=time)
    box_origin_hcc = box_origin.transform_to(frame_hcc)
    box_center = SkyCoord(x=box_origin_hcc.x,
                            y=box_origin_hcc.y,
                            z=box_origin_hcc.z + box_dims[2] / 2,
                            frame=box_origin_hcc.frame)
    
    # 2. Instantiate the Box object
    box = Box(frame_obs=frame_obs, box_origin=box_origin, box_center=box_center, box_dims=box_dims, box_res=box_res)

    # 5. Overlay Box edges onto the map
    # Transform box edges to the map's coordinate frame
    bottom_edges_transformed = [edge.transform_to(m.coordinate_frame) for edge in box.bottom_edges]
    non_bottom_edges_transformed = [edge.transform_to(m.coordinate_frame) for edge in box.non_bottom_edges]

    # Plot bottom edges in blue and non-bottom edges in red
    alpha = 0.6
    for c in bottom_edges_transformed:
        ax.plot_coord(c, color='white', linewidth=2, alpha=alpha)
    for c in non_bottom_edges_transformed:
        ax.plot_coord(c, color='white', linewidth=2, alpha=alpha - 0.2)
    
    axs = kwargs.get('axes', False)
    if axs:
      # Plot MHD axes in box
      # Find coordinates for center of 2 bottom edges
      box_edge_x = SkyCoord(x=box_origin_hcc.x + box_dims[0] / 2,
                              y=box_origin_hcc.y,
                              z=box_origin_hcc.z,
                              frame=box_origin_hcc.frame)
      box_edge_y = SkyCoord(x=box_origin_hcc.x,
                              y=box_origin_hcc.y + box_dims[1] / 2,
                              z=box_origin_hcc.z,
                              frame=box_origin_hcc.frame)
      
      # Transform all coordinates to map coordinate frame
      ori = box_origin.transform_to(m.coordinate_frame)
      cen = box_center.transform_to(m.coordinate_frame)
      bex = box_edge_x.transform_to(m.coordinate_frame)
      bey = box_edge_y.transform_to(m.coordinate_frame)

      # Plot all coordinates
      ax.plot_coord(ori, marker='o', markersize=8, color='magenta')
      ax.plot_coord(cen, marker='o', markersize=8, color='green')
      ax.plot_coord(bex, marker='o', markersize=8, color='red')
      ax.plot_coord(bey, marker='o', markersize=8, color='blue')

      # Convert coordinates to pixel pair
      oripix = m.wcs.world_to_pixel(ori)
      cenpix = m.wcs.world_to_pixel(cen)
      bexpix = m.wcs.world_to_pixel(bex)
      beypix = m.wcs.world_to_pixel(bey)

      # Split pixel pairs into appropriate (x1, x2), (y1, y2) pairs for plotting
      cen_xpix = [oripix[0], cenpix[0]]
      cen_ypix = [oripix[1], cenpix[1]]
      bex_xpix = [oripix[0], bexpix[0]]
      bex_ypix = [oripix[1], bexpix[1]]
      bey_xpix = [oripix[0], beypix[0]]
      bey_ypix = [oripix[1], beypix[1]]

      # Plot new pixel pairs
      ax.plot(cen_xpix, cen_ypix, color='green')
      ax.plot(bex_xpix, bex_ypix, color='red')
      ax.plot(bey_xpix, bey_ypix, color='blue')
      
      # Annotate MHD axis labels where axes end
      fs = kwargs.get('fontsize', 16)
      xoff = kwargs.get('xoffset', 10)
      ax.text(cenpix[0] + xoff, cenpix[1], 'y', color='black', fontsize=fs)
      ax.text(bexpix[0] + xoff, bexpix[1], 'z', color='black', fontsize=fs)
      ax.text(beypix[0] + xoff, beypix[1], 'x', color='black', fontsize=fs)

