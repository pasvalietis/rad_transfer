import sys

sys.path.insert(1, '/home/saber/CoronalLoopBuilder')
clb_path = '/home/saber/CoronalLoopBuilder/examples/testing/'

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, spherical_to_cartesian as stc
import sunpy.map
import astropy.constants as const

# Import synthetic image manipulation tools
import yt
from utils.proj_imag import SyntheticFilterImage as synt_img
import pickle

# noinspection PyUnresolvedReferences
from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d

# Method to create synthetic map of MHD data from rad_transfer
def synthmap_plot(img, fig, normvector=None, northvector=None, comp=False, fm_coord=None,
                  **kwargs):

    # Default initialization of normvector and northvector
    if normvector is None:
        normvector = [1, 0, 0]
    if northvector is None:
        northvector = [0, 0, 1]

    # Load and crop subsampled 3D MHD file
    downs_file_path = '/home/saber/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
    subs_ds = yt.load(downs_file_path)
    cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

    # Instrument settings for synthetic image
    instr = 'aia'  # keywords: 'aia' or 'xrt'
    channel = 131

    # Prepare cropped MHD data for imaging
    aia_synthetic = synt_img(cut_box, instr, channel)

    # Match parameters of the synthetic image to observed one
    samp_resolution = img.data.shape[0]
    obs_scale = [img.scale.axis1, img.scale.axis2] * (u.arcsec / u.pixel)

    reference_coord = img.reference_coordinate

    # Calculate reference pixel for synth img using loop midpoint
    if fm_coord is None:
        reference_pixel = u.Quantity([img.reference_pixel[0].value,
                                      img.reference_pixel[1].value], u.pixel)
    else:
        fm_coord = fm_coord.transform_to(img.coordinate_frame)
        fm_pix = fm_coord.to_pixel(img.wcs)
        reference_pixel = u.Quantity([img.reference_pixel[0].value - fm_pix[0],
                                      img.reference_pixel[1].value + fm_pix[1]*0.5], u.pixel)

    # Dynamic synth plot settings
    synth_plot_settings = {'resolution': samp_resolution,
                           'vmin': 1e-15,
                           'vmax': 1e6,
                           'cmap': 'inferno',
                           'logscale': True}
    synth_view_settings = {'normal_vector': normvector,  # Line of sight - changes 'orientation' of projection
                           'north_vector': northvector}  # rotates projection in xy

    aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[0, 0],  # move the bottom center of the flare in [x,y]
                                bkg_fill=np.min(img.data))

    # Import scale from an AIA image:
    synth_map = aia_synthetic.make_synthetic_map(
                                                 obstime='2013-10-28',
                                                 # obstime=reference_coord.obstime,
                                                 observer='earth',
                                                 detector='Synthetic AIA',
                                                 scale=obs_scale,
                                                 reference_coord=reference_coord,
                                                 reference_pixel=reference_pixel
                                                 )

    if comp:
        synth_map = sunpy.map.Map(synth_map, vmin=1e-5, vmax=8e1, cmap='inferno')
        # comp = sunpy.map.Map(img, synth_map, composite=True)
        comp = sunpy.map.Map(synth_map, img, composite=True)
        comp.set_alpha(1, 0.50)
        ax = fig.add_subplot(projection=comp.get_map(0))
        # ax = fig.add_subplot(projection=img)
        comp.plot(axes=ax)
    else:
        ax = fig.add_subplot(projection=synth_map)
        synth_map.plot(axes=ax, vmin=1e-5, vmax=8e1, cmap='inferno')

    return ax


def calc_vect(radius=const.R_sun, height=10 * u.Mm, theta0=0 * u.deg, phi0=0 * u.deg, el=90 * u.deg, az=0 * u.deg,
              samples_num=100, **kwargs):
    if 'pkl' in kwargs:
        with open(kwargs.get('pkl'), 'rb') as f:
            dims = pickle.load(f)
            print(f'Loop dimensions loaded:{dims}')
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

    r_1 = const.R_sun

    r0 = r_1 + height
    x0 = u.Quantity(0 * u.cm)
    y0 = u.Quantity(0 * u.cm)
    z0 = r0.to(u.cm)

    theta = el.to(u.rad).value  # np.pi / 2  # Elevation angle
    phi = az.to(u.rad).value  # np.pi / 4  # Azimuth angle
    t = np.linspace(0, 2 * np.pi, int(samples_num))  # Parameter t

    dx, dy, dz = circle_3d(0, 0, 0, radius, theta, phi, t)

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rdiff = r - r_1
    rsort = np.argmin(np.abs(rdiff))
    if rdiff[rsort] + rdiff[rsort + 1] < 0:
        rsort += 1
    r = np.roll(r, -rsort)
    x = np.roll(x, -rsort)
    y = np.roll(y, -rsort)
    z = np.roll(z, -rsort)
    dx = np.roll(dx, -rsort)
    dy = np.roll(dy, -rsort)
    dz = np.roll(dz, -rsort)

    i_r = np.where(r > r_1)

    x = x[i_r]
    y = y[i_r]
    z = z[i_r]
    dx = dx[i_r]
    dy = dy[i_r]
    dz = dz[i_r]

    # Calculate the length of the loop based on the angle between the start and end points.
    # Define the vectors v1 and v2
    v1 = np.array([x[0].value, y[0].value, z[0].value]) * x[0].unit
    v2 = np.array([x[-1].value, y[-1].value, z[-1].value]) * x[0].unit

    # Use the cross product to determine the orientation
    cross_product = np.cross(v1, v2)

    # Normal Vector
    norm0 = cross_product / np.linalg.norm(cross_product)
    # Transformation to MHD coordinate frame
    norm = [0, 0, 0]
    norm[0] = norm0[1]
    norm[1] = norm0[2]
    norm[2] = norm0[0]

    # Derive the cartesian coordinates of a normalized vector pointing in the direction
    # of the coronal loop's spherical coordinates (midpoint of footpoints)
    midptn_cart = stc(1, theta0, phi0)

    # North Vector
    north0 = [midptn_cart[0].value, midptn_cart[1].value, midptn_cart[2].value]
    # Transformation to MHD coordinate frame
    north = [0, 0, 0]
    north[0] = north0[1]
    north[1] = north0[2]
    north[2] = north0[0]

    # normal vector, north vector
    return norm, north


# Method to load maps from pickle file
def load_maps(**kwargs):
    """
    Shortcut function to load pre-rendered (and cropped) maps from AIA and Stereo

    :return: Paired list of aia, stereo A maps
    """
    maps = []

    mapdirs171 = [clb_path + 'maps/0_AIA-STEREOA_171_2012.pkl', clb_path + 'maps/1_AIA-STEREOA_171_2012.pkl']
    mapdirs195 = [clb_path + 'maps/0_AIA-STEREOA_195_2012.pkl', clb_path + 'maps/1_AIA-STEREOA_195_2012.pkl']
    mapdirs304 = [clb_path + 'maps/0_AIA-STEREOA_304_2012.pkl', clb_path + 'maps/1_AIA-STEREOA_304_2012.pkl']

    channel = kwargs.get('channel', 195)

    mapdirs = []
    if channel == 171:
        mapdirs = mapdirs171
    if channel == 195:
        mapdirs = mapdirs195
    if channel == 304:
        mapdirs = mapdirs304

    for name in mapdirs:
        with open(name, 'rb') as f:
            maps.append(pickle.load(f))
            f.close()

    # num_maps = len(maps)

    maps = [maps[0]]

    return maps


# Path to fits image of event
# img_path = ('/home/saber/CoronalLoopBuilder/examples/testing/downloaded_events/'
#             'aia_lev1_193a_2012_07_19t06_40_08_90z_image_lev1.fits')

img_path = ('/home/saber/CoronalLoopBuilder/examples/testing/downloaded_events/'
            'aia_lev1_131a_2012_07_19t06_40_11_97z_image_lev1.fits')

# Load aia image map from fits image
img = sunpy.map.Map(img_path)
# Crop image map
bl = SkyCoord(850 * u.arcsec, -330 * u.arcsec, frame=img.coordinate_frame)
res = 250 * u.arcsec
img = img.submap(bottom_left=bl, width=res, height=res)

# Path to clb loop parameters
params_path = 'loop_params/front_loop_131.pkl'
# params_path = clb_path + 'loop_params/STEREOA_2012.pkl'

# Load pre-rendered maps for selected channel
# maps = load_maps(channel=131)

# Calculate normal and north vectors for synthetic image alignment
norm, north = calc_vect(pkl=params_path)

# Retreive the heliographic sky coordinate of the midpoint of the loop
fm = SkyCoord(lon=93.5 * u.deg, lat=-15.0 * u.deg, radius=const.R_sun, frame='heliographic_stonyhurst',
              obstime=img.reference_coordinate.obstime)

fig = plt.figure()

synth_axs = [synthmap_plot(img, fig, normvector=norm, northvector=north,
                           comp=True, fm_coord=fm, path=img_path)]

# sunpy.map.Map(img_path).plot(axes=synth_axs[0], autoalign=True)

coronal_loop1 = CoronalLoopBuilder(fig, synth_axs, [img], pkl=params_path)

plt.show()
plt.close()

# coronal_loop1.save_params_to_pickle("front_loop_131.pkl")
coronal_loop1.save_to_fig("figs/composite_bw.jpg", dpi=300, bbox_inches='tight')
