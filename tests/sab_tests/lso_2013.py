# Copy of loop synth over made to clean up this routine

import sys

sys.path.insert(1, '/home/saber/CoronalLoopBuilder')

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
    """

    @param img: Real image to project synthetic map onto
    @param fig: Matplotlib figure reference for plotting
    @param normvector: Vector indicating vector normal to the surface of the sun
    @param northvector: Vector indicating pointing direction for candleflame 'tip' of flare
    @param comp: 'True' for real + synthetic image, 'False' for just synthetic image
    @param fm_coord: Coordinate for the foot midpoint of the flare loop
    @param kwargs: Optional keywords: 'datacube', 'center', 'left_edge', 'right_edge', 'instr', 'channel'
    @return:
    """

    # Default initialization of normvector and northvector
    if normvector is None:
        normvector = [1, 0, 0]
    if northvector is None:
        northvector = [0, 0, 1]

    # Load subsampled 3D MHD file
    shen_datacube = '/home/saber/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
    downs_file_path = kwargs.get('datacube', shen_datacube)
    subs_ds = yt.load(downs_file_path)

    # Crop MHD file
    center = [0.0, 0.5, 0.0]
    left_edge = [-0.5, 0.016, -0.25]
    right_edge = [0.5, 1.0, 0.25]
    cut_box = subs_ds.region(center=kwargs.get('center', center), left_edge=kwargs.get('left_edge', left_edge),
                             right_edge=kwargs.get('right_edge', right_edge))

    # Instrument settings for synthetic image
    instr = kwargs.get('instr', 'aia')  # keywords: 'aia' or 'xrt'
    channel = kwargs.get('channel', 131)

    # Prepare cropped MHD data for imaging
    aia_synthetic = synt_img(cut_box, instr, channel)

    # Match parameters of the synthetic image to observed one
    samp_resolution = img.data.shape[0]
    obs_scale = [img.scale.axis1, img.scale.axis2] * (u.arcsec / u.pixel)
    # reference_coord = img.reference_coordinate

    # # Calculate reference pixel for synth img using loop midpoint
    # if fm_coord is None:
    #     reference_pixel = u.Quantity([img.reference_pixel[0].value,
    #                                   img.reference_pixel[1].value], u.pixel)
    # else:
    #     fm_coord = fm_coord.transform_to(img.coordinate_frame)
    #     fm_pix = fm_coord.to_pixel(img.wcs)
    #
    #     xdisp = 0
    #     ydisp = 0
    #
    #     # xdisp = fm_pix[0]
    #     # ydisp = -1 * fm_pix[1] * 0.5
    #
    #     reference_pixel = u.Quantity([img.reference_pixel[0].value + xdisp,
    #                                   img.reference_pixel[1].value + ydisp], u.pixel)

    # Dynamic synth plot settings
    synth_plot_settings = {'resolution': samp_resolution,
                           'vmin': 1e-15,
                           'vmax': 8e1,
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
                                                 obstime=img.reference_coordinate.obstime,
                                                 observer='earth',
                                                 detector='Synthetic AIA',
                                                 scale=obs_scale,
                                                 reference_coord=fm_coord,
                                                 # reference_coord=reference_coord,
                                                 # reference_pixel=reference_pixel
                                                 )

    if comp:
        comp = sunpy.map.Map(synth_map, img, composite=True)
        comp.set_alpha(1, 0.50)
        ax = fig.add_subplot(projection=comp.get_map(0))
        comp.plot(axes=ax)
    else:
        ax = fig.add_subplot(projection=synth_map)
        synth_map.plot(axes=ax, vmin=1e-5, vmax=8e1, cmap='inferno')

        coord=synth_map.reference_coordinate
        pixels = synth_map.wcs.world_to_pixel(coord)
        coord_img=img.reference_coordinate
        pixels_img=img.wcs.world_to_pixel(coord_img)
        center_image_pix = [synth_map.data.shape[0] / 2., synth_map.data.shape[1] / 2.] * u.pix
        ax.plot_coord(coord, 'o', color='r')
        ax.plot_coord(coord_img, 'o', color='b')
        ax.plot(pixels[0] * u.pix, pixels[1] * u.pix, 'x', color='w')
        ax.plot(pixels_img[0] * u.pix, pixels_img[1] * u.pix, 'x', color='w')
        # ax.plot(center_image_pix[0], center_image_pix[1], 'x', color='g')

    return ax


def calc_vect(radius=const.R_sun, height=10 * u.Mm, theta0=0 * u.deg, phi0=0 * u.deg, el=90 * u.deg, az=0 * u.deg,
              samples_num=100, **kwargs):

    DEFAULT_RADIUS = 10.0 * u.Mm
    DEFAULT_HEIGHT = 0.0 * u.Mm
    DEFAULT_PHI0 = 0.0 * u.deg
    DEFAULT_THETA0 = 0.0 * u.deg
    DEFAULT_EL = 90.0 * u.deg
    DEFAULT_AZ = 0.0 * u.deg

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


# Path to local Coronal Loop Builder
clb_path = '/home/saber/CoronalLoopBuilder/examples/testing/'

# Path to target sunpy map
map_path = (clb_path + 'maps/2013/AIA-171.pkl')

# Retrieve sunpy map object
with open(map_path, 'rb') as f:
    img = pickle.load(f)
    f.close()

# Path to clb loop parameters
params_path = clb_path + 'loop_params/2013/front_2013.pkl'

# Calculate normal and north vectors for synthetic image alignment
norm, north = calc_vect(pkl=params_path)

# Retrieve loop latitude and longitude
with open(params_path, 'rb') as f:
    params = pickle.load(f)
    f.close()
lat = params['theta0'].value
lon = params['phi0'].value
height = params['height']

# define the heliographic sky coordinate of the midpoint of the loop
# fm = SkyCoord(lon=lon * u.deg, lat=lat * u.deg, radius=const.R_sun, frame='heliographic_stonyhurst',
#               obstime=img.reference_coordinate.obstime)
hheight = 75*u.Mm
fm = SkyCoord(lon=lon * u.deg, lat=lat * u.deg, radius=const.R_sun + hheight, frame='heliographic_stonyhurst',
              observer='earth', obstime=img.reference_coordinate.obstime).transform_to(frame='helioprojective')

# plot real and synthetic views
fig = plt.figure()
synth_axs = [synthmap_plot(img, fig, normvector=norm, northvector=north, comp=False, fm_coord=fm,
                           instr='aia', channel=171)]
# coronal_loop1 = CoronalLoopBuilder(fig, synth_axs, [img], pkl=params_path)

plt.show()
plt.close()
