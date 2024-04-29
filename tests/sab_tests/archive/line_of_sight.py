import astropy.constants as const
import astropy.units as u
from sunpy.coordinates import Heliocentric
from astropy.coordinates import SkyCoord
import numpy as np

# The purpose of this script is to translate a z+ los vector [0, 0, 1] into
# a los vector defined in the loop coordinate system

def circle_3d(x0, y0, z0, r, theta, phi, t):
    """
    Compute the parametric equations for a circle in 3D.

    Parameters:
    - x0, y0, z0: Coordinates of the center of the circle.
    - r: Radius of the circle.
    - theta: Elevation angle of the circle's normal (in radians).
    - phi: Azimuth angle of the circle's normal (in radians).
    - t: Array of parameter values, typically ranging from 0 to 2*pi.

    Returns:
    - x, y, z: Arrays representing the x, y, and z coordinates of the circle in 3D.
    """

    # Normal vector
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # Arbitrary vector (z-axis)
    z = np.array([0, 0, 1])

    # Orthogonal vector in the plane of the circle
    u = np.cross(n, z)
    u /= np.linalg.norm(u)  # Normalize

    # Another orthogonal vector in the plane
    v = np.cross(n, u)
    v /= np.linalg.norm(v)  # Normalize

    # Parametric equations
    x = x0 + r * np.cos(t) * u[0] + r * np.sin(t) * v[0]
    y = y0 + r * np.cos(t) * u[1] + r * np.sin(t) * v[1]
    z = z0 + r * np.cos(t) * u[2] + r * np.sin(t) * v[2]

    return x, y, z


def semi_circle_loop(radius, height, theta0=0 * u.deg, phi0=0 * u.deg, el=90 * u.deg, az=0 * u.deg, samples_num=100):
    '''
    Compute a semicircular loop with both footpoints rooted on the surface of the Sun.

    Parameters:
    - radius: Radius of the semi-circular loop in units compatible with astropy.units (e.g., u.Mm).
    - height: Height of the center of the circle relative to the photosphere in units compatible with astropy.units (e.g., u.Mm).
    - theta0: Heliographic Stonyhurst latitude (theta) of the center of the circle. Default is 0 degrees.
    - phi0: Heliographic Stonyhurst longitude (phi) of the center of the circle. Default is 0 degrees.
    - el: Elevation angle of the circle's normal. It ranges from 0 to 180 degrees. Default is 90 degrees.
    - az: Azimuth angle of the circle's normal. It ranges from 0 to 360 degrees. Default is 0 degrees.
    - samples_num: Number of samples for the parameter t. Default is 1000.


    Returns:
    - SkyCoord object: Represents the coordinates of the semi-circular loop in the heliographic Stonyhurst coordinate system.
    '''

    # Radius of the sun
    r_1 = const.R_sun

    # Variables for offsetting center of the circle
    r0 = r_1 + height           # Distance of the center of the circle from the center of the sun
    x0 = u.Quantity(0 * u.cm)   # Zero (cm)
    y0 = u.Quantity(0 * u.cm)   # Zero (cm)
    z0 = r0.to(u.cm)            # Distance of the center of the circle from the center of the sun (cm)

    theta = el.to(u.rad).value  # np.pi / 2  # Elevation angle (angle relative to tangent plane)
    phi = az.to(u.rad).value  # np.pi / 4  # Azimuth angle (rotation about normal vector)
    t = np.linspace(0, 2 * np.pi, int(samples_num))  # Parameter t (spans 0 to 2pi in sample_num steps)

    # Retrieves parametric equations for circle in x, y, z (array elements calculated for every point t)
    dx, dy, dz = circle_3d(0, 0, 0, radius, theta, phi, t)

    # Offsets parameterized circle coordinates by x0, y0, z0
    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)       # Distances of points on the circle from center of the sun
    rdiff = r - r_1                             # Distance of point above surface of the sun
    rsort = np.argmin(np.abs(rdiff))            # Index of minimum distance
    if rdiff[rsort] + rdiff[rsort + 1] < 0:
        rsort += 1

    # Orders lists so that the first point in parametric x, y z is on the solar surface
    r = np.roll(r, -rsort)
    x = np.roll(x, -rsort)
    y = np.roll(y, -rsort)
    z = np.roll(z, -rsort)
    dx = np.roll(dx, -rsort)
    dy = np.roll(dy, -rsort)
    dz = np.roll(dz, -rsort)

    i_r = np.where(r > r_1)     # Returns indices of points above the solar surface
    # r = r[i_r]
    # phi = phi[i_r]

    # Overwrites all coordinates with versions without points below the solar surface
    x = x[i_r]
    y = y[i_r]
    z = z[i_r]
    dx = dx[i_r]
    dy = dy[i_r]
    dz = dz[i_r]

    # Calculate the length of the loop based on the angle between the start and end points.
    # Define the vectors v1 and v2
    v1 = np.array([dx[0].value, dy[0].value, dz[0].value]) * dx[0].unit
    v2 = np.array([dx[-1].value, dy[-1].value, dz[-1].value]) * dx[0].unit
    # Calculate the angle between the vectors (alpha) using the dot product
    cos_alpha = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    alpha = np.arccos(cos_alpha)

    # Use the cross product to determine the orientation
    cross_product = np.cross(v1, v2)
    if cross_product[2] < 0:  # Assuming z is the up direction
        alpha = 2 * np.pi * alpha.unit - alpha

    # Calculate the arc length
    loop_length = alpha.value * radius
    print('Loop length:', loop_length)

    hcc_frame = Heliocentric(observer=SkyCoord(lon=phi0, lat=theta0, radius=r_1, frame='heliographic_stonyhurst'))
    return (SkyCoord(x=x, y=y, z=z, frame=hcc_frame).transform_to('heliographic_stonyhurst')), loop_length

loop, length = semi_circle_loop(10 * u.Mm, 10 * u.Mm)

# Arbitrary vector (z-axis)
z = np.array([0, 0, 1])
