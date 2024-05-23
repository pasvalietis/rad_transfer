import os
import yt
#import sys
#sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from buffer import downsample

directory = os.getcwd()

downs_factor = 3

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and f.endswith('.vtk') and filename.startswith('flarecs'):
        print(f)
        ds = yt.load(f, units_override=units_override, hint='AthenaDataset')
        rad_buffer_obj = downsample(ds, rad_fields=True, n=downs_factor)

