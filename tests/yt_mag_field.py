import numpy as np
import sys
import os.path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

# Import synthetic image manipulation tools
import yt
from yt.visualization.api import Streamlines
from unyt import unyt_quantity
from unyt import Mm
from unyt import UnitRegistry, Unit
from unyt.dimensions import length

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
# from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
# from rad_transfer.emission_models import uv, xrt

export_pm = 'tex'

if export_pm == 'tex':
    import matplotlib
    params = {
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        #'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 12, # fontsize for x and y labels (was 10)
        'axes.titlesize': 12,
        'font.size': 10, # was 10
        #'legend.fontsize': 6, # was 10
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'figure.figsize': [6.39, 4.10],
        'font.family': 'sans',
    }
    matplotlib.rcParams.update(params)

reg = UnitRegistry()
reg.add("code_length", base_value=1.4e8, dimensions=length, tex_repr=r"\rm{Code Length}")

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    # "density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")  # , units_override=units_override)
rad_buffer_obj.length_unit = 140 * Mm

cut_box = rad_buffer_obj.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
'''
Introduce streamlines to trace vector fields
Following the tutorial from https://yt-project.org/doc/visualizing/streamlines.html 
'''

c = rad_buffer_obj.domain_center
N = 300

scale_x = rad_buffer_obj.domain_width[0].to('Mm')
scale_y = rad_buffer_obj.domain_width[1].to('Mm')
scale_z = rad_buffer_obj.domain_width[2].to('Mm')  # Possibly lowest dimension of the dataset (?)
scale = scale_z

pos_d = np.random.random((N, 3)) * scale - scale / 2.0
pos = c + pos_d  # initialize pos as a unyt.array.unyt_array

pos_dx = np.random.random((N, 1)) * scale_x - scale_x / 2.0
pos_dy = np.random.random((N, 1)) * scale_y - scale_y / 2.0
pos_dz = np.random.random((N, 1)) * scale_z - scale_z / 2.0

pos[:, 0] = c[0] + np.ravel(pos_dx)
pos[:, 1] = c[1] + np.ravel(pos_dy)
pos[:, 2] = c[2] + np.ravel(pos_dz)
pos=pos.to('Mm')
'''
Create streamlines of the 3D magnetic field and integrate them through the box volume
'''
streamlines = Streamlines(
    rad_buffer_obj,
    pos,
    ('gas', 'magnetic_field_x'),
    ('gas', 'magnetic_field_y'),
    ('gas', 'magnetic_field_z'),
    length=150. * Mm,
    get_magnitude=True,
)
streamlines.integrate_through_volume()
#%%
'''
Creating a matplotlib 3D plot to trace the streamlines through the 3D volume of the plot
'''
# Create a 3D plot, trace the streamlines through the 3D volume of the plot
xsize, ysize = 4.5, 4.5
fig = plt.figure(figsize=(xsize, ysize), dpi=140)
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

for stream in streamlines.streamlines.to('Mm'):
    stream = stream[np.all(stream != 0.0, axis=1)]
    ax.plot3D(stream[:, 0], stream[:, 1], stream[:, 2], alpha=0.25, linewidth=0.6, color='indigo')
    ax.view_init(100, -20, 90)
    plt.grid(visible=None)

# ax.text2D(0.75, 0.05, "Newly reconnected \n highly bent field lines", transform=ax.transAxes)

# ax.annotate3D('Newly reconnected \n highly bent field lines', (0, 100, 0),
#               xytext=(0.75, 0.05),
#               textcoords='offset points',
#               #bbox=dict(boxstyle="round", fc="lightyellow"),
#               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=1))

ax.set_xlabel('$x$, Mm')
ax.set_ylabel('$y$, Mm')
ax.set_zticks([-30, 30])
ax.set_zlabel('$z$, Mm')
ax.set_aspect('equal')
# Axes3D.text(x, y, z, s, zdir=None, **kwargs)

# Save the plot to disk.
plt.savefig("mag_field_lines.pgf")




