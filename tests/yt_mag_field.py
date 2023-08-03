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

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
# from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
# from rad_transfer.emission_models import uv, xrt

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/flarecs-id.0035_ss3.h5'
rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")

cut_box = rad_buffer_obj.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

'''
Introduce streamlines to trace vector fields
Following the tutorial from https://yt-project.org/doc/visualizing/streamlines.html 
'''
c = rad_buffer_obj.domain_center
N = 100

scale_x = rad_buffer_obj.domain_width[0]
scale_y = rad_buffer_obj.domain_width[1]
scale_z = rad_buffer_obj.domain_width[2]  # Possibly lowest dimension of the dataset (?)
scale = scale_z

pos_d = np.random.random((N, 3)) * scale - scale / 2.0
pos = c + pos_d  # initialize pos as a unyt.array.unyt_array

pos_dx = np.random.random((N, 1)) * scale_x - scale_x / 2.0
pos_dy = np.random.random((N, 1)) * scale_y - scale_y / 2.0
pos_dz = np.random.random((N, 1)) * scale_z - scale_z / 2.0

pos[:, 0] = c[0] + np.ravel(pos_dx)
pos[:, 1] = c[1] + np.ravel(pos_dy)
pos[:, 2] = c[2] + np.ravel(pos_dz)

'''
Create streamlines of the 3D magnetic field and integrate them through the box volume
'''
streamlines = Streamlines(
    rad_buffer_obj,
    pos,
    ('gas', 'cell_centered_B_x'),
    ('gas', 'cell_centered_B_y'),
    ('gas', 'cell_centered_B_z'),
    length=1.0,
    get_magnitude=True,
)
streamlines.integrate_through_volume()
#%%
'''
Creating a matplotlib 3D plot to trace the streamlines through the 3D volume of the plot
'''
# Create a 3D plot, trace the streamlines through the 3D volume of the plot
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

for stream in streamlines.streamlines:
    stream = stream[np.all(stream != 0.0, axis=1)]
    ax.plot3D(stream[:, 0], stream[:, 1], stream[:, 2], alpha=0.5, linewidth=0.4, color='k')
    ax.view_init(90, 0, 90)

# Save the plot to disk.
plt.savefig("mag_field_lines.png")




