import os, sys

import yt
import yt_idv

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.visualization.colormaps import color_tables
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

#%%
# load subsampled MHD dataset

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path)

cut_box = subs_ds.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
# Add synthetic AIA brightness
instr = 'xrt' #'aia'  # aia or xrt
channel = 'Ti-poly' #131 #'Be-thin' #131
#sdo_aia_model = uv.UVModel('temperature', 'density', channel)
hinode_xrt_model = xrt.XRTModel('temperature', 'density', channel)

hinode_xrt_model.make_intensity_fields(subs_ds)
# sdo_aia_model.make_intensity_fields(subs_ds)
#%%
rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(subs_ds, "temperature", no_ghost=True) #xrt_filter_band
#sg.components[0].render_method = 'inferno'
#yt_idv.scene_components.base_component.SceneComponent(colormap='inferno')
rc.run()
#

