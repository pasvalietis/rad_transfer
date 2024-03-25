import os, sys

import yt
import yt_idv
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.visualization.colormaps import color_tables
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.emission_models import uv, xrt

# Load Glue
from glue_qt import qglue
#%%
from glue.core.data_factories import load_data
from glue.core import Data
from glue.core import DataCollection
from glue.core.link_helpers import LinkSame
from glue_qt.app.application import GlueApplication



#%%
# load subsampled MHD dataset

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path)

cut_box = subs_ds.region(center=[0.0, 0.5, 0.0],
                                left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

#%%
# Add synthetic XRT brightness
instr = 'xrt' #'aia'  # aia or xrt
channel = 'Ti-poly' #131 #'Be-thin' #131
#sdo_aia_model = uv.UVModel('temperature', 'density', channel)
hinode_xrt_model = xrt.XRTModel('temperature', 'density', channel)
xrt_synthetic = synt_img(cut_box, instr='xrt', channel='Ti-poly')
hinode_xrt_model.make_intensity_fields(subs_ds)
synth_plot_settings = {'resolution': 512}
synth_view_settings = {'normal_vector': [0.12, 0.05, 0.916],
                       'north_vector': [0.0, 1.0, 0.0]}

xrt_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                            view_settings=synth_view_settings,
                            image_shift=[-52, 105],
                            bkg_fill=10)

start_time = Time('2011-03-07T12:30:00', scale='utc', format='isot')
obs_scale = u.Quantity([0.6, 0.6], u.arcsec / u.pixel)
reference_pixel = u.Quantity([833.5, -333.5], u.pixel)
synth_map = xrt_synthetic.make_synthetic_map(obstime=start_time,
                                                 observer='earth',
                                                 detector='Synthetic AIA',
                                                 scale=obs_scale,
                                                 # reference_coord=reference_coord,
                                                 reference_pixel=reference_pixel)

df = subs_ds.to_glue([('gas', 'xrt_filter_band')])
# sdo_aia_model.make_intensity_fields(subs_ds)
#%%
#rc = yt_idv.render_context(height=800, width=800, gui=True)
#sg = rc.add_scene(subs_ds, "temperature", no_ghost=True) #xrt_filter_band
#sg.components[0].render_method = 'inferno'
#yt_idv.scene_components.base_component.SceneComponent(colormap='inferno')
#rc.run()
#
aia_fits = '/media/ivan/TOSHIBA EXT/aia_img/2011_event/flare_roi/calibrated/aia.lev1.5_euv_12s_roi.2011-03-07T141509.618.fits'
image = load_data(aia_fits)
#dc = DataCollection()
dc = DataCollection([image])
#%%


#%%
# start Glue
app = GlueApplication(dc)
app.start()
#%%



