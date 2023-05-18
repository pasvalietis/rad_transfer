import matplotlib.pyplot as plt
import yt
import numpy as np
import sunpy
import pickle

import matplotlib.pyplot as plt

import astropy.units as u
from aiapy.response import Channel
from scipy.io import readsav
from scipy import interpolate
from preload_subs import load_subs

trm_path = 'sdo_aia/aia_temp_response.npy'
aia_trm = np.load(trm_path, allow_pickle=True)

#%%
ch_ = 1  # 131A

aia_trm_interpf = interpolate.interp1d(
    aia_trm.item()['logt'],
    aia_trm.item()['temp_response'][:, ch_],
    fill_value="extrapolate",
    kind='cubic',
)

#%%
temp_x = np.logspace(5, 8, 290)
tlog = np.log10(temp_x)
plt.loglog(tlog, aia_trm_interpf(tlog))
plt.show()

#%%
'''
Sample 3d np.array of temperatures
Try to pass it into a 3d interpolated function
'''

temps = 10**(5 + np.random.rand(3, 3, 3))
aia_response = aia_trm_interpf(temps)

#%%
'''
load the pre-edited subsampled datacube
'''

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    # "density_unit": (2.5e14, "kg/m**3"),
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

indstype = 'full'
path_default = "datacubes/flarecs-id.0035.vtk"
path_subs = "datacubes/flarecs-id.0035_ss3.h5"

if indstype == 'full':
    path = path_default
    ds = yt.load(path, units_override=units_override, default_species_fields='ionized')
elif indstype == 'subs':
    path = path_subs
    ds = yt.load(path)

#%%
def _aia_filter_band(field, data):
    norm = yt.YTQuantity(1.0, "cm**5/s/(g**2)")
    return(norm * data["gas", "density"] * data["gas", "density"] *
           aia_trm_interpf(np.log10(np.abs(data["gas", "temperature"]))))

ds.add_field(
    name=("gas", "aia_filter_band"),
    function=_aia_filter_band,
    sampling_type="local",
    units="1/(cm*s)",
    force_override=True,
)

#%%
'''
Project the ('gas', 'aia_filter_band') field to produce UV image
'''

N = 512
norm_vec = [1.0, 0.0, 0.0]
prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(ds,
                        [0.0, 0.5, 0.0],  # center position in code units
                        norm_vec,  # normal vector (z axis)
                        1.0,  # width in code units
                        N,  # image resolution
                        ('gas', 'aia_filter_band'),  # respective field that is being projected
                        north_vector=[0.0, 1.0, 0.0]
                        )

#%%
import matplotlib.colors as colors
Mm_len = ds.length_unit.to('Mm').value
Z, Y = np.mgrid[-0.25*Mm_len:0.25*Mm_len:complex(0, N), 0*Mm_len:Mm_len:complex(0, N)]
fig, ax = plt.subplots()
data_img = np.array(prji)
imag = data_img

pcm = ax.pcolor(Z, Y, imag,
                       norm=colors.LogNorm(vmin=7, vmax=7e2),
                       cmap='Blues_r', shading='auto')
int_units = str(prji.units)
fig.colorbar(pcm, ax=ax, extend='max', label='DN/pixel * '+'$'+int_units.replace("**", "^")+'$')
ax.set_xlabel('z, Mm')
ax.set_ylabel('y, Mm')

#plt.show()
plt.savefig('img/rad_tr_sdo_aia/aia_131.png')