import yt
from buffer import RadDataset  # How to solve yt.utilities.exceptions.YTAmbiguousDataType error?
from buffer import downsample
import os.path
from emission_models import xray_bremsstrahlung
from emission_models import uv

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np

import sunpy.visualization.colormaps as cm

original_file_path = '../datacubes/flarecs-id.0035.vtk'

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

downs_factor = 3
downs_file_path = './subs_dataset_' + str(downs_factor) + '.h5'

#%%
if not os.path.isfile(downs_file_path):
    ds = yt.load(original_file_path, units_override=units_override, hint='AthenaDataset')
    rad_buffer_obj = downsample(ds, rad_fields=True, n=downs_factor)
else:
    rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")

# Create x_ray fields and produce an image
#%%
thermal_model = xray_bremsstrahlung.ThermalBremsstrahlungModel("temperature", "density", "mass")
thermal_model.make_intensity_fields(rad_buffer_obj)

#%%
channel = 'A171'
sdo_aia_model = uv.UVModel("temperature", "density", channel)
sdo_aia_model.make_intensity_fields(rad_buffer_obj)
#%%
# Use sunpy AIA colormaps
sdoaia171 = matplotlib.colormaps['sdoaia171']
#%%
N = 512
nframes = 25
for i in range(nframes):
    norm_vec = [1.0 - i*(1./nframes), 0.0+i*(1./(nframes*3.)), i*(1./nframes)]
    prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
                            rad_buffer_obj,
                            [0.0, 0.5, 0.0],  # center position in code units
                            norm_vec,  # normal vector (z axis)
                            1.0,  # width in code units
                            N,  # image resolution
                            'aia_filter_band',  # respective field that is being projected
                            north_vector=[0.0, 1.0, 0.0])

    Mm_len = 1 # ds.length_unit.to('Mm').value

    X, Y = np.mgrid[-0.5*150*Mm_len:0.5*150*Mm_len:complex(0, N),
           0*Mm_len:150*Mm_len:complex(0, N)]

    #%%
    fig, ax = plt.subplots()
    data_img = np.array(prji)
    imag = data_img #+ 1e-17*np.ones((N, N))  # Eliminate zeros in logscale

    vmin=40
    vmax=150
    imag[imag == 0] = vmin

    pcm = ax.pcolor(X, Y, imag,
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                            #vmin=1e-5,
                            #vmax=1.5e4,
                            cmap=sdoaia171, shading='auto')
    int_units = str(prji.units)
    #fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
    fig.colorbar(pcm, ax=ax, extend='max', label='DN/pixel * '+'$'+int_units.replace("**", "^")+'$')
    ax.set_xlabel('x, Mm')
    ax.set_ylabel('y, Mm')

    #figpath = '../img/rad_tr_thermal_brem/'
    #plt.savefig(figpath + 'therm_brem_front_view_rad_buff.png')

    ax.set_title('AIA '+channel[1:]+' Å')
    #plt.show()
    figpath = '../img/rad_tr_sdo_aia/mov_'+channel+'/'
    plt.savefig(figpath + 'sdo_aia_'+channel+'_mov_'+str(i).zfill(3)+'.png')