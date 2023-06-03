import yt
yt.enable_parallelism()

import numpy as np
import sys
import os.path

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pyxsim

sys.path.insert(0, '/home/ivan/Study/Astro/solar')

from rad_transfer.buffer import downsample
from rad_transfer.emission_models import xray_bremsstrahlung


#%% Load input data
downs_factor = 3
original_file_path = '../datacubes/flarecs-id.0035.vtk'
downs_file_path = './subs_dataset_' + str(downs_factor) + '.h5'

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

if not os.path.isfile(downs_file_path):
    ds = yt.load(original_file_path, units_override=units_override,
                 default_species_fields='ionized', hint='AthenaDataset')
    # Specifying default_species_fields is required to produce emission_measure field so that PyXSIM thermal
    # emission model can be applied

    rad_buffer_obj = downsample(ds, rad_fields=True, n=downs_factor)
else:
    rad_buffer_obj = yt.load(downs_file_path, hint="YTGridDataset")

#%% Add emission_measure field
# def _em_field(field, data):
#     return (
#         data["gas", "density"]**2.
#     )
#
# rad_buffer_obj.add_field(
#     name=("gas", "emission_measure"),
#     function=_em_field,
#     sampling_type="local",
#     units="g**2/cm**6",
#     force_override=True,
# )

#%% Define projection and imaging
def proj_and_imag(data, field, norm_vec=[0.0, 0.0, 1.0], resolution=512, vmin=1e-15, vmax=1e6, cmap='inferno',
                  logscale=True):
    prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
                            data,
                            [0.0, 0.5, 0.0],  # center position in code units
                            norm_vec,  # normal vector (z axis)
                            1.0,  # width in code units
                            resolution,  # image resolution
                            field,  # respective field that is being projected
                            north_vector=[0.0, 1.0, 0.0])
    Mm_len = 1  # ds.length_unit.to('Mm').value
    X, Y = np.mgrid[-0.5 * 150 * Mm_len:0.5 * 150 * Mm_len:complex(0, resolution),
           0 * Mm_len:150 * Mm_len:complex(0, resolution)]
    fig, ax = plt.subplots()
    data_img = np.array(prji)
    imag = data_img

    #vmin = 0.8
    #vmax = 80
    imag[imag == 0] = vmin

    if logscale:
        pcm = ax.pcolor(X, Y, imag, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, shading='auto')
    else:
        pcm = ax.pcolor(X, Y, imag, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
    int_units = str(prji.units)
    # fig.colorbar(pcm, ax=ax, extend='max', label='$'+int_units.replace("**", "^")+'$')
    if 'xray' in field:
        cbar_label = '$'+int_units.replace("**", "^")+'$'
    elif 'DN' or 'uv' in field:
        cbar_label = 'DN/pixel * '+'$'+int_units.replace("**", "^")+'$'
    fig.colorbar(pcm, ax=ax, extend='max', label=cbar_label)
    ax.set_xlabel('x, Mm')
    ax.set_ylabel('y, Mm')

    figpath = '../img/rad_tr_thermal_brem/'
    plt.savefig(figpath + field +'.png')

#%%
"""
It is assumed here that the photon energy is fixed at 6 keV
"""
rd_thermal_model = xray_bremsstrahlung.ThermalBremsstrahlungModel("temperature", "density", "mass")
rd_thermal_model.make_intensity_fields(rad_buffer_obj, 6.0, 12.0)

#%% Apply PyXSIM thermal bremsstrahlung to a subsampled datacube
emin = 6.0
emax = 12.0
#px_thermal_model = pyxsim.CIESourceModel("apec", emin, emax, 100, 0.2, binscale='log')
#px_thermal_model.make_intensity_fields(ds, emin, emax, dist=(1.5e11, "m"))

#%% Produce images
#proj_and_imag(ds, 'xray_photon_intensity_6.0_12.0_keV', vmin=1e-3, vmax=2e1, cmap='jet', logscale=False)

proj_and_imag(rad_buffer_obj, 'xray_intensity_keV', vmin=1e-15, vmax=1e6, cmap='jet')

#%%
# Try to vectorize n-dim integration
# from scipy.integrate import quad
# ndim = 80
# arr = 10**(0.5+np.random.rand(ndim, ndim, ndim))
# brr = np.random.rand(ndim, ndim, ndim)
#
# def integrand(dens, temp, energy):
#     return dens*dens*np.sqrt(temp)*np.exp(-energy)
#
# def energy_int(dens, temp, emin, emax):
#     return quad(lambda energy: integrand(dens, temp, energy), emin, emax)
#
# vect_eint = np.vectorize(energy_int)
# res = vect_eint(arr, brr, 6., 12.)[0]
#
# print(arr)
# print(brr)
# print(res)

#efunc = [lambda energy: np.ravel(integrand(np.ravel(arr)[i], energy)) for i in range(len(np.ravel(arr)))]
#np.reshape(np.vectorize(quad)(efunc, 6., 12.)[0], np.shape(arr))