import yt
import os
import sys
import numpy as np

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from emission_models import uv, xrt
from visualization.colormaps import color_tables

def make_filter_image(
        ds,
        instr,
        channel,
        figpath=os.getcwd(),
        norm_vec=(0.0, 0.0, 1.0),  # pass vectors as mutable arguments
        north_vector=(-0.7, -0.3, 0.0),
        resolution=512,
        vmin=1e-15,
        vmax=1e6,
        logscale=True,
        frame=None,
        label=None):

    cmap = {}
    imaging_model = None

    if instr == 'xrt':
        imaging_model = xrt.XRTModel("temperature", "density", channel)
        cmap['xrt'] = color_tables.xrt_color_table()
    if instr == 'aia':
        imaging_model = uv.UVModel("temperature", "density", channel)
        try:
            cmap['aia'] = color_tables.aia_color_table(int(channel)*u.angstrom)
        except ValueError:
            raise ValueError("AIA wavelength should be one of the following:"
                             "1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335.")

    imaging_model.make_intensity_fields(ds)
    field = str(instr) + '_filter_band'

    proj_and_imag(ds, field, list(norm_vec), list(north_vector), resolution, vmin, vmax,
                  cmap[instr], logscale, figpath, frame, label)
    return


def proj_and_imag(data, field, norm_vec=[0.0, 0.0, 1.0], north_vector=[-0.7, -0.3, 0.0], resolution=512,
                  vmin=1e-15, vmax=1e6, cmap='inferno', logscale=True, figpath='./prj_plt.png', frame=None, label=None):
    plt.ioff()

    prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
        data,
        [0.1, 0.5, 0.0],  # center position in code units
        norm_vec,  # normal vector (z axis)
        1.0,  # width in code units
        resolution,  # image resolution
        field,  # respective field that is being projected
        north_vector=north_vector)

    Mm_len = 1  # ds.length_unit.to('Mm').value
    X, Y = np.mgrid[-0.5 * 150 * Mm_len:0.5 * 150 * Mm_len:complex(0, resolution),
           0 * Mm_len:150 * Mm_len:complex(0, resolution)]
    fig, ax = plt.subplots()
    data_img = np.array(prji)

    imag = data_img
    if field == 'velocity_divergence':
        imag = np.abs(imag)

    imag[imag == 0] = vmin

    if logscale:
        pcm = ax.pcolor(X, Y, imag, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, shading='auto')
    else:
        pcm = ax.pcolor(X, Y, imag, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')

    int_units = str(prji.units)

    if 'xray' in field:
        cbar_label = '$'+int_units.replace("**", "^")+'$'
    elif 'DN' or 'uv' in field:
        cbar_label = 'DN/pixel * '+'$'+int_units.replace("**", "^")+'$'

    fig.colorbar(pcm, ax=ax, extend='max', label=cbar_label)
    ax.set_xlabel('x, Mm')
    ax.set_ylabel('y, Mm')

    if label != None:
        plt.text(40, 135, label, color='white') #bbox=dict(fill=False, color='white', linewidth=0.0))
    # figpath = '../img/rad_tr_thermal_brem/'
    if frame == None:
        if label != None:
            plt.savefig(figpath + field + '_' + label + '.png')
        else:
            plt.savefig(figpath)
            plt.close()
            print('fig saved at: ', figpath)
    else:
        plt.savefig(figpath + field + '_' + str(frame) + '.png')

    return imag
