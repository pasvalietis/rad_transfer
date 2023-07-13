import xrtpy

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
import pkg_resources

from pathlib import Path
from sunpy.map import Map

import scipy.ndimage
import scipy.misc
from scipy.ndimage import rotate

import os, sys
import yt

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.visualization.colormaps import color_tables

class ProfilePlot():

    def __init__(self, pix1, pix2, res, imag):
        self.pix1 = pix1
        self.pix2 = pix2
        self.imag = imag
        self.num = res

        x, y = np.linspace(self.pix1[0], self.pix2[0], self.num), \
               np.linspace(self.pix1[1], self.pix2[1], self.num)

        self.profile = scipy.ndimage.map_coordinates(self.imag, np.vstack((x, y)))

directory = "/home/ivan/Study/Astro/solar/rad_transfer/datasets/hinode/xrt/level1/"
data_file = Path(directory) / "L1_XRT20110307_180930.8.fits"

xrt_map = Map(data_file)
#%%
roi_bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=150*u.arcsec, frame=xrt_map.coordinate_frame)
roi_top_right = SkyCoord(Tx=-150*u.arcsec, Ty=500*u.arcsec, frame=xrt_map.coordinate_frame)
cusp_submap = xrt_map.submap(roi_bottom_left, top_right=roi_top_right)

xrt_img = np.array(cusp_submap.data)
xrt_img_rot = rotate(np.flip(np.rot90(xrt_img, k=2), axis=1), -19, reshape=False)

#%%

'''
Import dataset
'''

L_0 = (1.5e8, "m")
units_override = {
    "length_unit": L_0,
    "time_unit": (109.8, "s"),
    "mass_unit": (8.4375e38, "g"),  # m_0 = \rho_0 * (L_0 ** 3)
    "velocity_unit": (1.366e6, "m/s"),
    "temperature_unit": (1.13e8, "K"),
}

downs_file_path = '/home/ivan/Study/Astro/solar/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
subs_ds = yt.load(downs_file_path, units_override=units_override, hint='AthenaDataset')


#%%
instr = 'xrt'
channel = 'Ti-poly'
timestep = downs_file_path[-7:-3]
fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

for i in range(4):

    xrt_synthetic = synt_img(subs_ds, instr, channel)
    synth_plot_settings = {'resolution': xrt_img_rot.shape[0]}
    synth_view_settings = {'normal_vector': [0.1*(i+1), 0.2, -np.sqrt(1.-(0.1*(i+1)))],
                           'north_vector': [0.0, 1.0, 0.0]}
    xrt_synthetic.proj_and_imag(plot_settings=synth_plot_settings, view_settings=synth_view_settings)

    #%%
    synth_imag_rot = np.rot90(np.array(xrt_synthetic.image))


    '''
    Examine synth_image brightness distribution along the line
    '''

    zi = ProfilePlot([160, 175], [160, 5], 500, xrt_img_rot)
    prof2 = ProfilePlot([300, 210], [50, 210], 500, synth_imag_rot)

    '''
    Plot obs and synthetic image
    '''

    xrt_map = color_tables.xrt_color_table()

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    plt.ioff()
    plt.style.use('fast')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    gs = fig.add_gridspec(3, 4)

    ax1 = fig.add_subplot(gs[:2, :2])

    im = ax1.imshow(xrt_img_rot, norm=colors.LogNorm(vmin=1, vmax=7200), cmap=xrt_map)
    ax1.plot([zi.pix1[0], zi.pix2[0]], [zi.pix1[1], zi.pix2[1]], 'ro-')
    ax1.axis('image')
    ax1.set_title('Hinode XRT '+channel+' 2011-03-07 18:09:30')
    ax1.set_xlabel('X, Pixel')
    ax1.set_ylabel('Y, Pixel')

    ax2 = fig.add_subplot(gs[:2, 2:])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax2.set_title('Synthetic image, LOS: '+str(["%.2f" % member for member in synth_view_settings['normal_vector']]))#synth_view_settings['normal_vector']))
    ax2.set_xlabel('X, Pixel')
    ax2.set_ylabel('Y, Pixel')
    ax2.imshow(synth_imag_rot, norm=colors.LogNorm(vmin=1, vmax=4200), cmap=xrt_map)
    ax2.plot([prof2.pix1[1], prof2.pix2[1]], [prof2.pix1[0], prof2.pix2[0]], 'ro-')
    ax2.text(200, 50, channel, color='white')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('DN pixel$^{-1}$ s$^{-1}$', rotation=270, labelpad=13)

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(zi.profile, label='AIA')
    ax3.plot(prof2.profile, label='MHD Synthetic')
    #ax3.plot(prof2.profile/1.5, label='MHD Synthetic (2)', linestyle='--')
    ax3.legend()
    ax3.set_yscale('log')

    #ax4 = fig.add_subplot(gs[-1, -2])

    #plt.tight_layout()

    plt.savefig('xrt_profiles_proj_'+str(channel)+'_'+str(i)+'.png')
    plt.close()


#%%
# fig = plt.figure()
# cusp_submap.plot(title="Original Image")
# plt.show()

# filter = "C-poly"
#
# date_time = "2007-09-22T23:59:59"
#
# Temperature_Response_Fundamental = xrtpy.response.TemperatureResponseFundamental(
#     filter, date_time
# )
#
# temperature_response = Temperature_Response_Fundamental.temperature_response()
#
# print("Temperature Response:\n", temperature_response)

