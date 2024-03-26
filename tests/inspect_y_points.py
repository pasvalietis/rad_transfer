#!/usr/bin/env python

"""Script to investigate the distribution of y-points in the MHD model
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
matplotlib.use("qt5agg")  # matplotlib.use('TkAgg')

import os
import sys
import numpy as np

import yt
from yt.visualization.image_writer import write_image

import time  # To test runs in parallel

from current_density import _current_density
sys.path.insert(0, '/home/ivan/Study/Astro/solar')

start_time = time.time()
yt.enable_parallelism()

def detect_sharp_slope(arr):
    pass

#%%
def find_roots(x,y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
#%%

if __name__ == '__main__':
    ds_dir = '/media/ivan/TOSHIBA EXT/subs'

    # sample j_z dataset
    downs_file_path = ds_dir + '/subs_3_flarecs-id_0054.h5'
    dataset = yt.load(downs_file_path, hint="YTGridDataset")

    rad_buffer_obj = dataset
    grad_fields_x = rad_buffer_obj.add_gradient_fields(("gas", "magnetic_field_x"))
    grad_fields_y = rad_buffer_obj.add_gradient_fields(("gas", "magnetic_field_y"))
    grad_fields_z = rad_buffer_obj.add_gradient_fields(("gas", "magnetic_field_z"))

    rad_buffer_obj.add_field(
        ("gas", "current_density"),
        function=_current_density,
        sampling_type="cell",
        units="G/cm**2",
        take_log=False,
        force_override=True
        # validators=[ValidateParameter(["center", "bulk_velocity"])],
    )

    #%%
    # make a cut along x = 0
    nslices = 1  # 20
    axes = 'xyz'
    nax = 2
    axis = axes[nax]

    coord_range = np.linspace(rad_buffer_obj.domain_left_edge[nax].value,
                              rad_buffer_obj.domain_right_edge[nax].value,
                              nslices)

    for i in range(nslices):

        coord = coord_range[i]
        if nslices == 1:
            coord = -0.2500

        if i == 0 or i == nslices - 1:
            coord = np.trunc(coord*1e3)/1e3  # floor the coordinate to avoid including the edges of the domain

        print("Creating a ds slice: %s seconds" % (time.time() - start_time))
        slc = rad_buffer_obj.slice(axis, coord)  #, 'current_density')
        print("Creating a fixed resolution buffer: %s seconds" % (time.time() - start_time))
        slc_frb = slc.to_frb((140.0, "Mm"), 512)
        print("Exporting data from the FRB: %s seconds" % (time.time() - start_time))
        jz_arr = slc_frb["gas", "current_density"].d
        print("Finished export from the FRB: %s seconds" % (time.time() - start_time))
        # cs_profile =
        if axis == 'z':
            z_coord = coord
        else:
            z_coord = None
#%%
        print("Cast a yt ray: %s seconds" % (time.time() - start_time))
        cs_loc = rad_buffer_obj.ray([-0.05, 0.98, z_coord],
                                    [0.05, 0.98, z_coord])  # Identify x coordinate of the current sheet
        print("Export profile of j_z from the yt ray: %s seconds" % (time.time() - start_time))
        cs_loc_profile = cs_loc[('gas', 'current_density')].value
        cs_max_x_coord = cs_loc.argmax(('gas', 'current_density'))[0].value

        cs_width_pix = 3  # three pixels
        print("Export profile of j_z from the yt ray: %s seconds" % (time.time() - start_time))
        cs_ray = rad_buffer_obj.ray([cs_max_x_coord, 0.0, z_coord], [cs_max_x_coord, 1.0, z_coord])

        cs_profile_coords = cs_ray.fcoords.value[:, 1]  # cs_profile ray coordinates along y

        cs_slit = np.zeros((cs_ray.fcoords[:, 0].shape[0], cs_width_pix))
#%%
        print("Create a cut along RCS: %s seconds" % (time.time() - start_time))
        for j in range(cs_width_pix):
            dx = 0.0045
            x_slit = cs_max_x_coord - (dx * (cs_width_pix // 2)) + j * dx
            print("Create a cut along RCS_ "+str(j)+": %s seconds" % (time.time() - start_time))
            cs_ray = rad_buffer_obj.ray([x_slit, 0.0, z_coord], [x_slit, 1.0, z_coord])
            # Normalize current density along the slit
            idx_max = np.argmin(cs_profile_coords - 0.96)
            print("Export j_z data in the ray_"+ str(j) + ": %s seconds" % (time.time() - start_time))
            max_val = cs_ray[('gas', 'current_density')].value[idx_max] #.max()
            cs_slit[:, j] = cs_ray[('gas', 'current_density')].value
#%%
        # cs_profile = cs_ray[('gas', 'current_density')].value
        # Find average value along the slit
        #cs_profile = np.sqrt(np.mean(cs_slit, axis=1)/ max_val)
        init_profile = np.mean(cs_slit, axis=1) / np.mean(cs_slit, axis=1)[np.argmin(cs_profile_coords - 0.96)]
        single_pix_profile = cs_slit[:, 1] / cs_slit[:, 1][np.argmin(cs_profile_coords - 0.96)]
        cs_profile = (np.mean(cs_slit, axis=1) / max_val) ** 2. # np.mean(cs_slit, axis=1) # / max_val

        # smoothing
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved = np.convolve(cs_profile, kernel, mode='same')

#%%
        # plt.plot(np.linspace(0, 1, cs_profile.shape[0]), cs_profile)

#%%
        print("Producing a matplotlib plot: %s seconds" % (time.time() - start_time))
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 7.0))
        im_cmap = 'BuPu'  #RdBu_r
        if axis == 'x':
            im = ax.imshow(np.rot90(jz_arr), origin='upper', vmin=8e-7, vmax=1e-5, cmap=im_cmap)
        elif axis == 'z':
            im = ax.imshow(jz_arr, origin='lower', vmin=8e-7, vmax=1e-5, cmap=im_cmap, extent=(-0.5, 0.5, 0, 1.0))
        else:
            im = None

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax1, label="Current density $j_z$", orientation="vertical")
        ax.spines['left'].set_color('none')
        ax.set_yticks([])
        ax.axvline(x=cs_max_x_coord, color='magenta', alpha=0.5,  linestyle='--', label='axvline - full height')

        ax.set_title("Current density $j_z$, "+axis+"="+f'{coord:.4f}')

        # divider2 = make_axes_locatable(ax)
        ax11 = divider.append_axes("left", size="40%", pad=0.30)
        ax11.plot(cs_profile, cs_profile_coords, linewidth=0.65, color='indigo')
        #ax11.plot(np.gradient(cs_profile, np.linspace(0, 1, cs_profile.shape[0])),
        #          np.linspace(0, 1, cs_profile.shape[0]))
        ax11.set_ylim(0, 1)
        ax11.set_xlim(0, 1.0)
        ax11.set_ylabel('y, code length')
        ax11.set_xlabel('$j_z$')

        ax12 = ax11.twiny()
        color = 'tab:blue'
        #ax12.set_xlabel('$dj_z/dy$', color=color)  # we already handled the x-label with ax1
        y_cut = np.linspace(0, 1, cs_profile.shape[0])
        cs_der_y = np.gradient(cs_profile, y_cut)
        #ax12.plot(cs_der_y, y_cut, color=color, linewidth=0.65, alpha=0.75)
        ax12.plot(init_profile, cs_profile_coords, label='init', linestyle='--')
        ax12.plot(single_pix_profile, cs_profile_coords, label='1Pix', linestyle='--')
        ax12.plot(data_convolved, cs_profile_coords, label='smooth', linestyle='--')
        ax12.legend()
        ax12.tick_params(axis='y', labelcolor=color)
        ax12.set_ylim(0, 1)
        ax12.set_xlim(0., 1.0)
        # ax12.set_xlim(0.25*cs_der_y.min(), 0.25*cs_der_y.max())

        '''
        Identifying location of y-point from the maximum of the first derivative
        '''
        crd_idx = cs_der_y[np.where((cs_profile_coords > 0.75) & (cs_profile_coords < 0.95))].argmax()
        yp_ycoord = cs_profile_coords[np.where((cs_profile_coords > 0.75) & (cs_profile_coords < 0.95))][crd_idx]
        # ax.axhline(y=yp_ycoord, color='magenta', alpha=0.2)
        # Plot the location of the y point
        # ax.scatter([cs_max_x_coord], [yp_ycoord], marker='x', color='magenta')
        #ax12.set_yticks([])
        # plt.show()

        '''
        Alternative method: find where j_z is about one half of initial value on top of the domain (y~0.95)
        '''
        init_idx = np.argmin(abs(cs_profile_coords - 0.96))
        half_idx = np.argmin(abs((cs_profile) - 0.30*cs_profile[init_idx]))

        yp_ycoord = cs_profile_coords[half_idx]

        #data_convolved
        init_idx = np.argmin(abs(cs_profile_coords - 0.96))
        half_idx = np.argmin(abs((data_convolved) - 0.30 * cs_profile[init_idx]))
        # determine intersections
        int_point_coords = find_roots(cs_profile_coords, data_convolved - 0.30 * cs_profile[init_idx])
        # keep intersection points that are not llocated around local minima or maxima (filter symmetric roots)
        '''
        # coords of local minima:
        from scipy.signal import argrelmin
        argrelmin(data_convolved) 
        # coords of local maxima:
        from scipy import signal
        peakind = signal.find_peaks_cwt(data_convolved, np.arange(0.1,0.5))
        *or*
        peaks, _ = find_peaks(data_convolved)
        '''
        yp_ycoord_conv = cs_profile_coords[half_idx]

        ax.scatter([cs_max_x_coord], [yp_ycoord], marker='x', color='red')
        ax.axhline(y=yp_ycoord, color='red', alpha=0.2)

        ax.scatter([cs_max_x_coord], [yp_ycoord_conv], marker='x', color='green')
        ax.axhline(y=yp_ycoord_conv, color='green', alpha=0.2)


        ax22 = divider.append_axes("bottom", size="25%", pad=0.5)
        ax22.plot(np.linspace(-0.05, 0.05, cs_loc_profile.shape[0]), cs_loc_profile, linewidth=0.65, color='magenta')
        ax22.axvline(x=cs_max_x_coord, color='red', alpha=0.95, linestyle='-', label='axvline - full height')
        ax22.set_ylabel('$j_z$')
        ax22.set_xlabel('x, code length')

        plt.savefig('cur_dens_slices/current_density_slice_'+axis+'_'+f'{coord:.4f}'+'.png')
        plt.close()
        print(i)

print("--- %s seconds ---" % (time.time() - start_time))
