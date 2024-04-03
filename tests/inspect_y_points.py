#!/usr/bin/env python

"""Script to investigate the distribution of y-points in the MHD model
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
matplotlib.use("qt5agg")  # matplotlib.use('TkAgg')

import os
import sys
from pathlib import Path
import copy
import numpy as np
import pickle

import yt
import unyt
from yt.visualization.image_writer import write_image

import time  # To test runs in parallel

from current_density import _current_density
# Identify features in the current density profile
from scipy.signal import argrelmin, find_peaks_cwt
from scipy.optimize import curve_fit
sys.path.insert(0, '/home/ivan/Study/Astro/solar')

start_time = time.time()
yt.enable_parallelism()

def detect_sharp_slope(arr):
    pass

'''
Gaussian fits, from 
https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466
'''
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

#%%
def find_roots(x,y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
#%%


def has_a_peak_counterpart(profile, coords, threshold, peak_range=np.arange(0.1, 0.5)):
    """Checking whether a root (the intersection point of a current density profile) has a mirror counterpart on the
     other side of the peak (local minimum or maximum).
    :param profile: 1D profile to be analyzed to find a location of y-point
    :type profile: 1D numpy array
    :param coords: 1D profile containing profile coordinates
    :type coords: 1D numpy array
    :param threshold: level to identify location of the y-point (should be between 0.15 and 0.45)
    :type threshold: float
    :param peak_range: range to detect peaks using wavelet transform
    :type peak_range: np.arange, default from 0.1 to 0.5
    :return: bool array of initial root points, if intersection point (root) has a symmetric counterpart next to closest
    peak
    """

    # init_idx = np.argmin(abs(cs_profile_coords - 0.96))
    # half_idx = np.argmin(abs((cs_profile) - threshold * cs_profile[init_idx]))

    '''
    Find the roots of the original profile, where it intersects with a specific threshold level
    '''

    # Determine the coordinates (float) of the intersection points between the profile and the threshold line
    root_point_coords = find_roots(coords, profile - threshold)

    if len(root_point_coords) == 0:
        raise ValueError('No intersections with a threshold')

    # Find coordinates of local minima and maxima
    minima = argrelmin(profile)
    maxima = find_peaks_cwt(profile, peak_range)
    # indices of coord array where extrema occur
    ind_extrema = np.concatenate((minima[0], maxima), axis=0)
    flags = np.full(len(root_point_coords), True)

    for point_idx in range(len(root_point_coords)): # iterate over root points array
        point_coord_idx = np.abs(coords - root_point_coords[point_idx]).argmin()
        point_coord = coords[point_coord_idx]

        # Identify closest peak
        peaks_coords = coords[ind_extrema]
        closest_peak_idx = np.abs(peaks_coords - point_coord).argmin()
        closest_peak_coord = peaks_coords[closest_peak_idx]
        # distance to peak
        peak_dist = closest_peak_coord - point_coord

        # Find a mirror counterpart at the closest peak location
        step = np.diff(coords)
        #mirr_point_coord = (np.where((root_point_coords > 355) & (arr < 357)))[0].size
        if peak_dist > 0:
            mp_idx = (np.where((root_point_coords > point_coord) & ((point_coord + 2*peak_dist) > root_point_coords)))[0] #.size
            flags[mp_idx] = False
            flags[point_idx] = True

        else:
            mp_idx = (np.where((root_point_coords < point_coord) & ((point_coord + 2 * peak_dist) < root_point_coords)))[0]  # .size
            flags[mp_idx] = False
            flags[point_idx] = True

        #np.isclose(0.5378, 0.5400, atol=0.001)
        #mirror_element = coords[np.where(np.isclose(0.5378, 0.5400, atol=0.001))]

        # if there is a mirror root, set flag to False

    y_point_coord = root_point_coords[np.where(flags == True)][-1]
    print(' ---------- y-point filtering routine ----------')
    print('flags: ', flags)
    print('y-point coordinate', y_point_coord)
    print(' -----------------------------------------------')
    # pick a first result as a position of y point

    return y_point_coord
#%%

if __name__ == '__main__':
    ds_dir = '/media/ivan/TOSHIBA EXT/subs'

    # sample j_z dataset
    downs_file_path = ds_dir + '/subs_3_flarecs-id_0050.h5'
    dataset = yt.load(downs_file_path, hint="YTGridDataset")
    cur_time = dataset.current_time.value.item()

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
    nslices = 20  # 20
    axes = 'xyz'
    nax = 2
    axis = axes[nax]

    coord_range = np.linspace(rad_buffer_obj.domain_left_edge[nax].value,
                              rad_buffer_obj.domain_right_edge[nax].value,
                              nslices)

    ypoint_coords = {'timestep': cur_time, 'coordinates': []}

    for i in range(nslices):

        coord = coord_range[i]
        if nslices == 1:
            coord = 0.2490

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
        cs_loc_height = 0.98
        cs_loc = rad_buffer_obj.ray([-0.05, cs_loc_height, z_coord],
                                    [0.05, cs_loc_height, z_coord])  # Identify x coordinate of the current sheet
        print("Export profile of j_z from the yt ray: %s seconds" % (time.time() - start_time))
        cs_loc_profile = cs_loc[('gas', 'current_density')].value
        cs_x_coords = cs_loc.fcoords.value[:, 0]  # cs_profile ray coordinates along x

        cs_width_pix = 3  # three pixels
        print("Export profile of j_z from the yt ray: %s seconds" % (time.time() - start_time))
        # Use gaussian fit to identify center of the current sheet
        H, A, x0, sigma = gauss_fit(cs_x_coords, cs_loc_profile)
        FWHM = 2.35482 * sigma
        print('The center of the gaussian fit is', x0)
        # cs_max_x_coord = cs_loc.argmax(('gas', 'current_density'))[0].value
        cs_max_x_coord = x0

        cs_ray = rad_buffer_obj.ray([cs_max_x_coord, 0.0, z_coord], [cs_max_x_coord, 1.0, z_coord])

        cs_profile_coords = cs_ray.fcoords.value[:, 1]  # cs_profile ray coordinates along y

        cs_slit = np.zeros((cs_ray.fcoords[:, 0].shape[0], cs_width_pix))

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

        ax.set_title("Current density $j_z$, "+axis+"="+f'{coord:.4f}'+', timestep = '+f'{cur_time:.1f}')

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
        half_idx = np.argmin(abs((data_convolved) - 0.15 * cs_profile[init_idx]))
        # determine intersections
        int_point_coords = find_roots(cs_profile_coords, data_convolved - 0.15 * cs_profile[init_idx])
        # keep intersection points that are not llocated around local minima or maxima (filter symmetric roots)

        '''
        # coords of local minima:
        from scipy.signal import argrelmin
        argrelmin(data_convolved) 
        # coords of local maxima:
        from scipy.signal import find_peaks_cwt
        peakind = find_peaks_cwt(data_convolved, np.arange(0.1,0.5))
        *or*
        peaks, _ = find_peaks(data_convolved)
        '''

        # has_a_peak_counterpart(profile, coords, threshold)

        yp_ycoord_conv = has_a_peak_counterpart(data_convolved, cs_profile_coords, 0.15) # cs_profile_coords[half_idx]

        ax.scatter([cs_max_x_coord], [yp_ycoord], marker='x', color='red')
        ax.axhline(y=yp_ycoord, color='red', alpha=0.2)

        ax.scatter([cs_max_x_coord], [yp_ycoord_conv], marker='x', color='green')
        ax.axhline(y=yp_ycoord_conv, color='green', alpha=0.2)

        ax22 = divider.append_axes("bottom", size="25%", pad=0.5)
        ax22.plot(np.linspace(-0.05, 0.05, cs_loc_profile.shape[0]), cs_loc_profile, linewidth=0.65, color='magenta')
        #ax22.plot(xdata, ydata_perfect, '-k', label='data (without_noise)')
        ax22.plot(cs_x_coords, gauss(cs_x_coords,
                                           *gauss_fit(cs_x_coords, cs_loc_profile)), '--r', label='fit')
        # gauss_fit(cs_profile_coords, cs_loc_profile)
        ax22.axvline(x=cs_max_x_coord, color='red', alpha=0.95, linestyle='-', label='axvline - full height')
        ax22.set_ylabel('$j_z$')
        ax22.set_xlabel('x, code length')

        dir_name = './cur_dens_slices/'+'00'+f'{10*cur_time:.0f}'
        # os.mkdir(dir_name)
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(dir_name+'/current_density_slice_'+axis+'_'+f'{coord:.4f}'+'.png')
        #plt.close()

        coords = cs_loc_profile
        ypoint_coords['coordinates'].append((cs_max_x_coord, yp_ycoord_conv, coord)*cs_ray.fcoords.units)
        print(i)

#%%
with open('cur_dens_slices/y_points_'+'00'+f'{10*cur_time:.0f}'+'.pickle', 'wb') as handle:
    pickle.dump(ypoint_coords, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
print("--- %s seconds ---" % (time.time() - start_time))