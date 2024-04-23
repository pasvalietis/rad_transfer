#!/usr/bin/env python

"""Script to overlay y points coordinates over the synthetic AIA image
import the file for corresponding timeframe and load the subsampled datacube
return the image with overplotted y points
"""

import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import yt
import astropy.units as u
from unyt import unyt_array
from astropy.time import Time, TimeDelta

sys.path.insert(0, '/home/ivan/Study/Astro/solar')

from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from rad_transfer.visualization.colormaps import color_tables
from read_timeseries import gen_map_from_timeseries
from read_timeseries import read_dataset
'''
Generate synthetic AIA image for respective timeframe
'''

#%%
def project_point_to_imag(point, norm_vector):
    """
    Assuming that norm_vector is a normal vector to the image plane on which the dataset is being projected,
    one can calculate the 3d coordinates of the points projected on that plane.
    :param point: np or yt array of point coordinates
    :param norm_vector: normal vector of the plane
    :return: np or yt array containing coordinates of the projected point
    """
    # Convert vec np.array to unyt.array providing code units
    # norm_vector = unyt_array(norm_vector, point.units)
    normal = norm_vector / np.linalg.norm(norm_vector)
    projected_point = point - np.dot(point, normal) * normal
    return projected_point


#%%
def proj_points_2d_coords_in_imag(point, norm_vector, north_vector):
    """
    Calculate point coordinates in the image plane reference frame by providing two base vectors:
    north_vector and a vector that is orthogonal to the north_vector and norm_vector
    :param point: point to be projected in dataset coordinates
    :param base_vec_1: unyt array: north_vector
    :param base_vec_2: unyt array: np.cross(north_vector, norm_vector)
    :return:
    """
    proj_point = project_point_to_imag(point, norm_vector)

    # second vector orthogonal to the north_vector to define a pair of basis vectors of the image plane
    base_vec = unyt_array(np.cross(norm_vector, north_vector), proj_point.units)
    base_vec_1, base_vec_2 = north_vector, base_vec
    x = np.dot(point, base_vec_1)
    y = -np.dot(point, base_vec_2)

    return x, y

#%%
if __name__ == '__main__':
    # %%
    start_time = Time('2011-03-07T12:30:00', scale='utc', format='isot')
    # fns is a list of all the simulation data files in the current directory.
    ds_dir = '/media/ivan/TOSHIBA EXT/subs'
    ts = read_dataset(ds_dir)

    # sample j_z dataset
    #downs_file_path = ds_dir + '/subs_3_flarecs-id_0050.h5'
    #dataset = yt.load(downs_file_path, hint="YTGridDataset")
#%%
    
    # Read y-point dictionary from a pickle file:
    pickle_file_name = './cur_dens_slices/y_points_0050.pickle'
    with open(pickle_file_name, 'rb') as yfile:
        y_points = pickle.load(yfile)

    sample_point = y_points['coordinates'][17]

    synth_map, proj_params = gen_map_from_timeseries(dataset, start_time, timescale=109.8)
    norm_vec = unyt_array(proj_params[0]['normal_vector'], sample_point.units)
    north_vec = unyt_array(proj_params[0]['north_vector'], sample_point.units)

    projected_points = []
    for point in y_points['coordinates']:
        proj_point = proj_points_2d_coords_in_imag(point, norm_vec, north_vec)
        projected_points.append(proj_point)
        print('point', point, 'proj_point', proj_point)

    rcs_bottom = np.array(projected_points)
#%%
    # Display synthetic image
    xsize, ysize = 5.5, 4.5
    fig, ax = plt.subplots(1, 1, figsize=(xsize, ysize), dpi=140)
    Mm_len = 1
    N = proj_params[1]['resolution']
    # X, Y = np.mgrid[-0.5 * 150 * Mm_len:0.5 * 150 * Mm_len:complex(0, N),
    #        0 * Mm_len:150 * Mm_len:complex(0, N)]
    vmin, vmax = 1e0, 3e2
    ax.imshow(np.transpose(synth_map.data), origin='lower', extent=(-0.5, 0.5, 0, 1.0), norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    ax.scatter(rcs_bottom[:, 1], rcs_bottom[:, 0], color='r', marker='+')
#%%
   # Use yt plotting window with annotate_marker
    L = norm_vec  # vector normal to cutting plane
    north_vector = north_vec
    prj = yt.ProjectionPlot(
        dataset, L, ('gas', 'aia_filter_band'), width=(1, sample_point.units), north_vector=north_vector,
    )
    prj.set_cmap(field=("gas", "aia_filter_band"), cmap=color_tables.aia_color_table(int(131) * u.angstrom))
    prj.set_zlim(('gas', 'aia_filter_band'), zmin=(1e0, "1/s"), zmax=(3e2, "1/s"))
    for ypt in y_points['coordinates']:
        prj.annotate_marker(ypt.value, coord_system="data")
    prj.save()


#  main()

