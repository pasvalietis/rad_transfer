#!/usr/bin/env python

"""Script to overlay y points coordinates over the synthetic AIA image
import the file for corresponding timeframe and load the subsampled datacube
return the image with overplotted y points
"""

import os
import sys

import yt
from astropy.time import Time, TimeDelta
sys.path.insert(0, '/home/ivan/Study/Astro/solar')

from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from read_timeseries import gen_map_from_timeseries

'''
Generate synthetic AIA image for respective timeframe
'''

if __name__ == '__main__':
    start_time = Time('2011-03-07T12:30:00', scale='utc', format='isot')
    ds_dir = '/media/ivan/TOSHIBA EXT/subs'
    # sample j_z dataset
    downs_file_path = ds_dir + '/subs_3_flarecs-id_0050.h5'
    dataset = yt.load(downs_file_path, hint="YTGridDataset")

    synth_map = gen_map_from_timeseries(dataset, start_time, timescale=109.8)



#  main()

