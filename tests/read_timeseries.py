import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib

import yt
import os, sys, re
import glob  # to load specific timeframes

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

import sunpy.map

# from sunpy.instr.aia import aiaprep
from sunpy.net import Fido
from sunpy.net import attrs as a
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/ivan/Study/Astro/solar')
from rad_transfer.utils.proj_imag import SyntheticFilterImage as synt_img
from visualization.colormaps import color_tables

# matplotlib.use('Qt5Agg')
yt.enable_parallelism()


def read_dataset(ds_dir):
    ds_lst = glob.glob(ds_dir+"/*.h5") #[1][0-9].h5")
    ds_lst.sort(key=lambda f: int(re.sub('\D', '', f)))  # Sort filenames in dataset list in ascending order
    ts = yt.DatasetSeries(ds_lst)
    return ts

#%%
def gen_map_from_timeseries(ds, start_time, timescale=109.8):
    cut_box = ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

    timestep = ds.current_time.value.item()
    timediff = TimeDelta(timestep * timescale * u.s)

    instr = 'aia'  # aia or xrt
    channel = 131

    aia_synthetic = synt_img(cut_box, instr, channel)
    samp_resolution = 584  # cusp_submap.data.shape[0]
    obs_scale = u.Quantity([0.6, 0.6], u.arcsec / u.pixel)  # [0.6, 0.6]*(u.arcsec/u.pixel)
    reference_pixel = u.Quantity([833.5, -333.5], u.pixel)
    reference_coord = None

    img_tilt = -23 * u.deg

    synth_plot_settings = {'resolution': samp_resolution}
    synth_view_settings = {'normal_vector': [0.12, 0.05, 0.916],
                           'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value, 0.0]}

    aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[-52, 105],
                                bkg_fill=10)

    synth_map = aia_synthetic.make_synthetic_map(obstime=start_time+timediff,
                                                 observer='earth',
                                                 detector='Synthetic AIA',
                                                 scale=obs_scale,
                                                 # reference_coord=reference_coord,
                                                 reference_pixel=reference_pixel)

    return synth_map

#%%
def download_maps():
    start_time = Time('2011-03-07T12:30:00', scale='utc', format='isot')
    bottom_left = SkyCoord(Tx=-500*u.arcsec, Ty=200*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
    top_right = SkyCoord(Tx=-150*u.arcsec, Ty=550*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")

    end_time = Time('2011-03-07 16:25:00')
    trange = a.Time(start_time, end_time)

    cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)

    jsoc_email = "ido4@njit.edu"  # Credentials for JSOC database

    query = Fido.search(
        a.Time(start_time - 0*u.h, start_time + 2*u.h),
        a.Wavelength(131*u.angstrom),
        a.Sample(2.5*u.min),
        a.jsoc.Series.aia_lev1_euv_12s,
        a.jsoc.Notify(jsoc_email),
        a.jsoc.Segment.image,
        cutout,
    )
    print(query)

    #%%
    files = Fido.fetch(query)
    files.sort()


def write_slit_profile(Map, line_coords, intensities, timesteps, start_time, timescale=None, synth=False):

    intensity_coords_ = sunpy.map.pixelate_coord_path(Map, line_coords)
    intensity_ = sunpy.map.sample_at_coords(Map, intensity_coords_)
    angular_separation_ = intensity_coords_.separation(intensity_coords_[0]).to(u.arcsec)
    if synth:
        intensities.append(intensity_)
    else:
        intensities.append(intensity_[:242])
    timesteps.append((Map.date - start_time).to_value('s'))

    return angular_separation_
    #%%

def process_aia_maps(obs_data_path, start_time):
    intensities_AIA = []
    timesteps_AIA = []
    maps_list = []
    storage = {}

    if len(os.listdir(obs_data_path)) == 0:  # Check if data is already downloaded
        download_maps()

    mapsequence = sunpy.map.Map(obs_data_path + '/aia.lev1_euv_12s.2011-03-07T*.fits', sequence=True)

    #m_seq_base = sunpy.map.Map([m - m_seq[0].quantity for m in m_seq[1:]], sequence=True)
    m_seq_running = sunpy.map.Map(
        [m - prev_m.quantity for m, prev_m in zip(mapsequence[2:], mapsequence[:-2])],
        sequence=True
    )

    line_coords_ = SkyCoord([-335, -380], [380, 480], unit=(u.arcsec, u.arcsec), frame=mapsequence[0].coordinate_frame)

    # RETURN AIA MAP SLIT PROFILES
    idx = 0
    plt.ioff()
    for map in m_seq_running:
        #fig = plt.figure()
        #ax = fig.add_subplot(projection=map)
        #ax.imshow(map.data,
        #          origin="lower", norm=colors.Normalize(vmin=-200, vmax=200), cmap='Greys_r')
        #plt.savefig('aiamaps/map_'+str(idx)+'.png')
        #idx += 1
        angular_separation_ = write_slit_profile(map, line_coords_, intensities_AIA, timesteps_AIA, start_time)

    #plt.close()
    timesteps = np.transpose(np.array(timesteps_AIA))

    xs = np.append(timesteps - np.diff(timesteps)[0] / 2., timesteps[-1] + np.diff(timesteps)[0] / 2.)[:]
    ys = np.append(angular_separation_ - np.diff(angular_separation_)[0] / 2.,
                   angular_separation_[-1] + np.diff(angular_separation_)[0] / 2.)[:-1]
    X, Y = np.meshgrid(np.array(xs), np.array(ys))
    Z = np.vstack(intensities_AIA).transpose()

    cmap = color_tables.aia_color_table(int(131) * u.angstrom)
    plt.pcolormesh(X, Y, Z, shading='flat', norm=colors.Normalize(vmin=-50, vmax=50), cmap='Greys_r') #cmap=cmap)
    plt.xlabel('timestep, s')
    plt.ylabel('Angular distance, arcsec')
    plt.title('start_time: ' + start_time.value)

    plt.savefig('aia_obs_time_distance_diff.eps')
    plt.close()
    return line_coords_

#%%
def process_synth_maps(ds_dir, start_time, line_coords):
    ts = read_dataset(ds_dir)
    intensities_SYNTH = []
    timesteps_SYNTH = []
    maps_list = []
    storage = {}

    for sto, ds in ts.piter(storage=storage):
        synth_map = gen_map_from_timeseries(ds, start_time, timescale=109.8)
        angular_separation_ = write_slit_profile(synth_map, line_coords, intensities_SYNTH, timesteps_SYNTH,
                                                 start_time, synth=True)

    xs = np.append(timesteps_SYNTH - np.diff(timesteps_SYNTH)[0] / 2.,
                   timesteps_SYNTH[-1] + np.diff(timesteps_SYNTH)[0] / 2.)
    ys = np.append(angular_separation_ - np.diff(angular_separation_)[0] / 2.,
                   angular_separation_[-1] + np.diff(angular_separation_)[0] / 2.)
    X, Y = np.meshgrid(np.array(xs), np.array(ys))
    Z = np.array(intensities_SYNTH)

    cmap = color_tables.aia_color_table(int(131) * u.angstrom)
    plt.pcolormesh(X, Y, Z.transpose(), shading='flat', cmap=cmap)
    plt.xlabel('timestep, s')
    plt.ylabel('Angular distance, arcsec')
    #plt.legend()
    plt.savefig('time_distance.eps')
    plt.close()
    return

#%%

if __name__ == '__main__':

    ds_dir = '/media/ivan/TOSHIBA EXT/subs'
    obs_data_path = '/home/ivan/sunpy/data'  # '/media/ivan/TOSHIBA EXT/aia_img'
    ts = read_dataset(ds_dir)
    start_time = Time('2011-03-07T13:45:26', scale='utc', format='isot')

    line_coords = process_aia_maps(obs_data_path, start_time)
    #%%
    #TODO: FIX upper boundary limit for intensity array (242 max)
    #process_synth_maps(ds_dir, start_time, line_coords)