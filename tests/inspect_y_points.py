import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
import sys
import numpy as np

import yt
from yt.visualization.image_writer import write_image
from current_density import _current_density
sys.path.insert(0, '/home/ivan/Study/Astro/solar')

if __name__ == '__main__':

    ds_dir = '/media/ivan/TOSHIBA EXT/subs'

    # sample j_z dataset
    downs_file_path = ds_dir + '/subs_3_flarecs-id_0060.h5'
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
    nslices = 1

    x_coord = 0.
    slc = rad_buffer_obj.slice("x", x_coord)  #, 'current_density')
    slc_frb = slc.to_frb((140.0, "Mm"), 512)

    jz_arr = (slc_frb["gas", "current_density"].d)
    plt.imshow(np.rot90(jz_arr), origin='upper', vmin=8e-7, vmax=1e-5, cmap='RdBu_r')

    plt.colorbar(label="Current density", orientation="vertical")
    plt.title("Current density $j_z$, "+"x="+f'{x_coord:.4f}')
    # plt.show()
    plt.savefig('jz_slices/current_density_slice_x_'+f'{x_coord:.4f}'+'.png')
    plt.close()


    # nslices = 100
    # for i in range(nslices):
    #
    #     # z from -0.25 to 0.25
    #     z_coord = -0.25 + (i * 0.5 / nslices)
    #     slc = rad_buffer_obj.slice("z", z_coord)  #, 'current_density')
    #     slc_frb = slc.to_frb((140.0, "Mm"), 512)
    #     #write_image(np.log10(slc_frb[("gas", "current_density")]), "current_density.png")
    #
    #     # export current_density array frb["gas", "current_density"].d
    #     jz_arr = (slc_frb["gas", "current_density"].d)
    #
    #     plt.imshow(jz_arr, origin='lower', vmin=8e-7, vmax=1e-5, cmap='RdBu_r')
    #     # plt.imshow(jz_arr, origin='lower', norm=colors.LogNorm(vmin=jz_arr.min(), vmax=jz_arr.max()))
    #     plt.colorbar(label="Current density", orientation="vertical")
    #     plt.title("Current density $j_z$, "+"z="+str(z_coord))
    #     # plt.show()
    #     plt.savefig('jz_slices/current_density_slice_z_'+f'{z_coord:.4f}'+'.png')
    #     plt.close()
    #     #slc = yt.SlicePlot(rad_buffer_obj, "z", ("gas", "current_density"))
    #     #slc.save()
    #     #p = yt.ProjectionPlot(rad_buffer_obj, 'z', ("gas", "current_density"))








