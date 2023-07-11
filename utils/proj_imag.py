import yt
import os
import sys
import numpy as np

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from emission_models import uv, xrt
from visualization.colormaps import color_tables

class SyntheticFilterImage:
    """
    Load a yt readable dataset and plot synthetic image having given the instrument name and wavelength
    """

    def __init__(self, dataset, instr, channel, hint=None, units_override=None,
                 view_settings=None, plot_settings=None, **kwargs):
        """
        :param dataset: Path of the downsampled dataset or a dataset itself
        """
        if isinstance(dataset, str):
            self.data = yt.load(dataset, units_override=units_override, hint=hint)
        elif isinstance(dataset, yt.data_objects.static_output.Dataset):
            self.data = dataset

        self.instr = instr
        self.channel = channel
        self.view_settings = {'normal_vector': (0.0, 0.0, 1.0),  # pass vectors as mutable arguments
                              'north_vector': (-0.7, -0.3, 0.0)}
        self.__imag_field = None
        self.image = None

        self.plot_settings = {'resolution': 512,
                              'vmin': 1e-15,
                              'vmax': 1e6,
                              'cmap': 'inferno',
                              'logscale': True,
                              'figpath': './prj_plt.png',
                              'frame': None,
                              'label': None}

    def make_filter_image_field(self, **kwargs):

        cmap = {}
        imaging_model = None
        instr_list = ['xrt', 'aia']

        if self.instr not in instr_list:
            raise ValueError("instr should be in the instrument list: ", instr_list)

        if self.instr == 'xrt':
            imaging_model = xrt.XRTModel("temperature", "density", self.channel)
            cmap['xrt'] = color_tables.xrt_color_table()
        if self.instr == 'aia':
            imaging_model = uv.UVModel("temperature", "density", self.channel)
            try:
                cmap['aia'] = color_tables.aia_color_table(int(self.channel) * u.angstrom)
            except ValueError:
                raise ValueError("AIA wavelength should be one of the following:"
                                 "1600, 1700, 4500, 94, 131, 171, 193, 211, 304, 335.")

        imaging_model.make_intensity_fields(self.data)

        field = str(self.instr) + '_filter_band'
        self.__imag_field = field

        if self.plot_settings:
            self.plot_settings['cmap'] = cmap[self.instr]

    def proj_and_imag(self, plot_settings=None, view_settings=None, **kwargs):

        self.make_filter_image_field()  # Create emission fields

        if plot_settings:
            self.plot_settings.update(plot_settings)

        if view_settings:
            self.view_settings.update(view_settings)

        prji = yt.visualization.volume_rendering.off_axis_projection.off_axis_projection(
            self.data,
            [0.1, 0.5, 0.0],  # center position in code units
            normal_vector=list(self.view_settings['normal_vector']),  # normal vector (z axis)
            width=self.data.domain_width[0].value,  # width in code units
            resolution=self.plot_settings['resolution'],  # image resolution
            item=self.__imag_field,  # respective field that is being projected
            north_vector=list(self.view_settings['north_vector']))

        self.image = np.array(prji)
