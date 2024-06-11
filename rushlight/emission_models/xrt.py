import yt
import numpy as np
from yt.data_objects.static_output import Dataset

from scipy import interpolate

from rushlight.config import config

'''
Class to plot synthetic X-ray images as observed from Hinode XRT
'''

class XRTModel:
    def __init__(self, temperature_field, density_field, channel):
        self.temperature_field = temperature_field
        self.density_field = density_field
        self.channel = channel  # 'Ti-poly', 'Al-poly', etc..
        pass

    def setup_model(self, data_source):
        if isinstance(data_source, Dataset):
            ds = data_source
        else:
            ds = data_source.ds
        self.temperature_field = ds._get_field_info(self.temperature_field).name
        self.density_field = ds._get_field_info(self.density_field).name
        self.ftype = self.temperature_field[0]
        self.left_edge = ds.domain_left_edge
        self.right_edge = ds.domain_right_edge
        self.L_0 = 1.5e10
        self.domain_dimensions = ds.domain_dimensions

    def process_data(self, chunk):
        trm_path = config.INSTRUMENTS['HINODE_XRT_TEMP_RESPONSE']
        xrt_trm = np.load(trm_path, allow_pickle=True)

        channel = self.channel

        xrt_trm_interpf = interpolate.interp1d(
            xrt_trm.item()['temps'],
            xrt_trm.item()[channel],
            fill_value="extrapolate",
            kind='cubic',
        )

        dens = chunk[self.density_field].d
        temp = chunk[self.temperature_field].d

        uvfield = (dens * dens * xrt_trm_interpf(temp))
        return uvfield

    def make_intensity_fields(self, ds):
        self.setup_model(ds)
        ftype = self.ftype
        ei_name = (ftype, f"xrt_intensity")
        ei_dname = rf"I_{{XRT}} (DN/pixel)"

        def _xrt_filter_band(field, data):
            norm = yt.YTQuantity(1.0, "cm**5/s/(g**2)")
            xrtfield = data.ds.arr(
                norm * self.process_data(data),
                "1/(cm*s)"
            )
            return xrtfield

        ds.add_field(
            name=("gas", "xrt_filter_band"),
            function=_xrt_filter_band,
            sampling_type="local",
            units="1/(cm*s)",
            force_override=True,
        )


