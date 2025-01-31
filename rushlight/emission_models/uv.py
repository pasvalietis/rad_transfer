import yt
import numpy as np
from pathlib import Path
from yt.data_objects.static_output import Dataset
import math

from scipy import interpolate
from astropy import constants as const
from astropy import units as u
from yt.fields.particle_fields import obtain_relative_velocity_vector
from yt.fields.vector_operations import get_bulk

from rushlight.config import config

class UVModel:
    def __init__(self, temperature_field, density_field, channel):
        self.temperature_field = temperature_field
        self.density_field = density_field
        self.channel = channel  # 'A94', 'A131', 'A171', 'A193', 'A211', 'A335'
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
        trm_path = config.INSTRUMENTS['SDO_AIA_TEMP_RESPONSE']
        aia_trm = np.load(trm_path, allow_pickle=True)

        channels = {'94': 0, '131': 1, '171': 2, '193': 3, '211': 4, '335': 5}
        ch_ = channels.get(str(self.channel))

        aia_trm_interpf = interpolate.interp1d(
            aia_trm.item()['logt'],
            aia_trm.item()['temp_response'][:, ch_],
            fill_value="extrapolate",
            kind='cubic',
        )
        dens = chunk[self.density_field].d
        temp = chunk[self.temperature_field].d

        uvfield = (dens * dens * aia_trm_interpf(np.log10(np.abs(temp))))
        return uvfield

    def make_intensity_fields(self, ds):
        self.setup_model(ds)
        ftype = self.ftype
        ei_name = (ftype, f"uv_intensity")
        ei_dname = rf"I_{{UV}} (DN/pixel)"

        def _aia_filter_band(field, data):
            norm = yt.YTQuantity(1.0, "cm**5/s/(g**2)")
            uvfield = data.ds.arr(
                norm * self.process_data(data),
                "1/(cm*s)"
            )
            return uvfield

        ds.add_field(
            name=("gas", "aia_filter_band"),
            function=_aia_filter_band,
            sampling_type="local",
            units="1/(cm*s)",
            force_override=True,
        )

        
