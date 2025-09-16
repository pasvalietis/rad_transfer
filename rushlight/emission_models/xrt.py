import yt
import numpy as np
from yt.data_objects.static_output import Dataset

from scipy import interpolate

from rushlight.config import config

'''
Class to plot synthetic X-ray images as observed from Hinode XRT
'''

class XRTModel:
    """
    A class to model synthetic X-ray intensity as observed by the Hinode XRT instrument.

    This model uses temperature and density fields along with the temperature response
    functions for specific XRT filters to estimate the observed X-ray intensity.
    """
    def __init__(self, temperature_field, density_field, channel):
        """
        Initializes the XRTModel with temperature and density field names and the XRT channel.

        Parameters
        ----------
        temperature_field : str
            The name of the field representing temperature.
        density_field : str
            The name of the field representing density.
        channel : str
            The Hinode XRT channel/filter to use (e.g., 'Ti-poly', 'Al-poly').
        """
        self.temperature_field = temperature_field
        self.density_field = density_field
        self.channel = channel  # 'Ti-poly', 'Al-poly', etc..
        pass

    def setup_model(self, data_source):
        """
        Sets up the model parameters based on the provided data source.

        This method extracts field names, domain information, and initializes
        constants needed for the X-ray intensity calculation.

        Parameters
        ----------
        data_source : Dataset or object with a 'ds' attribute
            The data source containing the temperature and density fields
            and domain information. If not a Dataset object, it is assumed
            to have a 'ds' attribute that is a Dataset.
        """
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
        """
        Processes a data chunk to calculate the synthetic X-ray intensity.

        This method interpolates the temperature response function for the specified
        XRT channel and then uses the density and temperature data from the chunk
        to compute the X-ray intensity.

        Parameters
        ----------
        chunk : yt.data_objects.chunk.DataChunk
            A chunk of data containing the temperature and density fields.

        Returns
        -------
        numpy.ndarray
            An array containing the calculated X-ray intensity for each cell in the chunk.
        """
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
        """
        Adds a derived field for the synthetic X-ray intensity to the provided dataset.

        This method sets up the model using the dataset and defines a function
        that calculates the X-ray intensity using the `process_data` method.
        This derived field can then be accessed and used within the yt analysis framework.

        Parameters
        ----------
        ds : yt.data_objects.dataset.Dataset
            The yt dataset to which the X-ray intensity field will be added.
        """
        self.setup_model(ds)
        ftype = self.ftype
        ei_name = (ftype, f"xrt_intensity")
        ei_dname = rf"I_{{XRT}} (DN/pixel)"

        def _xrt_filter_band(field, data):
            """
            Calculates the synthetic X-ray intensity for a given data object.

            Parameters
            ----------
            field : yt.fields.yt_field.YTField
                The field object being calculated.
            data : yt.data_objects.data_containers.DataContainer
                The data container for which the field is being calculated.

            Returns
            -------
            yt.arraymath.physical_quantity.YTQuantity
                The calculated X-ray intensity field with appropriate units.
            """
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
