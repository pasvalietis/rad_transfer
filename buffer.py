from yt import load
from yt.data_objects.static_output import Dataset
from yt.fields.magnetic_field import get_magnetic_normalization
import numpy as np


class RadDataset(Dataset):
    _index_class = None             # Class used for indexing
    _field_info_class = None        # Class used to set up field information
    _dataset_type = "raddataset"    # Name of type of dataset

    def __init__(self,
                 ytobj,                                 # This dataset will accept yt object as input
                 dataset_type="RadDataset",             # Both      x
                 storage_filename=None,                 # Both      x
                 particle_filename=None,                # Flash
                 parameters=None,                       # Athena
                 units_override=None,                   # Both      x
                 nprocs=1,                              # Athena
                 unit_system="cgs",                     # Both      x
                 default_species_fields=None,           # Both      x
                 magnetic_normalization="gaussian",     # Athena
                 downsample_factor=1,                   # Rad
                 debug=True):  # Athena

        print('\nRadDataset || Debug Mode ' + ("on" if debug else "off"))

        # for debugging, use n=3 subsampled vtk dataset
        if not debug:                                   # Rad
            ytobj = self.downsample(ytobj, downsample_factor)
        else:
            ytobj = load('ss3.h5', hint="YTGridDataset")

        self.fluid_types += ("raddataset",)             # Both      x

        self.nprocs = nprocs                            # Athena
        if parameters is None:
            parameters = {}
        self.specified_parameters = parameters.copy()
        if units_override is None:
            units_override = {}
        self._magnetic_factor = get_magnetic_normalization(magnetic_normalization)

        Dataset.__init__(                               # Both      x
            self,
            ytobj.filename,  # Note - ensure filename is accessible
            dataset_type,
            units_override=units_override,
            unit_system=unit_system,
            default_species_fields=default_species_fields,
        )

        if storage_filename is None:                    # Athena
            storage_filename = self.basename + ".yt"

        self.storage_filename = storage_filename        # Both      x


    @classmethod
    def _is_valid(cls, filename, *args, **kwargs):  # Note - passed for agreement w/ abstract methods
        return True

    def _parse_parameter_file(self):                # Note - passed for agreement w/ abstract methods
        self.domain_left_edge = 0
        self.domain_right_edge = -self.domain_left_edge
        self.domain_width = self.domain_right_edge - self.domain_left_edge
        self.domain_dimensions = np.array([0, 0, 0])

    def _set_code_unit_attributes(self):            # Note - passed for agreement w/ abstract methods
        self.mass_unit = 0
        self.time_unit = 0
        self.length_unit = 0

        if "length_unit" not in self.units_override:
            self.no_cgs_equiv_length = True
        for unit, cgs in [("length", "cm"), ("time", "s"), ("mass", "g")]:
            # We set these to cgs for now, but they may have been overridden
            if getattr(self, unit + "_unit", None) is not None:
                continue
            setattr(self, f"{unit}_unit", self.quan(1.0, cgs))
        self.magnetic_unit = np.sqrt(
            self._magnetic_factor
            * self.mass_unit
            / (self.time_unit**2 * self.length_unit)
        )
        self.magnetic_unit.convert_to_units("gauss")
        self.velocity_unit = self.length_unit / self.time_unit

    def __str__(self):
        return self.basename.rsplit(".", 1)[0]

    def downsample(self, obj, n=1):
        """
        Uses yt fixed resolution function to load a yt object from the target path
        at a specified subsampling rate

        If n is not specified, default n=1 and no down sampling will occur

        :param obj: Path containing full dataset
        :param n: Down-sampling factor
        :return: Down-sampled yt object
        """

        nDim = obj.domain_dimensions / n
        nx = complex(0, nDim[0])
        ny = complex(0, nDim[1])
        nz = complex(0, nDim[2])

        print('\nDownsampling...\n')
        rds = obj.r[::nx, ::ny, ::nz]

        # newName = obj.basename.rsplit(".", 1)[0] + "_ss" + str(n)
        newName = "ss" + str(n)
        save = rds.save_as_dataset(filename=newName, fields=obj.field_list)

        print('\nLoading YTGridDataset...\n')
        nds = load(save, hint="YTGridDataset")

        return nds
