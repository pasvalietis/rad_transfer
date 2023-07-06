from abc import ABC

from yt import load
from yt.data_objects.static_output import Dataset
from yt.fields.magnetic_field import get_magnetic_normalization
from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.geometry.grid_geometry_handler import GridIndex
import numpy as np


class RadGrid(AMRGridPatch):
    _id_offset = 0

    def smooth(self, *args, **kwargs):
        pass

    def particle_operation(self, *args, **kwargs):
        pass


class RadHierarchy(GridIndex):
    grid = RadGrid
    _dataset_type = "raddataset"
    _data_file = None

    def _detect_output_fields(self):
        pass

    def _count_grids(self):
        pass

    def _parse_index(self):
        pass

    def _populate_grid_objects(self):
        pass


class RadDataset(Dataset):
    _index_class = RadHierarchy     # Class used for indexing
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
                 debug=True):  # Rad

        # for debugging, use n=3 subsampled vtk dataset
        print('\nRadDataset || Debug Mode ' + ("on" if debug else "off"))
        # boolean debug parameter True or False
        if not debug:
            # Change this flag later. Checks if passed ytobj has already been subsampled
            if ytobj.basename.startswith('ss'):
                self.orig_tempobj = ytobj  # Copy of the initial object before downsampling
                                           # (export parameters, mag. field units etc.)
                self.tempobj = ytobj
            else:
                self.orig_tempobj = ytobj
                self.tempobj = ytobj  # self.downsample(ytobj, downsample_factor)
        else:
            self.tempobj = load('ss3.h5', hint="YTGridDataset")

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
            self.tempobj.filename,  # Note - ensure filename is accessible
            dataset_type,
            units_override=units_override,
            unit_system=ytobj.unit_system,
            default_species_fields=default_species_fields,
        )

        if storage_filename is None:                    # Athena
            storage_filename = self.basename + ".yt"

        self.storage_filename = storage_filename        # Both      x
        if ytobj.filename.endswith(('.h5', '.hdf5')):
            is_h5 = True
        else:
            is_h5 = False

        #del self.tempobj   # I think this wouldn't work this way because tempobj is just a pointer
                            # to the original yt dataset and therefore can not be deleted (Ivan)

    @classmethod
    def _is_valid(cls, filename, *args, **kwargs):  # Note - passed for agreement w/ abstract methods
        return True

    def _parse_parameter_file(self):                # Note - passed for agreement w/ abstract methods
        self.domain_left_edge = self.tempobj.domain_left_edge
        self.domain_right_edge = -self.tempobj.domain_right_edge
        self.domain_width = self.tempobj.domain_width
        self.domain_dimensions = self.tempobj.domain_dimensions
        self.current_time = self.tempobj.current_time

    def _set_code_unit_attributes(self):            # Note - passed for agreement w/ abstract methods
        self.mass_unit = self.tempobj.mass_unit
        self.time_unit = self.tempobj.time_unit
        self.length_unit = self.tempobj.length_unit

        self.no_cgs_equiv_length = self.tempobj.no_cgs_equiv_length
        for unit, cgs in [("length", "cm"), ("time", "s"), ("mass", "g")]:
            # We set these to cgs for now, but they may have been overridden
            if getattr(self, unit + "_unit", None) is not None:
                continue
            setattr(self, f"{unit}_unit", self.quan(1.0, cgs))

        self.magnetic_unit = self.tempobj.magnetic_unit
        # self.magnetic_unit.convert_to_units("gauss")
        self.velocity_unit = self.tempobj.velocity_unit

    def __str__(self):
        return self.basename.rsplit(".", 1)[0]

    # def create_buffer:  # Creating a buffer file in case if original dataset is not a h5 file


'''
Try to write into a downsampled .h5 file only physical fields that we need
i.e. 
# coordinates information (domain_dimensions, domain_left_edge, domain_right_edge)
# ('gas', 'density')
# ('gas', 'temperature')
# Later...
# ('gas', 'magnetic_field')
'''

def downsample(obj, rad_fields=False, n=1):
    """
    Uses yt fixed resolution function to load a yt object from the target path
    at a specified subsampling rate

    If n is not specified, default n=1 and no down sampling will occur

    :param obj: Path containing full dataset
    :param rad_fields: True if fields to be downsampled are
    [('gas','temperature'),('gas','density')] and perhaps [('gas','magnetic_field')]
    :param n: Down-sampling factor
    :return: Down-sampled yt object
    """
    if n != 1:
        nDim = obj.domain_dimensions / n
        nx = complex(0, nDim[0])
        ny = complex(0, nDim[1])
        nz = complex(0, nDim[2])

        print('\nDownsampling...\n')
        rds = obj.r[::nx, ::ny, ::nz]
    else:
        rds = obj.all_data()

    # newName = obj.basename.rsplit(".", 1)[0] + "_ss" + str(n)
    rad_fields_list = [('gas', 'temperature'),
                       ('gas', 'density'),
                       ('gas', 'emission_measure'),
                       ('gas', 'velocity_x'),
                       ('gas', 'velocity_y'),
                       ('gas', 'velocity_z')]

    # rad_fields_list = obj.derived_field_list

    if rad_fields and n == 1:
        newName = "rad_fields_info_" + str(n)
        save = rds.save_as_dataset(filename=newName, fields=rad_fields_list)
        print('\nLoading YTDataContainerDataset...\n')
        nds = load(save, hint="YTDataContainerDataset")
    else:
        newName = "subs_dataset_" + str(n)

        if rad_fields:
            save = rds.save_as_dataset(filename=newName, fields=rad_fields_list)
        else:
            save = rds.save_as_dataset(filename=newName, fields=obj.field_list)
        print('\nLoading YTGridDataset...\n')
        nds = load(save, hint="YTGridDataset")

    return nds
