from yt import load
from yt.data_objects.static_output import Dataset
from yt.fields.magnetic_field import get_magnetic_normalization


class customDataset(Dataset):
    _index_class = None             # Class used for indexing
    _field_info_class = None        # Class used to set up field information
    _dataset_type = "RadDataset"    # Name of type of dataset

    def __init__(self, ytobj,
                 dataset_type="RadDataset",
                 storage_filename=None,
                 particle_filename=None,
                 parameters=None,
                 units_override=None,
                 nprocs=1,
                 unit_system="cgs",
                 default_species_fields=None,
                 magnetic_normalization="gaussian", ):

        self.fluid_types += ("RadDataset",)
        self.nprocs = nprocs
        if parameters is None:
            parameters = {}
        self.specified_parameters = parameters.copy()
        if units_override is None:
            units_override = {}
        self._magnetic_factor = get_magnetic_normalization(magnetic_normalization)


def dsl(obj, n=1):
    """
    "Downsample Load"
    Uses built-in yt fixed resolution function to load a yt object from the target path
    at a particular subsampling rate

    If n is not specified, default n=1 and no down sampling will occur

    :param obj: Path containing full dataset
    :param n: Down-sampling factor
    :return: Down-sampled yt object
    """

    nDim = obj.domain_dimensions / n
    nx = complex(0, nDim[0])
    ny = complex(0, nDim[1])
    nz = complex(0, nDim[2])

    rds = obj.r[::nx, ::ny, ::nz]

    newName = obj.basename.rsplit(".", 1)[0] + "_ss" + str(n)
    save = rds.save_as_dataset(filename=newName, fields=obj.field_list)

    # load(save) will turn newly downsampled ds into arbitrary grid dataset
    nds = customDataset(load(save))

    return nds
