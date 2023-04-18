from yt import load
from yt.frontends.ytdata.data_structures import YTDataset
from yt.frontends.athena.data_structures import AthenaDataset


class customDataset(AthenaDataset):

    def __init__(self, filename):
        super().__init__(filename)


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
