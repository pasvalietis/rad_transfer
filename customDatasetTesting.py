import yt
from yt.frontends.athena.data_structures import AthenaDataset


# ds = yt.load("C:/Users/sabas/Documents/NJIT/CSTR/flarecs-id.0035.vtk")
#
# n = 3
# nDim = ds.domain_dimensions / n
# nx = complex(0, nDim[0])
# ny = complex(0, nDim[1])
# nz = complex(0, nDim[2])
# rds = ds.r[::nx, ::ny, ::nz]
#
# save = rds.save_as_dataset(filename='TestDs', fields=ds.field_list)

class customDataset(AthenaDataset):
    def __init__(self, filename):
        super().__init__(filename)


nds = customDataset('TestDs.h5')
