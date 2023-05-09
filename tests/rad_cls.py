import yt
from buffer import RadDataset

ds = yt.load('../datacubes/ShockCloud/id1/Cloud-id1.0050.vtk', hint='AthenaDataset')
#%%
radcube = RadDataset(ds, debug=False, downsample_factor=5)

