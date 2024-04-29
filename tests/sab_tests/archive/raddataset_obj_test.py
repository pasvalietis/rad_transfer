from yt import load
from buffer import RadDataset

path1 = "C:/Users/sabas/Documents/NJIT/CSTR/flarecs-id.0035.vtk"
path2 = "C:/Users/sabas/Documents/GitHub/pyxsim_mod/pyxsim/buffer/flarecs-id.0035_ss2.h5"
path3 = "ss4.h5"

debug = True

orig_ds = None if debug else load(path1, hint="AthenaDataset")
ds = RadDataset(orig_ds, downsample_factor=1.2, debug=debug)

print(ds)
