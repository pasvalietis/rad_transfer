import matplotlib

import numpy as np
from matplotlib import pyplot as plt

import yt

ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")

_, c = ds.find_max(("gas", "density"))
proj = ds.proj(("gas", "density"), 0)

width = (10, "kpc")  # we want a 1.5 mpc view
res = [200, 200]  # create an image with 1000x1000 pixels
frb = proj.to_frb(width, res, center=c)

plt.imshow(np.array(frb["gas", "density"]))
plt.savefig("my_perfect_figure.png")
plt.show()
