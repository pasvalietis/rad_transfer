# 
# Local functions:
#
import numpy as np
from numba import njit

@njit(parallel=True)
def func_interpol_3d(xgrid, ygrid, zgrid, arrgrid, points, npoints):
    arrout = np.zeros(npoints)
    xs = xgrid[0]
    dx = xgrid[1] - xs
    nx = len(xgrid)
    ys = ygrid[0]
    dy = ygrid[1] - ys
    ny = len(ygrid)
    zs = zgrid[0]
    dz = zgrid[1] - zs
    nz = len(zgrid)

    for ipt in range(npoints):
        z0 = points[0,ipt]
        y0 = points[1,ipt]
        x0 = points[2,ipt]
    
        izc = int((z0 - zs)/dz)
        if izc < 0:
            izc = 0
        if izc >= nz-1:
            izc = nz-2
        
        iyc = int((y0 - ys)/dy)
        if iyc < 0:
            iyc = 0
        if iyc >= ny-1:
            iyc = ny-2

        ixc = int((x0 - xs)/dx)
        if ixc < 0:
            ixc = 0
        if ixc >= nx-1:
            ixc = nx-2

        # Get four points at the z0 plane
        z1 = zgrid[izc]
        z2 = zgrid[izc+1]
        v00 = func_interpol_1d(z1, z2, arrgrid[izc, iyc, ixc], arrgrid[izc+1, iyc, ixc], z0)
        v01 = func_interpol_1d(z1, z2, arrgrid[izc, iyc, ixc+1], arrgrid[izc+1, iyc, ixc+1], z0)
        v10 = func_interpol_1d(z1, z2, arrgrid[izc, iyc+1, ixc], arrgrid[izc+1, iyc+1, ixc], z0)
        v11 = func_interpol_1d(z1, z2, arrgrid[izc, iyc+1, ixc+1], arrgrid[izc+1, iyc+1, ixc+1], z0)

        # Get two points at z0, and y0
        y1 = ygrid[iyc]
        y2 = ygrid[iyc+1]
        v00_0 = func_interpol_1d(y1, y2, v00, v10, y0)
        v00_1 = func_interpol_1d(y1, y2, v01, v11, y0)

        # Get value at z0, y0, and x0
        x1 = xgrid[ixc]
        x2 = xgrid[ixc+1]
        vc = func_interpol_1d(x1, x2, v00_0, v00_1, x0)

        arrout[ipt] = vc
    return arrout

@njit()
def func_interpol_1d(x1, x2, v1, v2, xc):
    if (xc <= x1):
        yc = v1
    elif (xc >= x2):
        yc = v2
    else:
        k = (v2-v1)/(x2-x1)
        yc = (xc-x1)*k + v1
    return yc