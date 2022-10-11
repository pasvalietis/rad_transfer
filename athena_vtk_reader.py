#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name:
    athena_vtk_reader
Purpose:
    Read vtk data created by Athena code.
Author:
    Chengcai
Update:    
    Created on Wed Jul 19 16:18:15 2017
    2017-07-25
    Bug fixed for computing x, y, z from half-grids.
    2020-04-30
    Remove parameter: bc_symm.
"""
import numpy
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as vn
#import mhdpost_read as mhdrd

#%%
def read(filename, outtype='prim', nscalars=0):
    """
    This routine is used to read vtk file created by Athena v4.2.
    :param filename: the vtk file name with full-path
    :param nscalars: the number of passive scalars. The defaut is 0.
    :param outtype: prim or cons
    :return: return all variables: rho, p, bx, by, bz, vx, vy, vz, scalar, x, y, z, and time.
    """
    reader = vtkStructuredPointsReader()
    # Get all data
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    # Get time 
    header_str = reader.GetHeader()
    temp_str = header_str.split(',')
    time_str = (temp_str[0]).split()
    time = float(time_str[len(time_str) - 1])
    print(" Time = ", time)

    dims = data.GetDimensions()
    vec = list(dims)
    nx = vec[0] - 1
    ny = vec[1] - 1
    nz = vec[2] - 1

    if nz == 0:
        nz = 1

    vec = [nx, ny, nz]
    # print(vec)

    origin = data.GetOrigin()
    x0 = origin[0]
    y0 = origin[1]
    z0 = origin[2]

    spacing = data.GetSpacing()
    dx = spacing[0]
    dy = spacing[1]
    dz = spacing[2]

    x = numpy.arange(nx) * dx + x0 + 0.5 * dx
    y = numpy.arange(ny) * dy + y0 + 0.5 * dy
    z = numpy.arange(nz) * dz + z0 + 0.5 * dz

    # read var 
    data_cells = data.GetCellData()

    # Magnetic field
    b_data = vn.vtk_to_numpy(data.GetCellData().GetArray("cell_centered_B"))
    bx = b_data[:, 0]
    bx = bx.reshape(nz, ny, nx)
    by = b_data[:, 1]
    by = by.reshape(nz, ny, nx)
    bz = b_data[:, 2]
    bz = bz.reshape(nz, ny, nx)

    # Density
    dens_data = vn.vtk_to_numpy(data.GetCellData().GetArray("density"))
    rho = dens_data.reshape(nz, ny, nx)

    # Velocity and Pressure
    if (outtype == 'prim'):
        # Velocity
        v_data = vn.vtk_to_numpy(data.GetCellData().GetArray("velocity"))
        vx = v_data[:, 0]
        vx = vx.reshape(nz, ny, nx)
        vy = v_data[:, 1]
        vy = vy.reshape(nz, ny, nx)
        vz = v_data[:, 2]
        vz = vz.reshape(nz, ny, nx)

        # Pressure
        pres_data = vn.vtk_to_numpy(data.GetCellData().GetArray("pressure"))
        p = pres_data.reshape(nz, ny, nx)

        mx = -1
        my = -1
        mz = -1
        e = -1

        # Passive scalars
        scalar = []
        if nscalars >= 1:
            for iscalar in range(nscalars):
                print('iscalar#',iscalar,' of ', nscalars)
                scalar_str = "specific_scalar[{:d}]".format(iscalar)
                vartemp = vn.vtk_to_numpy(data.GetCellData().GetArray(scalar_str))
                s = vartemp.reshape(nz, ny, nx)
                scalar.append(s)

    else:
        # Momentum
        m_data = vn.vtk_to_numpy(data.GetCellData().GetArray("momentum"))
        mx = m_data[:, 0]
        mx = mx.reshape(nz, ny, nx)
        my = m_data[:, 1]
        my = my.reshape(nz, ny, nx)
        mz = m_data[:, 2]
        mz = mz.reshape(nz, ny, nx)

        # Total_energy
        e_data = vn.vtk_to_numpy(data.GetCellData().GetArray("total_energy"))
        e = e_data.reshape(nz, ny, nx)

        vx = -1
        vy = -1
        vz = -1
        p = -1

        # Passive scalars
        scalar = []
        if nscalars >= 1:
            # scalar = np.ndarray(shape=(nscalars, nz, ny, nx), dtype=np.double, order='C')
            for iscalar in range(nscalars):
                print('*iscalar#', iscalar, ' of ', nscalars)
                #scalar_str = "specific_scalar[{:d}]".format(iscalar)
                scalar_str = "scalar[{:d}]".format(iscalar)
                # print(scalar_str)
                vartemp = vn.vtk_to_numpy(data.GetCellData().GetArray(scalar_str))
                s = vartemp.reshape(nz, ny, nx)
                # scalar[i, 0:nz, 0:ny, 0:nx] = s[0:nz, 0:ny, 0:nx]
                scalar.append(s)

    # Close file
    reader.CloseVTKFile

    # return
    var = {'rho': rho,
           'p': p,
           'e': e,
           'bx': bx,
           'by': by,
           'bz': bz,
           'vx': vx,
           'vy': vy,
           'vz': vz,
           'mx': mx,
           'my': my,
           'mz': mz,
           's': scalar,
           'x': x,
           'y': y,
           'z': z,
           'time': time}
    return var

def read_var(filename, var_name='b1i'):
    """
    This routine is used to read the user defined variable from vtk file 
    created by Athena v4.2.
    :param filename: the vtk file name with full-path
    :param var_name: the variable name.
    :return: var
    """
    reader = vtkStructuredPointsReader()
    # Get all data
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    # Get time 
    header_str = reader.GetHeader()
    temp_str = header_str.split(',')
    time_str = (temp_str[0]).split()
    time = float(time_str[len(time_str) - 1])
    print(" Time = ", time)

    dims = data.GetDimensions()
    vec = list(dims)
    nx = vec[0] - 1
    ny = vec[1] - 1
    nz = vec[2] - 1

    if nz == 0:
        nz = 1

    vec = [nx, ny, nz]

    origin = data.GetOrigin()
    x0 = origin[0]
    y0 = origin[1]
    z0 = origin[2]

    spacing = data.GetSpacing()
    dx = spacing[0]
    dy = spacing[1]
    dz = spacing[2]

    x = numpy.arange(nx) * dx + x0 + 0.5 * dx
    y = numpy.arange(ny) * dy + y0 + 0.5 * dy
    z = numpy.arange(nz) * dz + z0 + 0.5 * dz

    # read var 
    data_cells = data.GetCellData()

    # Density
    var_data = vn.vtk_to_numpy(data.GetCellData().GetArray(var_name))
    var = var_data.reshape(nz, ny, nx)

    # Close file
    reader.CloseVTKFile

    # return
    var_out = {var_name: var,
               'x': x,
               'y': y,
               'z': z,
               'time': time}
    return var_out