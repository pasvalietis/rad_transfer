import numpy
from vtk import vtkStructuredPointsReader, vtkStructuredPointsWriter, vtkImageSincInterpolator, vtkImageReslice # read vtk file
from vtk import vtkImageResize # downscale dataset
from vtk.util import numpy_support as vn
import scipy.optimize as optimize

filename = 'datacubes/flarecs-id.0035.vtk'

reader = vtkStructuredPointsReader()
# Get all data
reader.SetFileName(filename)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data = reader.GetOutput()

spacing = data.GetSpacing()
dx = spacing[0]
dy = spacing[1]
dz = spacing[2]
print('spacing: ', dx, dy, dz)

# #Downsample the data
#%% Method1
resize = vtkImageResize()
resize.SetInputData(data)
resize.SetResizeMethodToOutputSpacing()
nfactor = 1 #downcaling factor
resize.SetOutputSpacing(dx*nfactor, dy*nfactor, dz*nfactor)

resize.Update()
# #resize.SetSourceData(data)
# #%%
# #resize.Update()
# #%%
# #Write data
# # writer = vtkStructuredPointsWriter()
# # writer.SetInputData(data)
# # writer.SetFileName('datacubes/flarecs_downs4.0035.vtk')
# # writer.Update()
#
# # Close file
# # reader.CloseVTKFile