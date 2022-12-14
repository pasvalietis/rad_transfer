import numpy
from vtk import vtkStructuredPointsReader, vtkStructuredPointsWriter, vtkImageSincInterpolator, vtkImageReslice # read vtk file
from vtk import vtkImageResize # downscale dataset
from vtk.util import numpy_support as vn
import scipy.optimize as optimize

def vtkZoomImage(image, zoomInFactor):
    """
	Zoom a volume
	"""
    zoomOutFactor = 1.0 / zoomInFactor
    reslice = vtkImageReslice()
    reslice.SetInputConnection(image.GetProducerPort())

    spacing = image.GetSpacing()
    extent = image.GetExtent()
    origin = image.GetOrigin()
    extent = (extent[0], extent[1] / zoomOutFactor, extent[2], extent[3] / zoomOutFactor, extent[4], extent[5])

    spacing = (spacing[0] * zoomOutFactor, spacing[1] * zoomOutFactor, spacing[2])
    reslice.SetOutputSpacing(spacing)
    reslice.SetOutputExtent(extent)
    reslice.SetOutputOrigin(origin)

    # These interpolation settings were found to have the
    # best effect:
    # If we zoom out, no interpolation
    if zoomOutFactor > 1:
        reslice.InterpolateOff()
    else:
        # If we zoom in, use cubic interpolation
        reslice.SetInterpolationModeToCubic()
        reslice.InterpolateOn()
    data = optimize.execute_limited(reslice)
    data.Update()
    return data

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

#Downsample the data
#%% Method1
# resize = vtkImageResize()
# resize.SetResizeMethodToOutputSpacing()
# nfactor = 4 #downcaling factor
# resize.SetOutputSpacing(dx*nfactor, dy*nfactor, dz*nfactor)

#%% Method2
interpolator = vtkImageSincInterpolator()
interpolator.AntialiasingOn()

reslice = vtkImageReslice()
reslice.SetInterpolator(interpolator)
nfactor = 4 #downcaling factor
reslice.SetOutputSpacing(dx*nfactor, dy*nfactor, dz*nfactor)
# reslice.setInputData()
# reslice.update()
# downsampled_cube = reslice.GetOutput()

#Write data
writer = vtkStructuredPointsWriter()
writer.SetInputData(reslice)
writer.SetFileName('datacubes/flarecs_downs4.0035.vtk')
writer.Update()

# Close file
reader.CloseVTKFile