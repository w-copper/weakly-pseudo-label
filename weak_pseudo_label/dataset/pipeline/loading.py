
from ..build import PIPELINES
from osgeo import gdal_array
import numpy as np

class LoadDataFromFile(object):

    def __init__(self, key, out, bands = 'all', backend = 'osgeo', tranpose = False) -> None:
        super().__init__()
        self.key = key
        self.out = out
        self.bands = bands
        self.backend = backend
        self.tranpose = tranpose
    
    def osgeo_load(self, fn):
        arr = gdal_array.LoadFile(fn)
        return arr

    def npy_load(self, fn):
        return np.load(fn)

    def __call__(self, data):
        func = getattr(self, self.backend + '_load')
        arr = func(data[self.key])
        if self.tranpose:
            if len(arr.shape) == 2:
                arr = arr[None, :, :]
            else:
                arr = np.transpose(arr, (2, 0, 1))
        if self.bands != 'all':
            arr = arr[self.bands]
        data[self.out] = arr
    
@PIPELINES.register_module()
class LoadImage(LoadDataFromFile):

    def __init__(self, key = 'img_file', out = 'img', bands='all', backend='osgeo', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)

@PIPELINES.register_module()
class LoadCAMTiff(LoadDataFromFile):

    def __init__(self, key = 'cam_file', out = 'cam', bands='all', backend='osgeo', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)

@PIPELINES.register_module()
class LoadCAMNPY(LoadDataFromFile):

    def __init__(self, key = 'cam_file', out = 'cam', bands='all', backend='npy', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)

@PIPELINES.register_module()
class LoadSaliencyMap(LoadDataFromFile):

    def __init__(self, key = 'sal_file', out = 'sal', bands='all', backend='tif', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)