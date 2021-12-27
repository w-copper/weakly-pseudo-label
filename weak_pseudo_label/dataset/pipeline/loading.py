
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
        return data
@PIPELINES.register_module()
class LoadAnn(LoadDataFromFile):
    def __init__(self, zero_to_ignore = True, ignore_index = 255, key = 'ann_file', out = 'ann', bands='all', backend='osgeo', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)
        self.zero_to_ignore = zero_to_ignore
        self.ignore_index = ignore_index
    
    def __call__(self, data):
        data = super().__call__(data)
        ann = data[self.out]
        if self.zero_to_ignore:
            ann[ann == 0] = self.ignore_index
            ann = ann - 1
            ann[ann == self.ignore_index - 1] = self.ignore_index
        # ann = ann.long()
        data[self.out] = ann
        if self.ignore_index is not None:
            data['ignore_index'] = self.ignore_index
        return data

@PIPELINES.register_module()
class LoadImage(LoadDataFromFile):

    def __init__(self, key = 'img_file', out = 'img', bands='all', backend='osgeo', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)

@PIPELINES.register_module()
class LoadCAMTiff(LoadDataFromFile):

    def __init__(self, normalize = True, key = 'cam_file', out = 'cam', bands='all', backend='osgeo', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)
        self.normalize = normalize

    def __call__(self, data):
        data = super().__call__(data)
        if self.normalize:
            data[self.out] = data[self.out] / 255.0
        return data

@PIPELINES.register_module()
class LoadCAMNPY(LoadDataFromFile):

    def __init__(self, normalize = True, key = 'cam_file', out = 'cam', bands='all', backend='npy', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)
        self.normalize = normalize
    def __call__(self, data):
        data = super().__call__(data)
        if self.normalize:
            data[self.out] = data[self.out].type_as(np.float32) / 255.0
        return data

@PIPELINES.register_module()
class LoadSaliencyMap(LoadDataFromFile):

    def __init__(self, t = 50, key = 'sal_file', out = 'sal', bands='all', backend='tif', tranpose=False) -> None:
        super().__init__(key, out, bands=bands, backend=backend, tranpose=tranpose)
        self.t = t
    
    def __call__(self, data):
        data = super().__call__(data)
        if self.t is not None:
            data[self.out] = data[self.out] > self.t
            data[self.out] = np.asarray(data[self.out], np.uint8)
        return data