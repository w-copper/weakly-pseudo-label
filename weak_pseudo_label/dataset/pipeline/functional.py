import numpy as np
import cv2 as cv

def apply_multi_band(func, arr, args, kargs):
    results = []
    for i in range(arr.shape[0]):
        r = func(arr[i], *args, **kargs)
        results.append(r)
    return results

def multi_band(func):

    def _f(*args, **kargs):
        if 'channel_last' in kargs:
            channel_last = kargs['channel_last']
        else:
            channel_last = False
        arr = args[0]
        if len(arr.shape) == 3:
            if channel_last:
                arr = np.transpose(arr, (2, 0, 1))
            result = apply_multi_band(func, arr, args[1:], kargs)
            result = np.stack(result, 0)
            if channel_last:
                result = np.transpose(result, (1, 2, 0))
            return result
        else:
            return func(*args, **kargs)
    return _f

@multi_band
def resize(arr, size = None, scale = None, channel_last = False, interpolation = cv.INTER_LINEAR ):
    if size is None and scale is None:
        return arr
    assert len(arr.shape) == 2
    ow, oh = arr.shape
    # print(size)
    if size is None:
        size = (int(ow * scale[0]), int(oh * scale[1]))
    if size[0] == ow and size[1] == oh:
        return arr
    arr = cv.resize(arr, size, interpolation = interpolation )
    return arr    

@multi_band
def crop(arr, x, y, w, h, channel_last = False):
    assert len(arr.shape) == 2
    ow, oh = arr.shape
    assert x + w <= ow and y + h <= oh
    narr = arr[x:x+w, y:y+h]
    return narr

@multi_band
def flip(arr, h_v, channel_last = False):
    assert len(arr.shape) == 2
    assert h_v in (0, 1)
    arr = np.flip(arr, axis=h_v)
    return arr

@multi_band
def rot90(arr, k, r = False, channel_last = False):
    assert len(arr.shape) == 2
    if r:
        k = -k
    arr = np.rot90(arr, k, (0, 1))
    return arr
