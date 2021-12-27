from weak_pseudo_label.utils.result import calculate_bg_one, calculate_bg_score
from ..build import PIPELINES
import numpy as np
import os

@PIPELINES.register_module()
class CamToLabel:
    def __init__(self, key, out, zero_is_bg = True) -> None:
        self.key = key
        self.out = out
        self.zero_is_bg = zero_is_bg
        pass

    def __call__(self, data:dict) -> dict:
        cam = data[self.key]
        if self.zero_is_bg:
            pred = np.argmax(cam[1:], axis=0)
            # pred = pred + 1
        else:
            pred = np.argmax(cam, axis=0)
        bg = calculate_bg_one(cam, self.zero_is_bg)
        # bg = bg[None, :, :]
        data[self.out] = pred
        data['bg'] = bg
        return data

@PIPELINES.register_module()
class WriteLabelOutput():

    def __init__(self, fn_key, key, dir = None, colors = []) -> None:
        self.dir = dir
        self.key = key
        self.fn_key  = fn_key
        if len(colors) > 0:
            pass
    
    def __call__(self, data) -> dict:
        fn = os.path.basename(data[self.fn_key])
        arr = data[self.key]
        if self.is_label:
            assert len(arr.shape) == 2
            
