import torch.nn as nn
from .build import CAM

@CAM.register_module()
class NoCam(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kargs):
        return