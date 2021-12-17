from typing import List, Tuple
import torch.nn as nn
from torchvision.models import vgg

from . import utils
from .build import BACKBONE


def split_vgg(features):
    stages = []
    pre = 0
    curr = 0
    for idx, layer in enumerate(features):
        if isinstance(layer, nn.MaxPool2d):
            curr = idx
            stages.append(features[pre:curr+1])
            pre = curr
    return stages


@BACKBONE.register_module()
class VGG(nn.Module):

    _11 = '11'
    _13 = '13'
    _16 = '16'
    _19 = '19'

    def __init__(self,
                 cfg=_11,
                 pretrain=False,
                 bn=False,
                 replace_max_pool=True,
                 remove_last_max_pool=True,
                 ):
        assert cfg in [VGG._11, VGG._13, VGG._16, VGG._19]
        name = 'vgg%s' % cfg
        if bn:
            name = name + '_bn'
        super(VGG, self).__init__()
        _vgg = getattr(vgg, name)(pretrain)
        # _vgg = vgg.vgg13(pretrain)
        if replace_max_pool:
            utils.replace_max_pool(_vgg.features, 4)
        stages = split_vgg(_vgg.features)
        if remove_last_max_pool:
            stages[-1][-1] = nn.Identity()
        self.stages = nn.ModuleList()
        self.from_scratch_layers = []
        for s in stages:
            self.stages.append(s)
        # self.features = _vgg.features

    def get_parameter_groups(self, groups:Tuple[List]):
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.GroupNorm)):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups
    def forward(self, x):
        out_features = []
        for s in self.stages:
            x = s(x)
            out_features.append(x)
        return out_features
