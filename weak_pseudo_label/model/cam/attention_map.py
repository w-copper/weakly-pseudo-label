
from typing import List, Tuple
import torch
from .build import CAM
import torch.nn as nn
import torch.nn.functional as F
@CAM.register_module()
class AttentionMap(nn.Module):

    def __init__(self, num_class, inc, emd_dim = 512, blocks = 5):
        super().__init__()
        layers = [
            nn.Conv2d(inc, emd_dim, 1),
            nn.ReLU(True)
        ]
        for i in range(blocks - 1):
            k = 1 if i % 2 == 0 else 3
            layers += [ nn.Conv2d(emd_dim, emd_dim, k, k//2),
            nn.ReLU(True) ]
        layers.append(nn.Conv2d(emd_dim, num_class, 1))
        self.layers = nn.Sequential(
            *layers
        )
        self.from_scratch_layers = layers
        self.avg = nn.AdaptiveAvgPool2d((1,1))

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

    def forward(self, info, return_loss = True):
        features = info['features']
        label = info['label']
        x = features[-1]
        x = self.layers(x)
        # features[-1] = x
        if return_loss:
            x = self.avg(x)
            x = torch.flatten(x, 1, -1)
            return F.multilabel_soft_margin_loss(x, label)
        return x