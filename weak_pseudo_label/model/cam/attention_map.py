
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
            k = 3 if i % 2 == 0 else 1
            layers += [ nn.Conv2d(emd_dim, emd_dim, k, 1, (k)//2),
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
        x = features[-1]
        cam = self.layers(x)
        # if 'ann' in info:
        #     info['cam_acc'].add_batch(cam.argmax(dim = 1),  info['ann'])
        if return_loss:
            label = info['label']
            x = self.avg(cam)
            x = torch.flatten(x, 1, -1)
            loss = F.multilabel_soft_margin_loss(x, label)
            # info['cls_acc'].add_batch(x, loss)
            result = dict(loss = loss, cam = cam, logit = x)
            return result
        if 'label' in info:
            cam = cam * info['label'].unsqueeze(2).unsqueeze(3)
        return cam