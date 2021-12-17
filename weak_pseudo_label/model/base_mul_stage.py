import torch.nn as nn
from .backbones import build_backbone
from .cam import build_cam
from .refine import build_refine

class PSModel(nn.Module):
    
    STAGES = ( 'train_cls', 'infer_cam', 'train_refine', 'infer_refine' )

    def __init__(self, backbone, cam_head, refine):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.cam_head = build_cam(cam_head)
        self.refine = build_refine(refine)
        self.cfg = dict(backbone = backbone, cam_head = cam_head, refine = refine)

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        self.backbone.get_parameter_groups(groups)
        self.cam_head.get_parameter_groups(groups)
        self.refine.get_parameter_groups(groups)
        return groups
        
    def train_cls(self, info):
        img = info['img']
        features = self.backbone(img)
        info['features'] = features
        loss = self.cam_head(info, return_loss = True)
        results = dict(loss = loss, log_vars = dict(loss = loss.item()))
        return results

    def infer_cam(self, info):
        img = info['img']
        features = self.backbone(img)
        info['features'] = features
        info['backbone'] = self.backbone
        cam = self.cam_head(info, return_loss = False)
        return cam

    def train_refine(self, info):
        info = dict(**info, backbone = self.backbone, cam_head = self.cam_head)
        return self.refine(info, return_loss = True)

    def infer_refine(self, info):
        info = dict(**info, backbone = self.backbone, cam_head = self.cam_head)
        return self.refine(info, return_loss = False)

    def forward(self, info, stage):
        func = getattr(self, stage)
        return func(info)