import torch.nn as nn

from weak_pseudo_label.utils import metrics
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
        results = self.cam_head(info, return_loss = True)
        
        if 'log_vars' in results:
            _log_vars = results.pop('log_vars')
            log_vars  = dict(loss = results['loss'].item(), **_log_vars)
        else:
            log_vars  = dict(loss = results['loss'].item())
        if 'ann' in info:
            iou = metrics.batch_iou(results['cam'].argmax(1), info['ann'], info['ignore_index'])
            log_vars['cam_iou'] = iou
        if 'label' in info:
            acc = metrics.cls_acc(results['logit'], info['label'])
            log_vars['cam_acc'] = acc

        mresults = dict(**results, log_vars = log_vars)
        return mresults

    def infer_cam(self, info):
        img = info['img']
        features = self.backbone(img)
        info['features'] = features
        info['backbone'] = self.backbone
        cam = self.cam_head(info, return_loss = False)
        results = dict(cam = cam)
        return results

    def train_refine(self, info):
        info = dict(**info, backbone = self.backbone, cam_head = self.cam_head)
        results = self.refine(info, return_loss = True)
        if 'log_vars' in results:
            _log_vars = results.pop('log_vars')
            log_vars  = dict(loss = results['loss'].item(), **_log_vars)
        else:
            log_vars  = dict(loss = results['loss'].item())
        if 'ann' in info:
            iou = metrics.batch_iou(results['refine'].argmax(1), info['ann'], info['ignore_index'])
            log_vars['cam_iou'] = iou
    
        mresults = dict(**results, log_vars = log_vars)
        return mresults

    def infer_refine(self, info):
        info = dict(**info, backbone = self.backbone, cam_head = self.cam_head)
        refine = self.refine(info, return_loss = False)
        results = dict(refine = refine)
        log_vars = dict()
        if 'ann' in info:
            iou = metrics.batch_iou(results['refine'].argmax(1), info['ann'], info['ignore_index'])
            log_vars['cam_iou'] = iou
        results['log_vars'] = log_vars
        return results

    def forward(self, info, stage):
        func = getattr(self, stage)
        return func(info)