from typing import List, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

from weak_pseudo_label.utils import calculate_bg_score, max_norm
from .build import REFINE

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

@REFINE.register_module()
class SEAM(nn.Module):
    def __init__(self, img_c = 3, feature1_c = 256, feature2_c = 256, zero_is_bg = False, scale = 0.3):
        super(SEAM, self).__init__()
        self.f8_3 = torch.nn.Conv2d(feature1_c, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(feature2_c, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + img_c, 192, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9]
        self.scale = scale
        self.zero_is_bg = zero_is_bg
    
    def forward_one(self, x, backbone, cam_head):
        N, C, H, W = x.size()
        d = backbone(x)
        cam = cam_head(d)
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d_norm = calculate_bg_score(cam, self.zero_is_bg)
        f8_3 = F.relu(self.f8_3(d[-2].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d[-3].detach()), inplace=True)
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        n,c,h,w = f.size()

        cam_rv = F.interpolate(self.PCM(cam_d_norm, f), (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        return cam, cam_rv
        
    def get_parameter_groups(self, groups:Tuple[List]):
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):
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

    def forward_train(self, info):
        img1 = info['img']
        backbone = info['backbone']
        cam_head = info['cam_head']
        img2 = F.interpolate(img1,scale_factor=self.scale,mode='bilinear',align_corners=True) 
        N,C,H,W = img1.size()
        label = info['label']
        if self.zero_is_bg:
            pass
        else:
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
        label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
        cam1, cam_rv1 = self.forward_one(img1, backbone, cam_head)
        label1 = F.adaptive_avg_pool2d(cam1, (1,1))
        loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
        cam1 = F.interpolate(max_norm(cam1),scale_factor=self.scale,mode='bilinear',align_corners=True)*label
        cam_rv1 = F.interpolate(max_norm(cam_rv1),scale_factor=self.scale,mode='bilinear',align_corners=True)*label

        cam2, cam_rv2 = self.forward_one(img2, backbone, cam_head)
        label2 = F.adaptive_avg_pool2d(cam2, (1,1))
        loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
        cam2 = max_norm(cam2)*label
        cam_rv2 = max_norm(cam_rv2)*label

        loss_cls1 = F.multilabel_soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
        loss_cls2 = F.multilabel_soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])

        ns,cs,hs,ws = cam2.size()
        loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
        #loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
        cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
        cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]
        tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)#*eq_mask
        tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)#*eq_mask
        loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
        loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
        loss_ecr = loss_ecr1 + loss_ecr2

        loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2 
        loss = loss_cls + loss_er + loss_ecr

        return loss

    def forward_val(self, info):
        x = info['img']
        backbone = info['backbone']
        cam_head = info['cam_head']
        _, cam = self.forward_one(x, backbone, cam_head)
        return cam

    def forward(self, info, return_loss = True):
        if return_loss:
            return self.forward_train(info)
        else:
            return self.forward_val(info)
    
    def PCM(self, cam, f):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.f9(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)

        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        
        return cam_rv
