import torch
import torch.nn.functional as F

from weak_pseudo_label.utils.visualization import max_norm

def save_cam(info):
    cam = info['cam']
    
def calculate_bg_score(cam, zero_is_bg = True):
    cam = cam.detach()
    n, c, h, w = cam.size()
    cam_d_norm = max_norm(cam)
    if zero_is_bg:
        cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
        cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
        cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
        return cam_d_norm
    else:
        #  torch.max(cam_d_norm, dim=1)[0]
        cam_max = torch.max(cam_d_norm, dim=1, keepdim=True)[0]
        bg = 1 - cam_max
        cam_d_norm[cam_d_norm < cam_max] = 0
        cam_d_norm = torch.cat([bg, cam_d_norm], dim=1)
        return cam_d_norm