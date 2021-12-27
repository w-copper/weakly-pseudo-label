import torch
import torch.nn.functional as F
import numpy as np
from weak_pseudo_label.utils.visualization import max_norm
from osgeo import gdal, gdal_array, gdalconst

def save_output(data, batch):
    if 'cam' in data:
        save_cam(data['cam'], batch['out_file'])
    if 'refine' in data:
        save_cam(data['refine'], batch['out_file'])

def save_arr_to_tif(arr, fn, colors = None, dtype = gdalconst.GDT_Byte):
    driver = gdal.GetDriverByName('GTiff')
    if len(arr.shape) == 2:
        arr = arr[None, :,:]
    C, W, H = arr.shape
    ds = driver.Create(fn, W, H, C, dtype)
    for i in range(C):
        band = ds.GetRasterBand(i + 1)
        gdal_array.BandWriteArray(band, arr[i])
    if colors is not None:
        ds.GetRasterBand(1).SetColorTable(colors)
    del ds

def save_cam(cams, fns):
    # cam = info['cam']
    driver = gdal.GetDriverByName('GTiff')
    cams = cams.detach().cpu()
    for idx, fn in enumerate(fns):
        cam = cams[idx]
        C, W, H = cam.size()
        cam = cam * 255
        cam = cam.numpy().astype(np.uint8)
        ds = driver.Create(fn, W, H, C, gdalconst.GDT_Byte)
        for i in range(C):
            band = ds.GetRasterBand(i + 1)
            gdal_array.BandWriteArray(band, cam[i])

def calculate_bg_one(cam, zero_is_bg = False):
    cam = np.maximum(cam, 0)
    c, w, h  = cam.shape
    if zero_is_bg:
        bg = 1 - np.max(cam, dim = 0)
    else:
        bg = 1 - np.max(cam, dim = 1)
    bg = bg >  0.7
    return bg

 
def calculate_bg_score(cam, zero_is_bg = True):
    cam_d_norm = F.relu(cam.detach())
    n, c, h, w = cam.size()
    # cam_d_norm = max_norm(cam)
    if zero_is_bg:
        cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
        cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
        cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
        return cam_d_norm
    else:
        #  torch.max(cam_d_norm, dim=1)[0]
        cam_max = torch.max(cam_d_norm, dim=1, keepdim=True)[0]        
        cam_d_norm[cam_d_norm < cam_max] = 0

        return cam_d_norm