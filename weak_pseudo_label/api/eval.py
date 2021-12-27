from weak_pseudo_label.utils import Cls_Accuracy, IOUMetric
import os
from osgeo import gdal_array, gdal
import numpy as np
from weak_pseudo_label.utils import imutils
from weak_pseudo_label.dataset import build_dataset

def _crf_with_alpha(orig_img, cam, alpha):
    v = cam
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])
    return crf_score

def run_eval(cfg, pre):

    dataset = build_dataset(cfg['data'])
    color_table = cfg['color_table']
    colors = gdal.ColorTable()
    for i, rgb in enumerate(color_table):
        colors.SetColorEntry(i, tuple(rgb))  #第一个参数是value，第二个是RGB分量
    import tqdm
    for _ in tqdm.tqdm(range(len(dataset))):
        pass
            
