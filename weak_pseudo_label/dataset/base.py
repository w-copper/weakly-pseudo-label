import torch
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import numpy as np
import os.path as osp

from .build import DATASET
from .pipeline import Compose
 
def load_img_name_list(dataset_path, num_class):
    with open(dataset_path) as f:
        lines = f.readlines()
    img_name_list = []
    img_labels = []
    for line in lines:
        fields = line.strip().split()
        image = fields[0]
        if len(fields) > 1:
            labels = np.zeros((num_class,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
        else:
            labels = np.ones((num_class,), dtype=np.float32)
        img_name_list.append(image)
        img_labels.append(labels)
    
    return img_name_list,img_labels

@DATASET.register_module()
class Dataset(dataset.Dataset):

    def __init__(self, 
                file_list, 
                img_dir,
                num_class,
                img_suffix = '.tif',
                ann_dir = None,
                ann_suffix = '.tif',
                cam_dir = None,
                cam_suffix = '.tif',
                out_dir = None,
                out_suffix = '.tif',
                sal_dir = None,
                sal_suffix = '.tif',
                pipelines = [] ) -> None:
        self.file_list, self.label_list = load_img_name_list(file_list, num_class)
        self.num_class = num_class
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.ann_suffix = ann_suffix
        self.cam_dir = cam_dir
        self.cam_suffix = cam_suffix
        self.out_dir = out_dir
        self.out_suffix = out_suffix
        self.sal_dir = sal_dir
        self.sal_suffix = sal_suffix
        
        self.pipelines = Compose(pipelines)
        self.infos = self.pre_infos()
    
    def pre_infos(self):
        infos = []
        for idx, fn in enumerate(self.file_list):
            infos.append(dict(
                label = self.label_list[idx],
                img_file=osp.join(self.img_dir, fn + self.img_suffix),
                cam_file='No File' if self.cam_dir is None else osp.join(self.cam_dir, fn + self.cam_suffix),
                ann_file='No File' if self.ann_dir is None else osp.join(self.ann_dir, fn + self.ann_suffix),
                sal_file='No File' if self.sal_dir is None else osp.join(self.sal_dir, fn + self.sal_suffix),
                out_file='No File' if self.out_dir is None else osp.join(self.out_dir, fn + self.out_suffix),
            ))
        return infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info = self.infos[index]
        return self.pipelines(info)
