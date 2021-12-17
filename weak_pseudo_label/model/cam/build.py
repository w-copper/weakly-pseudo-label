from weak_pseudo_label.utils import Registry

CAM = Registry('cam')

def build_cam(cfg):
    return CAM.build(cfg)