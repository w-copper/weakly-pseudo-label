from weak_pseudo_label.utils import Registry

BACKBONE = Registry('backbone')

def build_backbone(cfg):
    return BACKBONE.build(cfg)