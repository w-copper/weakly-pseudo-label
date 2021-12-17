from weak_pseudo_label.utils import Registry

REFINE = Registry('Refine')

def build_refine(cfg):
    return REFINE.build(cfg)