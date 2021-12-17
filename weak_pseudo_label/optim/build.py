from weak_pseudo_label.utils import Registry

OPTIM = Registry('optim')

def build_optim(cfg):
    return OPTIM.build(cfg)