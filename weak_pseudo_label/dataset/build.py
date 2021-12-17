from torch.utils.data.dataloader import DataLoader
from weak_pseudo_label.utils import Registry
from pytorch_lightning.core.datamodule import LightningDataModule
DATASET = Registry('dataset')
PIPELINES = Registry('piplines')

def build_dataset(cfg):
    return DATASET.build(cfg)

def build_datamodule(cfg):
    dm = CustomDataModule(cfg)
    return dm

class CustomDataModule(LightningDataModule):

    def __init__(self, cfg):
        # super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        self.cfg = cfg
        super().__init__()

    def prepare_data(self) -> None:
        return

    def train_dataloader(self):
        assert 'train' in self.cfg
        loader = self.cfg.pop('train_loader')
        dataset = build_dataset(self.cfg['train'])
        return DataLoader(dataset, **loader)

    def val_dataloader(self):
        assert 'val' in self.cfg
        loader = self.cfg.pop('val_loader')
        dataset = build_dataset(self.cfg['val'])
        return DataLoader(dataset, **loader)
    
    def test_dataloader(self):
        assert 'test' in self.cfg
        loader = self.cfg.pop('test_loader')
        dataset = build_dataset(self.cfg['test'])
        return DataLoader(dataset, **loader)