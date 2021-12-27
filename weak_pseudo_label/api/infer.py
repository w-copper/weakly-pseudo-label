import pytorch_lightning as pl
from weak_pseudo_label.utils import save_output

class Inferer(pl.LightningModule):

    def __init__(self, model, stage):
        super(Inferer, self).__init__()
        self.model = model
        assert stage in self.model.STAGES
        self.stage = stage

    def test_step(self, batch, batch_idx, **kargs ):
        outputs = self.model(batch, self.stage)
        save_output(outputs, batch)
        if 'log_vars' in outputs:
            self.log_dict(outputs['log_vars'])