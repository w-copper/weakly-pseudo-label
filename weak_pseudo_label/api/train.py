import pytorch_lightning as pl
from weak_pseudo_label.model.base_mul_stage import PSModel
from weak_pseudo_label.optim import build_optim
from weak_pseudo_label.utils import save_cam

class Trainer(pl.LightningModule):

    def __init__(self, model:PSModel, stage, optim):
        super(Trainer, self).__init__()
        self.model = model
        assert stage in self.model.STAGES
        self.stage = stage
        self.optim = optim
        
    
    def training_step(self,  batch, batch_idx, optimizer_idx = 0):
        logs = self.model(batch, self.stage)
        loss = 0
        for k in logs.keys():
            if 'loss' in logs:
                loss += logs[k]
        self.log_dict(logs['log_vars'])
        return loss

    def validation_step(self,  batch, batch_idx, optimizer_idx = 0):
        logs = self.model(batch, self.stage)
        loss = 0
        for k in logs.keys():
            if 'loss' in logs:
                loss += logs[k]
        self.log_dict(logs['log_vars'])
        return loss
        
    def test_step(self, batch, batch_idx, **kargs ):
        outputs = self.model(batch, self.stage)
        save_cam(outputs)
        self.log_dict(outputs['log_vars'])

    def configure_optimizers(self):
        params = self.model.get_parameter_groups()
        optim = build_optim(dict(params = params, **self.optim))
        return optim