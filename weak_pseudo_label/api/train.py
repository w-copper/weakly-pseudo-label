import pytorch_lightning as pl
from weak_pseudo_label.model.base_mul_stage import PSModel
from weak_pseudo_label.optim import build_optim
from weak_pseudo_label.utils import save_cam
from weak_pseudo_label.utils import Cls_Accuracy, IOUMetric
from weak_pseudo_label.utils.result import save_output

class Trainer(pl.LightningModule):

    def __init__(self, model:PSModel, stage, optim):
        super(Trainer, self).__init__()
        self.model = model
        assert stage in self.model.STAGES
        self.stage = stage
        self.optim = optim

    # def training_epoch_end(self, outputs) -> None:
    #     return super().training_epoch_end(outputs)

    # def validation_epoch_end(self, outputs) -> None:
    #     return super().validation_epoch_end(outputs)

    # def test_epoch_end(self, outputs) -> None:
    #     return super().validation_epoch_end(outputs)

    # def init_metric(self):
    #     pass

    def training_step(self,  batch, batch_idx, optimizer_idx = 0):
        # if batch_idx == 0:
        #     self.init_metric()
        # batch['cls_metric'] = self.cls_metric
        # batch['iou_metric'] = self.iou_metric
        logs = self.model(batch, self.stage)
        loss = 0
        for k in logs.keys():
            if 'loss' in k:
                loss += logs[k]
        self.log_dict(logs['log_vars'])
        return loss

    def validation_step(self,  batch, batch_idx, optimizer_idx = 0):
        # if batch_idx == 0:
        #     self.init_metric()
        # batch['cls_metric'] = self.cls_metric
        # batch['iou_metric'] = self.iou_metric
        logs = self.model(batch, self.stage)
        loss = 0
        for k in logs.keys():
            if 'loss' in k:
                loss += logs[k]
        self.log_dict(logs['log_vars'])
        return loss
        
    def test_step(self, batch, batch_idx, **kargs ):
        # if batch_idx == 0:
        #     self.init_metric()
        # batch['cls_metric'] = self.cls_metric
        # batch['iou_metric'] = self.iou_metric
        outputs = self.model(batch, self.stage)
        save_output(outputs, batch)
        self.log_dict(outputs['log_vars'])

    def configure_optimizers(self):
        if self.optim is None:
            return
        params = self.model.get_parameter_groups()
        optim = build_optim(dict(params = params, **self.optim))
        return optim