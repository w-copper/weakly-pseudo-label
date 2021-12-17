from argparse import ArgumentParser
import pytorch_lightning as pl
import yaml
from weak_pseudo_label.api.train import Trainer
from weak_pseudo_label.api.infer import Inferer
from weak_pseudo_label.dataset import build_datamodule
from weak_pseudo_label.model import PSModel
from argparse import Namespace

def get_parser():
    parser = ArgumentParser(description='Weakly supervised pseudo label generation tool')
    parser.add_argument('cfg', help='config file for the whole stage')
    parser = pl.Trainer.add_argparse_args(parser)
    return parser

def run():
    args = get_parser().parse_args()
    args = vars(args)
    cfg = args.pop('cfg')
    with open(cfg) as f:
        cfg = yaml.safe_load(f)
        # cfg = yaml.load(f, safe_loader)
    cfg = dict(cfg)
    
    args = Namespace(**args)
    stages = cfg.pop('stages')
    trainer = pl.Trainer.from_argparse_args(args)
    for idx, stage in enumerate(stages):
        stage_type = stage['type']
        print(f'[{idx+1}] Stage: {stage_type} ||||| Starting')
        model_cfg = stage['model']
        if model_cfg['type'] != 'pre':
            model_cfg.pop('type')
            model = PSModel(**model_cfg)
        data_cfg = stage['data']
        if data_cfg['type'] != 'pre':
            data = build_datamodule(data_cfg)
        #  trainer
        if 'infer' in stage_type:
            infer_model = Inferer(model, stage_type)
            trainer.test(infer_model, datamodule = data)
        elif 'train' in stage_type:
            optim_cfg = stage['optim']
            train_model = Trainer(model, stage_type, optim_cfg)
            trainer.fit(train_model, datamodule = data)
        print(f'[{idx+1}] Stage: {stage_type} ||||| Ending')
        
if __name__ == '__main__':
    run()