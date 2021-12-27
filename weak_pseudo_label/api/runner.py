from argparse import ArgumentParser
from datetime import datetime
import os
import sys
from osgeo import gdal_array
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch
import yaml
from weak_pseudo_label.api.train import Trainer
from weak_pseudo_label.dataset import build_datamodule, build_dataset
from weak_pseudo_label.dataset.pipeline.compose import Compose
from weak_pseudo_label.model import PSModel
import inspect
import copy

from weak_pseudo_label.utils.metrics import IOUMetric

def get_parser():
    parser = ArgumentParser(description='Weakly supervised pseudo label generation tool')
    parser.add_argument('cfg', help='config file for the whole stage')
    parser = pl.Trainer.add_argparse_args(parser)
    return parser

def get_trainer_with_cfg(args, cfg, stage_type, **kwargs):
    args = copy.deepcopy(args)
    for key in cfg:
        args[key] = cfg[key]
    valid_kwargs = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)
    trainer_kwargs.update(**kwargs)
    logger = get_logger(args, stage_type)
    trainer_kwargs['logger'] = logger

    if 'infer' in stage_type:
        trainer_kwargs['max_epochs'] = 1
        trainer_kwargs['max_steps'] = None

    return pl.Trainer(**trainer_kwargs)

def get_logger(args, stage, version = None):
    if version is None:
        version = str(int(datetime.now().timestamp() * 1000))
    tblogger = TensorBoardLogger(args['default_root_dir'], name = stage + '_' + version)
    csvlogger = CSVLogger(args['default_root_dir'], name = stage + '_' + version)
    return [tblogger, csvlogger]

def load_ckpt(ckpt_file):
    assert os.path.isfile(ckpt_file) and os.path.exists(ckpt_file)
    ckpt = torch.load(ckpt_file)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    return ckpt

def run_train(cfg, pre):
    model_cfg = cfg['model']
    if model_cfg['type'] != 'pre':
        model_cfg.pop('type')
        model = PSModel(**model_cfg)
    else:
        model = pre['model']
    data_cfg = cfg['data']
    if data_cfg['type'] != 'pre':
        data = build_datamodule(data_cfg)
    else:
        data = pre['data']
    if 'skip' in cfg and cfg['skip']:
        return dict(model = model, data = data, cfg = cfg)
    
    trainer = get_trainer_with_cfg(cfg['args'], cfg['runner'], cfg['type'])
    optim_cfg = cfg['optim']
    train_model = Trainer(model, cfg['type'], optim_cfg)
    if 'ckpt' in cfg:
        ckpt = load_ckpt(cfg['ckpt'])
        train_model.load_state_dict(ckpt)
    
    trainer.fit(train_model, datamodule = data)

    return dict(model = model, data = data, cfg = cfg)

def run_infer(cfg, pre):
    model_cfg = cfg['model']
    if model_cfg['type'] != 'pre':
        model_cfg.pop('type')
        model = PSModel(**model_cfg)
    else:
        model = pre['model']
    data_cfg = cfg['data']
    dataset = build_dataset(data_cfg)
    post_process = Compose(cfg['post'])
    train_model = Trainer(model, cfg['type'], None)
    if 'ckpt' in cfg:
        ckpt = load_ckpt(cfg['ckpt'])
        train_model.load_state_dict(ckpt)
    pred_keys = cfg['pred']
    metrics = dict()
    for key in pred_keys:
        metrics[key] = IOUMetric(cfg['num_class'])
        metrics[key + '_nobg'] = IOUMetric(cfg['num_class'])
        
    import tqdm
    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        output = model(data)
        for key in data:
            output[key] = data[key]
        output = post_process(output)

        for key in pred_keys:
            pred = data[key]
            pred_bg = data[key + '_bg']
            metrics[key].add_batch(pred, data['label'])
            pred[pred_bg == 1] = 255
            metrics[key + '_nobg'].add_batch(pred, data['label'])

    # return dict(model = model, data = data, cfg = cfg)

def run_eval(cfg, pre):
    '''
    CAM or Refine CAM to Label
    Save Label
    Label with/without BG
    Metric
    '''
    dataset = build_dataset(cfg['data'])
    pred_keys = cfg['pred']
    metrics = dict()
    for key in pred_keys:
        metrics[key] = IOUMetric(cfg['num_class'])
        metrics[key + '_nobg'] = IOUMetric(cfg['num_class'])
    import tqdm
    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        if data.get('label', None) is None:
            continue
        for key in pred_keys:
            pred = data[key]
            pred_bg = data[key + '_bg']
            metrics[key].add_batch(pred, data['label'])
            pred[pred_bg == 1] = 255
            metrics[key + '_nobg'].add_batch(pred, data['label'])
    


def run():
    args = get_parser().parse_args()
    args = vars(args)
    cfg = args.pop('cfg')
    with open(cfg) as f:
        cfg = yaml.safe_load(f)
        # cfg = yaml.load(f, safe_loader)
    cfg = dict(cfg)
    stages = cfg.pop('stages')
    pre = dict(cfg = cfg)
    for idx, stage in enumerate(stages):
        stage_type = stage['type']
        print(f'[{idx+1}] Stage: {stage_type} ||||| Starting')
        if 'train' in stage_type:
            stage['args'] = args
            pre = run_train(stage, pre)
        elif 'infer' in stage_type:
            stage['args'] = args
            pre = run_infer(stage, pre)
        elif 'eval' in stage_type:
            stage['args'] = args
            pre = run_eval(stage, pre)
        print(f'[{idx+1}] Stage: {stage_type} ||||| Ending')
        
if __name__ == '__main__':
    run()