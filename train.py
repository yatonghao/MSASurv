# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->设置参数
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='STAD/MSASurv.yaml',type=str)
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--fold', default = 5)
    args = parser.parse_args()
    return args

#---->main function
def main(cfg):

    #---->Initialize the seed
    pl.seed_everything(cfg.General.seed)

    #---->加载loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->加载callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->定义Data类
    DataInterface_dict = {
                'seed':cfg.General.seed,
                'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'fold':cfg.Data.fold,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'data_dir': cfg.Data.data_dir,
                'csv_path': cfg.Data.csv_path,
                'label_dir': cfg.Data.label_dir
    }
    dm = DataInterface(**DataInterface_dict)

    #---->定义Model类
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->实例化Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, #直接训练
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        gpus=cfg.General.gpus,
        accumulate_grad_batches=cfg.General.grad_acc,
        check_val_every_n_epoch=1,
    )

    #---->训练或者测试
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'c_index' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    #---->配置参数
    args = make_parse()
 
    #---->读取yaml配置
    cfg = read_yaml(args.config)

    #---->update: 将args的参数保存到cfg中
    cfg.config = args.config
    cfg.General.gpus = args.gpus

    k = args.fold
    for i in [0, 1, 2, 3, 4]:
        cfg.Data.fold = i
        cfg.General.server = 'train'
        main(cfg)

        cfg.General.server = 'test'
        main(cfg)
