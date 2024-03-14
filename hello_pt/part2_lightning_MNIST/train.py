

import os
from pathlib import Path
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from utils import load_config, PrintingCallback, GradientCheckCallback
from cnn import MNISTmodule


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               default=1234, type=int)
    parser.add_argument("--devices",            default=-1)
    parser.add_argument("--exp_path",           default="", type=str)
    parser.add_argument("--config_path",        default="", type=str)
    parser.add_argument("--num_workers",        default=3, type=int)
    parser.add_argument("--freq_val",           default=1., type=float)
    parser.add_argument('--test_on_start',      action='store_true')
    parser.add_argument('--deterministic',      action='store_true')
    parser.add_argument('--mixed_precision',    action='store_true')
    parser.add_argument('--ddp',                action='store_true')
    parser.add_argument('--profiler',           action='store_true')
    args = parser.parse_args()
    return args

def set_defaults(args):
    if args.devices == -1:
        args.devices = torch.cuda.device_count()
    if args.devices <= 1:
        args.ddp = False
    if args.exp_path == "":
        args.exp_path = os.path.join(Path(__file__).parent, "artefacts", "MNIST_CNN")
    if args.config_path == "":
        args.config_path = os.path.join(Path(__file__).parent, "configurations", "mnist_cnn.yaml")
    return args

def main():
    args = configure()
    args = set_defaults(args)
    config = load_config(args.config_path)
    pl_module = MNISTmodule(config, seed=args.seed, num_workers=args.num_workers)
    callbacks = [PrintingCallback(), LearningRateMonitor(logging_interval='epoch'), GradientCheckCallback()]
    trainer = pl.Trainer(max_steps=config["n_iters"], val_check_interval=args.freq_val, devices=args.devices,
                         precision="16-mixed" if args.mixed_precision else "32-true", benchmark=True,
                         default_root_dir=args.exp_path, callbacks=callbacks,
                         deterministic=args.deterministic, accelerator="cpu" if args.devices=="cpu" else "gpu",
                         enable_progress_bar=True, profiler=args.profiler, strategy="ddp" if args.ddp else "auto")
    if args.devices <=1 and args.test_on_start:  # check the "random" performance
        trainer.test(model=pl_module, dataloaders=pl_module.test_data, verbose=True)
    trainer.fit(model=pl_module, train_dataloaders=pl_module.train_data, val_dataloaders=pl_module.test_data)
    if args.devices <=1:  # if distributed training, it is recommended to run testing separately on single GPU
        trainer.test(model=pl_module, dataloaders=pl_module.test_data, verbose=True)

if __name__ == "__main__":
    # can be run with e.g.  (best to debug without distributed!)
    # CUDA_VISIBLE_DEVICES=0 python train.py --test_on_start --freq_val 0.2 --exp_path ./artefacts/MNIST_CNN_1
    # CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --ddp --exp_path ./artefacts/MNIST_CNN_2
    # then check logs with e.g. tensorboard --logdir
    # or use within python with e.g. from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    main()




