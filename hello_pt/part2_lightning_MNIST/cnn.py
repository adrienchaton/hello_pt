

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import torchmetrics

from utils import MNIST_CST
from mnist_data import MNISTdataset


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2, bias=False),
                                   nn.BatchNorm2d(c_out), nn.LeakyReLU())
        # H,W will be halved because of kernel_size=3, padding=1, stride=2
    def forward(self, x):
        return self.block(x)

class CNN(nn.Module):
    def __init__(self, img_hw, n_classes, n_layers, n_channels):
        super(CNN, self).__init__()
        hidden_size = int(img_hw/2**n_layers * n_channels * 2)
        print(f"\nbuilding CNN with hidden size = {hidden_size}")
        layers = [ConvBlock(1, n_channels)]+[ConvBlock(n_channels, n_channels) for _ in range(n_layers-1)]
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_size, n_classes)
        print(f"model with {sum(p.numel() for p in self.parameters() if p.requires_grad)} params.")
    def forward(self, x):
        x = self.cnn(x)
        y = self.proj(x.reshape(x.shape[0], -1))
        return y
    @torch.inference_mode()  # although pytorch_lightning will automatically disable gradients for eval/test/inference
    def predict(self, x):
        return torch.softmax(self.forward(x), dim=-1)
    @torch.inference_mode()
    def classify(self, x):
        return torch.argmax(self.predict(x), dim=-1)

class MNISTmodule(pl.LightningModule):
    def __init__(self, config, seed=1234, num_workers=3):
        super().__init__()
        pl.seed_everything(seed, workers=True)
        self.save_hyperparameters()
        self.cst = MNIST_CST()
        self.create_dataloaders()
        print(f"\ninstantiating MNIST classifier with config. {self.hparams}")  # built-in attribute from lightning
        self.model = CNN(config["img_hw"], self.cst.n_classes, config["n_layers"], config["n_channels"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.cst.n_classes)
        self.metrics_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.cst.n_classes)
    def create_dataloaders(self):
        self.train_data = DataLoader(MNISTdataset(self.cst.path_to_mnist, self.hparams.config["img_hw"], train=True),
                                     batch_size=self.hparams.config["batch_size"], num_workers=self.hparams.num_workers,
                                     shuffle=True, drop_last=True, pin_memory=True)
        self.test_data = DataLoader(MNISTdataset(self.cst.path_to_mnist, self.hparams.config["img_hw"], train=False),
                                    batch_size=self.hparams.config["batch_size"], num_workers=self.hparams.num_workers,
                                    shuffle=False, drop_last=False, pin_memory=False)
    def train_dataloader(self):
        return self.train_data
    def test_dataloader(self):
        return self.test_data
    def forward(self, x):
        return self.model.forward(x)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config["lr"],
                                      weight_decay=self.hparams.config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.config["n_iters"],
                                                               eta_min=self.hparams.config["end_lr"])
        return [optimizer], [scheduler]
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss  # lightning will automatically take care of the optimization step
    def validation_step(self, batch, batch_idx):  # lightning will automatically switch between train/eval modes
        # TODO: stratified split for train/validation and e.g. early stopping callback on validation metrics
        x, y = batch
        preds = self.model.classify(x)
        self.log('val_acc', self.metrics_acc(preds, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_f1', self.metrics_f1(preds, y), on_step=False, on_epoch=True, sync_dist=True)
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model.classify(x)
        self.log('test_acc', self.metrics_acc(preds, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_f1', self.metrics_f1(preds, y), on_step=False, on_epoch=True, sync_dist=True)

if __name__ == "__main__":
    cnn = CNN(32, 10, 4, 64)
    cnn.forward(torch.randn(3, 1, 32, 32))
    # building CNN with hidden size = 256
    # model with 114250 params. (MNIST is ~47M pixels ... most of which are black)
