from torchvision import models

# =========================
# Standard library imports
# =========================
import os
import os.path as osp
import math
import random
import zipfile
from datetime import datetime
from collections import OrderedDict

# =========================
# Third-party libraries
# =========================
import numpy as np

# =========================
# PyTorch and TorchVision
# =========================
import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

# =========================
# PyTorch Lightning
# =========================

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torchmetrics

from torchmetrics.classification import accuracy
from gsn_classification.lightning_modules.MMFN_model import MMFN

class MMFNLitModel(L.LightningModule):
    def __init__(self, num_classes, learning_rate, ratio, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = MMFN(num_classes=num_classes, tabular_dim=4, ratio=ratio)

    def forward(self, img, tab):
        return self.model(img, tab)

    def compute_loss(self, x, y):
        return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        img, tab, y = batch
        outputs = self(img, tab)
        loss = self.compute_loss(outputs,y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y, num_classes = self.num_classes, task="multiclass")
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, self.parameters()),
    lr=self.learning_rate)
        return optimizer