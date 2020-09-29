import os
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

class SimpleFNN(pl.LightningModule):

    def __init__(self, constant=False):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28, 512)
        self.bn1 = torch.nn.BatchNorm1d(512, affine=False)
        self.layer_2 = torch.nn.Linear(512, 512)
        self.bn2 = torch.nn.BatchNorm1d(512, affine=False)
        self.layer_3 = torch.nn.Linear(512, 512)
        self.bn3 = torch.nn.BatchNorm1d(512, affine=False)
        self.layer_4 = torch.nn.Linear(512, 3)
        self.constant = constant

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = self.bn3(x)
        x = F.relu(x)
        if self.constant:
            x *= 0
        x = self.layer_4(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result
