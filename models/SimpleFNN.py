import os
import torch
from argparse import ArgumentParser
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

class SimpleFNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        size = 2048
        num_layers = 10
        layers = []
        layers.append(torch.nn.Linear(28, size))
        layers.append(torch.nn.BatchNorm1d(size, affine=False))

        for _ in range(num_layers - 1):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(size, size))
            layers.append(torch.nn.BatchNorm1d(size, affine=False))

        layers.append(torch.nn.Linear(size, 3))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
        return x

    def configure_optimizers(self):
        print(self.hparams)
        optimizer = torch.optim.SGD(self.parameters(), lr=(self.hparams.lr), momentum=0.9)
        scheduler = StepLR(optimizer, 100, gamma=0.1)
        return [optimizer], [scheduler]

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
        result.log('val_x_errors', F.mse_loss(y_hat[:, 0], y[:, 0]))
        result.log('val_y_errors', F.mse_loss(y_hat[:, 1], y[:, 1]))
        result.log('val_z_errors', F.mse_loss(y_hat[:, 2], y[:, 2]))
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result
