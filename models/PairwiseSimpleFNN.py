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

class Transformer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        size = 512
        num_layers = 4
        layers = []
        layers.append(torch.nn.Linear(28, size))
        layers.append(torch.nn.BatchNorm1d(size, affine=False))

        for _ in range(num_layers - 1):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(size, size))
            layers.append(torch.nn.BatchNorm1d(size, affine=False))

        layers.append(torch.nn.Linear(size, 10))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PairwiseSimpleFNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        self.transformer = Transformer()
        
        layers = []
        size = 128
        num_layers = 3
        layers.append(torch.nn.Linear(10, size))
        layers.append(torch.nn.BatchNorm1d(size, affine=False))

        for _ in range(num_layers - 1):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(size, size))
            layers.append(torch.nn.BatchNorm1d(size, affine=False))

        layers.append(torch.nn.Linear(size, 1))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        a = self.transformer(x[:, 0, :])
        b = self.transformer(x[:, 1, :])

        x = torch.abs(a - b)
        x = self.layers(x)
        return x


    def configure_optimizers(self):
        print(self.hparams)
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.hparams.lr))
        scheduler = StepLR(optimizer, 100, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y > 0
        y_hat = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        result.log('train_acc', ((y_hat > 0) == y).float().mean())
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y > 0
        y_hat = self(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        result = pl.EvalResult(loss)
        result.log('val_loss', loss)
        result.log('val_acc', ((y_hat > 0) == y).float().mean())
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result
