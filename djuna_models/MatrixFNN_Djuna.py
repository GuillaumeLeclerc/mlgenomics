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
import numpy.ma as ma

# modify the code to run on multiple independent inputs 
# the 3D input, and all the 2D inputs, but then compute the loss 

# how do we train the network on multiple independent datasets?: Alternate between the different datasets

class SimpleFNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, lr, *args, **kwargs): # constructing the neural net
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        size = 128
        num_layers = 5
        layers = []
        layers.append(torch.nn.Linear(1030, size)) # applies linear transformation to incomping data
        layers.append(torch.nn.BatchNorm1d(size, affine=False)) # application of batch normalization

        for _ in range(num_layers - 1):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(size, size))
            layers.append(torch.nn.BatchNorm1d(size, affine=False))
            
        layers.append(torch.nn.Linear(size, 3)) # we care about the last layer as the model's projection into n-d space
      
        # add a final layer that implements data-specific scaling factor
        self.layers = torch.nn.Sequential(*layers)

        self.register_parameter('scale_weights',torch.nn.Parameter(torch.ones(11)))  # initiate empty torch
        

    def forward(self, x, ID): # accept tensor of input data and return tensor of output data
        
        q = x.view(x.size(0), -1)
       
        q = self.layers(q) # run the model on the data

        # multiply each col in q by this
        indexed_weights = torch.index_select(self.scale_weights, 0, ID.long())
        return indexed_weights[:,None]*q

    def compute_matrix(self, x, ID): # compute pairwise distances for input matrix 
        x = x[:, None, :] - x[None, :, :]
        x = (x ** 2).mean(2)

        mask = ID[:, None] - ID[None, :]
        mask[abs(mask)>0] = 1
        
        return x * (mask)

    def compute_loss(self, x, y, y_hat, ID):
        true_matrix = self.compute_matrix(y, ID)
        pred_matrix = self.compute_matrix(y_hat, ID)

        loss = (true_matrix - pred_matrix).abs().mean() # compute average difference in pairwise distances between true and predicted --> this is the loss

        return loss

    def configure_optimizers(self):
        print(self.hparams)
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.hparams.lr),
                                     weight_decay=1e-2)
        scheduler = StepLR(optimizer, 100, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx): # batch is the output of the dataloader
        x, y, ID = batch 
    
        y_hat = self(x, ID) # get model output (calls forward)

        loss = self.compute_loss(x, y, y_hat, ID) # compute loss between actual and predicted

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ID = batch
        y_hat = self(x, ID) # compute output on validation set
        loss = self.compute_loss(x, y, y_hat, ID) # get loss 
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)# log and return the loss
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, ID)
        loss = self.compute_loss(x, y, y_hat, ID)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
