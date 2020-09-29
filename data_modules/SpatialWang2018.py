from os import path

import numpy as np
import torch as ch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset


DEFAULT_PATH = '../datasets/sequentially_encoded_spatial_wang_science_2018.npz'
DEFAULT_PATH = path.join(path.dirname(path.realpath(__file__)), DEFAULT_PATH)

TEST_CELLS = 3000
VAL_CELLS = 3000

class SpatialWang2018DataModule(pl.LightningDataModule):

    def __init__(self, data_file: str = DEFAULT_PATH, random_seed=0,
                 batch_size=256, normalize_coords=True):
        super().__init__()
        self.data_file = data_file
        self.random_seed = random_seed
        self.normalize_coords = normalize_coords
        self.batch_size = batch_size
        self.dims = 28

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        content = np.load(self.data_file)['arr_0']

        # Extract the labels as the last 3 columns
        data = ch.from_numpy(content[:, :-3]).float()
        coordinates = ch.from_numpy(content[:, -3:]).float()

        if self.normalize_coords:
            mins = coordinates.min(0)[0]
            maxs = coordinates.max(0)[0]

            coordinates -= mins[None, :]
            coordinates /= maxs[None, :] - mins[None, :]

        full_dataset = TensorDataset(data, coordinates)

        num_samples = len(full_dataset)

        # We want to always left out the same data to make sure it never
        # leaks into our models/experiments
        test_generator = ch.Generator().manual_seed(42)
        # For validation we want to be able to sample different sets to do
        # cross validation for example
        val_generator = ch.Generator().manual_seed(self.random_seed)

        rest, test_dataset = random_split(full_dataset,
                                          [num_samples - TEST_CELLS, TEST_CELLS],
                                          generator=test_generator)

        train_dataset, val_dataset = random_split(rest,
                                                  [len(rest) - VAL_CELLS, VAL_CELLS],
                                                  generator=val_generator)


        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=False)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=False)



y = SpatialWang2018DataModule()
y.setup()
