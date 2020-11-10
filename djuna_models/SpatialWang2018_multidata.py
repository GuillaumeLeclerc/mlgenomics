from os import path

import numpy as np
import torch as ch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
import scipy.stats as stats


DEFAULT_PATH = '../datasets/sequentially_encoded_spatial_wang_science_2018.npz'
DEFAULT_PATH = path.join(path.dirname(path.realpath(__file__)), DEFAULT_PATH)

TEST_CELLS = 3000
VAL_CELLS = 3000

class SpatialWang2018DataModule(pl.LightningDataModule):

    def __init__(self, data_file: str = DEFAULT_PATH, random_seed=0,
                 batch_size=1024, normalize_coords=True):
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

        data /= ((data ** 2).sum(1) ** 0.5)[:, None] + 1e-5

        if self.normalize_coords:
            mean = coordinates.mean(0)
            maxs = coordinates.max()

            coordinates -= mean[None, :]
            coordinates /= coordinates.abs().max() * 2
        else:
            coordinates = coordinates / coordinates.abs().max()

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
                          pin_memory=True, shuffle=True, num_workers=0)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=False, num_workers=0)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=False, num_workers=0)

    def compute_baseline(self, mode='chance', samples=100, loss=ch.nn.functional.mse_loss):
        l = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))
        test_coords = next(iter(l))[1]
        l = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
        train_coords = next(iter(l))[1]

        results = []
        for _ in range(samples):
            if mode == 'chance':
                random_indices = ch.randint(0, train_coords.shape[0],
                                            (test_coords.shape[0],))
                prediction = ch.index_select(train_coords, 0, random_indices)
            elif mode == 'center':
                prediction = test_coords*0 + train_coords.mean(0)[None, :]

            error = ((prediction - test_coords)**2).mean(0).numpy()
            results.append(error)

        per_dim_errors =  np.mean(results, 0)
        return list(per_dim_errors), np.mean(per_dim_errors)


class PairwiseDataset(Dataset):
    def __init__(self, original_dataset, norm=2, length=1e5, scale=None):
        super().__init__()

        self.original_dataset = original_dataset
        self.length = int(length)
        self.norm = norm
        self.scale = scale

        if scale is not None:
            self.all_distances = np.linspace(0, stats.expon.ppf(0.999, loc=0, scale=scale), 100000)
            self.mapped = stats.norm.ppf(stats.expon.cdf(self.all_distances, 0, self.scale) + 1e-8)

    def __len__(self):
        return self.length

    def __getitem__(self, ix):
        count = len(self.original_dataset)

        try:
            seed = ch.utils.data.get_worker_info().seed
            np.random.seed(seed % int(2**32 - 1))
        except:
            pass

        ix1 = np.random.randint(0, count)
        ix2 = np.random.randint(0, count)

        features_a, coords_a = self.original_dataset[ix1]
        features_b, coords_b = self.original_dataset[ix2]

        distance = (((coords_a - coords_b) ** 2).sum())

        if self.scale is not None:
            pass 
            distance = np.interp(distance, self.all_distances, self.mapped).astype('float32')

        final_features = ch.stack([features_a, features_b])

        return final_features, distance

def compute_params(*args, **kwargs):
    ds = PairwiseDataset(*args, **kwargs)
    sample = next(iter(DataLoader(ds, batch_size=30000,
                                  num_workers=0, shuffle=True)))[1].numpy()

    return stats.expon.fit(sample, floc=0)


class PairWiseSpatialWang2018DataModule(SpatialWang2018DataModule):

    def train_dataloader(self):
        loc, scale = compute_params(self.train_dataset)
        return DataLoader(PairwiseDataset(self.train_dataset, length=1e7, scale=scale),
                          batch_size=self.batch_size,
                          pin_memory=True, shuffle=True, num_workers=40)

    def val_dataloader(self):
        loc, scale = compute_params(self.train_dataset)
        return DataLoader(PairwiseDataset(self.val_dataset, length=1e5, scale=scale),
                          batch_size=self.batch_size,
                          pin_memory=True, shuffle=False, num_workers=40)

    def test_dataloader(self):
        loc, scale = compute_params(self.train_dataset)
        return DataLoader(PairwiseDataset(self.test_dataset, length=1e6, scale=scale),
                          batch_size=self.batch_size,
                          pin_memory=True, shuffle=False, num_workers=40)




if __name__ == '__main__':
    y = SpatialWang2018DataModule()
    y.setup()
    print("Chance baseline:", y.compute_baseline(mode='chance'))
    print("Center baseline:", y.compute_baseline(mode='center'))

    y2 = PairWiseSpatialWang2018DataModule()
    y2.setup()

