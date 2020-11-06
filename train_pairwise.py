import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import GpuUsageLogger, LearningRateMonitor

from mlgenomics.data_modules.SpatialWang2018 import PairWiseSpatialWang2018DataModule
from mlgenomics.models.PairwiseSimpleFNN import PairwiseSimpleFNN

parser = ArgumentParser()
parser = PairwiseSimpleFNN.add_model_specific_args(parser)

args = parser.parse_args()

dm = PairWiseSpatialWang2018DataModule(normalize_coords=False)
dm.setup()


# init data
model = PairwiseSimpleFNN(**vars(args))


lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    val_check_interval=1.0,
    gpus=1,
    callbacks=[lr_monitor]
)

trainer.fit(model, datamodule=dm)

trainer.test(datamodule=dm)
