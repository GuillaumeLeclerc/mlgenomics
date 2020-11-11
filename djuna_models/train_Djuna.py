# %%
import pytorch_lightning as pl
from argparse import ArgumentParser
from uuid import uuid4
from pytorch_lightning.callbacks import LearningRateMonitor

from mlgenomics.djuna_models.SpatialWang2018_multidata import SpatialWang2018DataModule
from mlgenomics.djuna_models.MatrixFNN_Djuna import SimpleFNN

parser = ArgumentParser()
parser = SimpleFNN.add_model_specific_args(parser)

args = parser.parse_args()

dm = SpatialWang2018DataModule()
dm.setup()


# init data
model = SimpleFNN(**vars(args))


lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    val_check_interval=0.1,
    callbacks=[lr_monitor]
)

trainer.fit(model, datamodule=dm)

trainer.test(datamodule=dm)
trainer.save_checkpoint(f"./{uuid4()}.ckpt")

# %%
