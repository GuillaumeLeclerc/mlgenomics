import pytorch_lightning as pl

from mlgenomics.data_modules.SpatialWang2018 import SpatialWang2018DataModule
from mlgenomics.models.SimpleFNN import SimpleFNN

dm = SpatialWang2018DataModule()

# init data
model = SimpleFNN()

trainer = pl.Trainer(
    early_stop_callback=True,
    val_check_interval=0.1
)
trainer.fit(model, datamodule=dm)

trainer.test(datamodule=dm)
