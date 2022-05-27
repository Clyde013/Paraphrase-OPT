import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT
from training_datasets.parabank import ParabankDataModule

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

datamodule = ParabankDataModule("facebook/opt-350m", batch_size=64, steps_per_epoch=1000)
datamodule.setup()

model = ParaphraseOPT()

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="training_checkpoints/",
    filename="soft-opt-{epoch:03d}-{val_loss:.3f}",
    mode="min",
    every_n_epochs=30,
    save_top_k=-1
)

print("TRAINING MODEL")
trainer = Trainer(max_epochs=300, gpus=AVAIL_GPUS, val_check_interval=1.0, callbacks=[checkpoint_callback], fast_dev_run=True)
trainer.fit(model, datamodule=datamodule)

"""
print("VALIDATING MODEL")
trainer.validate(model, datamodule=datamodule)
"""
