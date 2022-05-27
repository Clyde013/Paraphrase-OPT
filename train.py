import torch
from pytorch_lightning import Trainer

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT
from training_datasets.parabank import ParabankDataModule

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

datamodule = ParabankDataModule("facebook/opt-350m", batch_size=32, steps_per_epoch=1000)
datamodule.setup()

model = ParaphraseOPT()

print("TRAINING MODEL")
trainer = Trainer(max_epochs=20, gpus=AVAIL_GPUS, val_check_interval=1.0)
trainer.fit(model, datamodule=datamodule)

print("VALIDATING MODEL")
trainer.validate(model, datamodule=datamodule)

trainer.save_checkpoint("trained_soft_opt.ckpt")
