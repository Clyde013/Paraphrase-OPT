import os

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT, SpecificLayersCheckpoint
from training_datasets.paracombined import ParaCombinedDataModule
from training_datasets.parabank.parabank import ParabankDataModule
from training_datasets.para_nmt.para_nmt import ParaNMTDataModule

# initialisation steps
torch.cuda.empty_cache()
AVAIL_GPUS = min(1, torch.cuda.device_count())

"""
# rewrite this dictionary to become raytune's hyperparameter config
raytune_hyperparams = dict(
    dropout=0.5,
    batch_size=100,
    learning_rate=0.001,
)
"""
wandb.init(project="paraphrase-opt", entity="clyde013")  # , config=raytune_hyperparams)

datamodule = ParaCombinedDataModule(wandb.config["model_name"], batch_size=wandb.config["batch_size"],
                                    steps_per_epoch=wandb.config["steps_per_epoch"],
                                    datamodules=[ParabankDataModule, ParaNMTDataModule],
                                    probabilities=[0.5, 0.5])
datamodule.setup()

if (wandb.config["load_from_checkpoint"] is not None) and (os.path.isfile(wandb.config["load_from_checkpoint"])):
    model = ParaphraseOPT.load_from_custom_save(wandb.config["model_name"], wandb.config["load_from_checkpoint"])
else:
    model = ParaphraseOPT(wandb.config["model_name"])

checkpoint_callback = SpecificLayersCheckpoint(
    monitor="val_loss",
    dirpath=wandb.config["checkpoint_save_dir"],
    filename="soft-opt-epoch={epoch:03d}-val_loss={val_loss:.3f}.ckpt",
    every_n_epochs=wandb.config["checkpoint_every_n_epochs"],
    layers_to_save=wandb.config["layers_to_optimize"]
)

# create wandb logger (obviously)
wandb_logger = WandbLogger()

print("TRAINING MODEL")
trainer = Trainer(max_epochs=wandb.config["max_epochs"], gpus=AVAIL_GPUS,
                  val_check_interval=wandb.config["val_check_interval"],
                  callbacks=[checkpoint_callback],
                  logger=wandb_logger, fast_dev_run=True)
trainer.fit(model, datamodule=datamodule)

wandb.finish()