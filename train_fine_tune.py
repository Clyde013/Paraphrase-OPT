import os
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from fine_tuning.fine_tune_opt import FineTuneOPT
from training_datasets.paracombined import ParaCombinedDataModule
from training_datasets.parabank.parabank import ParabankDataModule
from training_datasets.para_nmt.para_nmt import ParaNMTDataModule


if __name__ == "__main__":
    # initialisation steps
    torch.cuda.empty_cache()
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    run = wandb.init(project="fine-tune-opt", entity="clyde013")

    with run:
        datamodule = ParaCombinedDataModule(wandb.config["model_name"], batch_size=wandb.config["batch_size"],
                                            steps_per_epoch=wandb.config["steps_per_epoch"],
                                            datamodules=[ParabankDataModule, ParaNMTDataModule],
                                            probabilities=[0.5, 0.5])
        datamodule.setup()

        if (wandb.config["load_from_checkpoint"] is not None) and (os.path.isfile(wandb.config["load_from_checkpoint"])):
            model = FineTuneOPT.load_from_custom_save(wandb.config["model_name"], wandb.config["load_from_checkpoint"])
        else:
            model = FineTuneOPT(wandb.config["model_name"])

        checkpoint_callback = ModelCheckpoint(dirpath=wandb.config["checkpoint_save_dir"],
                                              save_top_k=2, monitor="val_loss",
                                              filename="fine-tune-opt-epoch={epoch:03d}-val_loss={val_loss:.3f}")

        # create wandb logger (obviously)
        wandb_logger = WandbLogger(checkpoint_callback=False)

        print("TRAINING MODEL")
        trainer = Trainer(max_epochs=wandb.config["max_epochs"], gpus=AVAIL_GPUS,
                          check_val_every_n_epoch=wandb.config["check_val_every_n_epoch"],
                          callbacks=[checkpoint_callback],
                          logger=wandb_logger)
        trainer.fit(model, datamodule=datamodule)

    wandb.finish()

    print("Training complete.")
