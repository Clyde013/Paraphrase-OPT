import os

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT, SpecificLayersCheckpoint
from training_datasets.paracombined import ParaCombinedDataModule
from training_datasets.parabank.parabank import ParabankDataModule
from training_datasets.para_nmt.para_nmt import ParaNMTDataModule

import optuna
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback

# initialisation steps
torch.cuda.empty_cache()
AVAIL_GPUS = min(1, torch.cuda.device_count())


def objective(trial: Trial):
    # clear cache so we don't RuntimeError: CUDA out of memory. Tried to allocate 17.00 GB (GPU 0; 39.59 GiB total capacity; 37.53 GiB already allocated; 22.19 MiB free; 37.53 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
    torch.cuda.empty_cache()

    # initialise hyperparameter search
    trial_config = dict()

    # number of embedding tokens
    embedding_n_tokens = trial.suggest_int("embedding_n_tokens", 0, 100)
    trial_config["embedding_n_tokens"] = embedding_n_tokens
    # optimizers
    optimizer_type = trial.suggest_categorical("optimizer_type", ["Adam", "SGD"])
    trial_config["optimizer_type"] = optimizer_type

    # override default params with the hyperparamters being searched for
    wandb.init(project="optimize-popt", entity="clyde013",
               name=f"optimizer_type={optimizer_type}-embedding_n_tokens={embedding_n_tokens}")
    wandb.config.update(trial_config, allow_val_change=True)

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

    early_stopping_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # create wandb logger (obviously)
    wandb_logger = WandbLogger()

    print("TRAINING MODEL")
    trainer = Trainer(max_epochs=wandb.config["max_epochs"], gpus=AVAIL_GPUS,
                      check_val_every_n_epoch=wandb.config["check_val_every_n_epoch"],
                      callbacks=[checkpoint_callback, early_stopping_callback],
                      logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)

    wandb.finish()

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
