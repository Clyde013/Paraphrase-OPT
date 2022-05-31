import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT, SpecificLayersCheckpoint
from training_datasets.parabank.parabank import ParabankDataModule

# initialisation steps
torch.cuda.empty_cache()
AVAIL_GPUS = min(1, torch.cuda.device_count())
wandb.init(project="paraphrase-opt", entity="clyde013")
model_name = "facebook/opt-1.3b"

datamodule = ParabankDataModule(model_name, batch_size=128, steps_per_epoch=5000)
datamodule.setup()

model = ParaphraseOPT(model_name)

# saves a file like: training_checkpoints/soft-opt-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = SpecificLayersCheckpoint(
    monitor="val_loss",
    dirpath="training_checkpoints/30-05-2022-1.3b/",
    filename="soft-opt-epoch={epoch:03d}-val_loss={val_loss:.3f}.ckpt",
    every_n_epochs=30,
    layers_to_save={"soft_embedding": model.model.soft_embedding}
)

# create wandb logger (obviously)
wandb_logger = WandbLogger()

print("TRAINING MODEL")
trainer = Trainer(max_epochs=300, gpus=AVAIL_GPUS, val_check_interval=1.0, callbacks=[checkpoint_callback],
                  logger=wandb_logger)
trainer.fit(model, datamodule=datamodule)

wandb.finish()

print("--------- MODEL COMPARISON -----------")


# thanks https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


default_model = ParaphraseOPT(model_name)
compare_models(default_model.model, model.model)
