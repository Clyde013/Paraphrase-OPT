import torch
from pytorch_lightning import Trainer

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT, SpecificLayersCheckpoint
from training_datasets.parabank import ParabankDataModule

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

datamodule = ParabankDataModule("facebook/opt-350m", batch_size=64, steps_per_epoch=1000)
datamodule.setup()

model = ParaphraseOPT()

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = SpecificLayersCheckpoint(
    monitor="val_loss",
    dirpath="training_checkpoints/",
    filename="soft-opt-epoch={epoch:03d}-val_loss={val_loss:.3f}",
    every_n_epochs=30,
    layers_to_save=[model.model.soft_embedding]
)

print("TRAINING MODEL")
trainer = Trainer(max_epochs=300, gpus=AVAIL_GPUS, val_check_interval=1.0, callbacks=[checkpoint_callback],
                  fast_dev_run=True)
trainer.fit(model, datamodule=datamodule)

"""
print("VALIDATING MODEL")
trainer.validate(model, datamodule=datamodule)
"""

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


default_model = ParaphraseOPT()
compare_models(default_model.model, model.model)
