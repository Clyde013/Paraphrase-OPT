import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import OPTForCausalLM

import torch
from torch.optim import Adam


class FineTuneOPT(LightningModule):
    """
    very straightforward direct fine tuning on the OPT model
    """
    def __init__(self, model_name="facebook/opt-350m"):
        super().__init__()
        self.model = OPTForCausalLM.from_pretrained(model_name)
        self.save_hyperparameters()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        # we care only about the last token being predicted
        pred_token_logits = logits[:, -1, :]
        pred_token = torch.argmax(pred_token_logits, dim=-1)
        labels = batch["labels"][:, -1]

        self.log("val_loss", val_loss)

        return {"loss": val_loss, "preds": pred_token, "labels": labels}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), **wandb.config["optimizer_params"])

        # configure learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, **wandb.config["lr_scheduler_params"])

        lr_scheduler_config = {"scheduler": lr_scheduler}
        lr_scheduler_config.update(wandb.config["lr_scheduler_config"])

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    """
    Note on following hooks (on_train_epoch_start and on_validation_epoch_start):

    Using the following code to access dataloaders: self.train_dataloader().dataset.set_epoch(self.current_epoch) 
    Results in an exception like such : pytorch_lightning.utilities.exceptions.MisconfigurationException: 
    `val_dataloader` must be implemented to be used with the Lightning Trainer 

    Although train_dataloader() is a valid hook, the hook is overridden only in the datamodule and we cannot reference
    that. We have to use self.trainer.train_dataloader.dataset which returns some CombinedDataset and then .datasets
    that one to get the original TorchIterableDataset.

    On the other hand, we can access validation dataloaders with self.trainer.val_dataloaders[0].dataset as that one is
    apparently a list and not a CombinedDataset.

    Pain.
    """

    def on_train_epoch_start(self) -> None:
        # reshuffle the dataset for every train epoch
        self.trainer.train_dataloader.dataset.datasets.set_epoch(self.trainer.current_epoch)

    def on_validation_epoch_start(self) -> None:
        # reshuffle the dataset for every validation epoch
        self.trainer.val_dataloaders[0].dataset.set_epoch(self.trainer.current_epoch)
