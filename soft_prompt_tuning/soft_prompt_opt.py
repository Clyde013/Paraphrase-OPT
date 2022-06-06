import re
from typing import Dict

from pytorch_lightning import LightningModule, Callback
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from transformers.models.opt.modeling_opt import *
from soft_prompt_tuning.soft_embedding import SoftEmbedding

import wandb
import os


class SoftOPTModelWrapper(OPTForCausalLM):
    """Wrapper class for OPTForCausalLM to add learnable embedding functionality
    Simply initialise it with from_pretrained OPT files and it should work out of the box.
    """
    _keys_to_ignore_on_load_missing = [r"soft_embedding.wte.weight", r"soft_embedding.learned_embedding",
                                       r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)

        # init parameters for embedding
        self.n_tokens = wandb.config["embedding_n_tokens"]
        self.init_from_vocab = wandb.config["init_from_vocab"]

        # initialise the embedding to learn
        self.soft_embedding = SoftEmbedding(self.get_input_embeddings(),
                                            n_tokens=self.n_tokens,
                                            initialize_from_vocab=self.init_from_vocab)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Incredibly scuffed but we have to set the input embeddings to the soft embeddings only AFTER
        the pretrained weights have been loaded in. All parameters are the same as a normal from_pretrained() call
        """

        pretrained_model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        pretrained_model.set_input_embeddings(pretrained_model.soft_embedding)
        return pretrained_model

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs):
        """Shitty forward pass
        need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
        even though it does not matter what we pad input_ids with, it's just to make HF happy
        """

        batch_size = input_ids.shape[0]
        # Note: concatenation of tensors have to happen on the same device
        # concat padding representing our learned embedding tokens for batched inputs
        # inputs come in as (batch_size, seq_len) and are padded to be (batch_size, n_tokens + seq_len)
        input_ids = torch.cat([torch.full((batch_size, self.n_tokens), 50256).to(input_ids.device), input_ids], dim=1)
        attention_mask = torch.cat(
            [torch.full((batch_size, self.n_tokens), 1).to(attention_mask.device), attention_mask], dim=1)
        if labels is not None:
            labels = torch.cat([torch.full((batch_size, self.n_tokens), 50256).to(labels.device), labels], dim=1)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)


class ParaphraseOPT(LightningModule):
    def __init__(self, model_name="facebook/opt-350m", init_optimizer=None, init_lr_scheduler=None):
        super().__init__()
        self.model = SoftOPTModelWrapper.from_pretrained(model_name)

        # these inits should be exclusively used for loading from checkpoints
        # see load_from_custom_save for why.
        self.init_optimizer = None
        self.init_lr_scheduler = None

        self.save_hyperparameters("model_name")

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
        # thanks stack overflow!
        # https://stackoverflow.com/questions/38460918/regex-matching-a-dictionary-efficiently-in-python
        # extracting all the layers that are specified by layers_to_optimize using regex for partial matches
        regex_matches = [re.compile(".*" + pattern + ".*").match for pattern in wandb.config["layers_to_optimize"]]
        layers_to_optimize = [v for k, v in self.model.state_dict().items()
                              if any(regex_match(k) for regex_match in regex_matches)]

        # configure optimizer
        optimizers_key = {"Adam": Adam, "SGD": SGD}
        if self.init_optimizer is None:
            optimizer_type = optimizers_key[wandb.config["optimizer_type"]]
            optimizer = optimizer_type(layers_to_optimize)

            # update with specified parameters
            optim_state_dict = optimizer.state_dict()
            optim_state_dict.update(wandb.config["optimizer_params"])
            optimizer.load_state_dict(optim_state_dict)
        else:
            optimizer = self.init_optimizer

        # configure learning rate scheduler
        lr_scheduler_key = {"ReduceLROnPlateau": ReduceLROnPlateau}
        if self.init_lr_scheduler is None:
            lr_scheduler_type = lr_scheduler_key[wandb.config["lr_scheduler_type"]]
            lr_scheduler = lr_scheduler_type(optimizer)

            # update with specified parameters
            lr_scheduler_dict = lr_scheduler.state_dict()
            lr_scheduler_dict.update(wandb.config["lr_scheduler_params"])
            lr_scheduler.load_state_dict(lr_scheduler_dict)
        else:
            lr_scheduler = self.init_lr_scheduler

        lr_scheduler_config = {"scheduler": lr_scheduler}
        lr_scheduler_config.update(wandb.config["lr_scheduler_config"])

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    @classmethod
    def load_from_custom_save(cls, model_name, path, optimizer: Optimizer = None, lr_scheduler: _LRScheduler = None):
        """
        Custom save function to load from checkpoints created by SpecificLayersCheckpoint callback.

        Unfortunately pytorch lightning locks the optimizers in place after instantiation and there is no clean way
        to change them afterwards. There are some workarounds but they all suck too:
        https://github.com/PyTorchLightning/pytorch-lightning/discussions/9354
        https://github.com/PyTorchLightning/pytorch-lightning/discussions/6131

        So the current implementation is to throw in the optimizer and lr_scheduler as optional parameters during model
        instantiation before then actually updating the model weights.

        To try different optimizers and lr_schedulers change configure_optimizers() directly.
        """
        # load the saved checkpoint
        state_dict = torch.load(path)

        # load optimizer if required
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer"])

        # load lr_scheduler if required
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        # instantiate lightningmodule with pretrained model
        model = cls(model_name, optimizer, lr_scheduler)

        # load updated state dict into the model (as long as no layers are named optimizer or lr_scheduler)
        model.model.load_state_dict(state_dict, strict=False)

        return model

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


class SpecificLayersCheckpoint(Callback):
    """
    Custom saving of specific layers into a state_dict that can be loaded in using torch.load()
    Ideally, we load in the model with from_pretrained, and then use state_dict.update() to update the
    weights of the loaded model.
    """

    def __init__(self, monitor: str, dirpath: str, filename: str,
                 every_n_epochs: int, layers_to_save: List[str]):
        super().__init__()
        self.monitor = monitor
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_epochs = every_n_epochs
        self.layers_to_save = layers_to_save

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # if model should be saved this epoch (+1 since epoch count starts from 0)
        # if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
        if True:
            # thanks stack overflow!
            # https://stackoverflow.com/questions/38460918/regex-matching-a-dictionary-efficiently-in-python
            # extracting all the layers that are specified by layers_to_save using regex for partial matches
            regex_matches = [re.compile(".*" + pattern + ".*").match for pattern in self.layers_to_save]
            save_dict = {k: v for k, v in pl_module.model.state_dict().items()
                         if any(regex_match(k) for regex_match in regex_matches)}

            # save the optimizer
            if pl_module.optimizers() is not None:
                save_dict.update({"optimizer": pl_module.optimizers().optimizer.state_dict()})

            # save the lr_scheduler
            if pl_module.lr_schedulers() is not None:
                save_dict.update({"lr_scheduler": pl_module.lr_schedulers().state_dict()})

            formatted_filename = self.filename.format(epoch=trainer.current_epoch, **trainer.callback_metrics)
            torch.save(save_dict, os.path.join(self.dirpath, formatted_filename))
