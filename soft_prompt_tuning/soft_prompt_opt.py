from pytorch_lightning import LightningModule, Callback
from torch.optim import Adam
from transformers.models.opt.modeling_opt import *
from .soft_embedding import SoftEmbedding
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
        self.n_tokens = 20
        self.init_from_vocab = True

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
    def __init__(self):
        super().__init__()
        self.model = SoftOPTModelWrapper.from_pretrained("facebook/opt-350m")

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
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
        optimizer = Adam([self.model.soft_embedding.learned_embedding])
        return optimizer

    """
    Note on following hooks (on_train_epoch_start and on_validation_epoch_start):
    
    Using the following code to access dataloaders:
    self.train_dataloader().dataset.set_epoch(self.current_epoch)
    Results in an exception like such :
    pytorch_lightning.utilities.exceptions.MisconfigurationException: `val_dataloader` must be implemented to be used with the Lightning Trainer
    
    Although train_dataloader() is a valid hook, the hook is overriden only in the datamodule and we cannot reference
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
    def __init__(self, monitor: str, dirpath: str, filename: str, every_n_epochs: int, layers_to_save: List[nn.Module]):
        super().__init__()
        self.monitor = monitor
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_epochs = every_n_epochs
        self.layers_to_save = layers_to_save

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # if model should be saved this epoch (+1 since epoch count starts from 0)
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            save_dict = dict()
            for layer in self.layers_to_save:
                save_dict.update(layer.state_dict())

            formatted_filename = self.filename.format(epoch=trainer.current_epoch, **trainer.callback_metrics)

            torch.save(save_dict, os.path.join(self.dirpath, formatted_filename))
