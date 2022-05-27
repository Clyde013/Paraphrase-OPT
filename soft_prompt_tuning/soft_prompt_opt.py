from pytorch_lightning import LightningModule
from torch.optim import Adam
from transformers.models.opt.modeling_opt import *

from .soft_embedding import SoftEmbedding


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
        # number of training examples sampled = total_steps * batch_size * grad_accumulation_batches
        # self.total_steps = total_steps

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
        # TODO: I expect there to be some error with the validation step dimension mismatch

        pred_token_logits = logits[:, -1, :]
        pred_token = torch.argmax(pred_token_logits, dim=-1)

        """
        IndexError: too many indices for tensor of dimension 2
        """
        labels = batch["labels"][:, -1]

        return {"loss": val_loss, "preds": pred_token, "labels": labels}

    def configure_optimizers(self):
        optimizer = Adam(self.model.soft_embedding.wte.parameters())
        return optimizer
