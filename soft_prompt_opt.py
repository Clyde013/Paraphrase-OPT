from transformers.models.opt.modeling_opt import *
from soft_prompt_tuning.soft_embedding import SoftEmbedding


class SoftOPTModelWrapper(OPTForCausalLM):
    """Wrapper class for OPTForCausalLM to add learnable embedding functionality
    Simply initialise it from pretrained OPT weights files and it should work out of the box.
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

        # concatenation of tensors have to happen on the same device
        input_ids = torch.cat([torch.full((1, self.n_tokens), 50256).to(input_ids.device), input_ids], 1)
        labels = torch.cat([torch.full((1, self.n_tokens), 50256).to(labels.device), labels], 1)
        attention_mask = torch.cat([torch.full((1, self.n_tokens), 1).to(attention_mask.device), attention_mask], 1)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
