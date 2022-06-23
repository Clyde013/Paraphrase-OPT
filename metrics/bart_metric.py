from typing import Dict, Any, List

from torch import Tensor
import torch.nn as nn
import torch

from torchmetrics import Metric
from transformers import BartTokenizer, BartForConditionalGeneration


class BartScore(Metric):
    """
    Torchmetric version of BartScore as adapted from their github
    https://github.com/neulab/BARTScore
    With lots of reference to bertscore implementation
    https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/text/bert.py#L40-L235

    Compute the score by:
    ```
    bartscore = BartScore()
    score = bartscore(['This is interesting.', 'This is a good idea.'], ['This is fun.', 'Sounds like a good idea.'])
    ```
    and it should return [-2.152808666229248, -2.948076009750366].
    """

    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn',
                 **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

        # Set up model
        self.device_ = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

        # Set up metric state variables which keep track of state on each call of update
        self.add_state("src_input_ids", [], dist_reduce_fx="cat")
        self.add_state("src_attention_mask", [], dist_reduce_fx="cat")
        self.add_state("target_input_ids", [], dist_reduce_fx="cat")
        self.add_state("target_attention_mask", [], dist_reduce_fx="cat")

    def update(self, preds: List[str], target: List[str]) -> None:
        # dict of 2d list of tensors [batch_size, input_size] although input_size is not fixed
        encoded_src = self.tokenizer(preds, padding=True, return_tensors='pt')
        encoded_targets = self.tokenizer(target, padding=True, return_tensors='pt')

        # 3d list of 2d tensors, since default values of state variables can only be lists or tensors
        self.src_input_ids.append(encoded_src['input_ids'])
        self.src_attention_mask.append(encoded_src['attention_mask'])
        self.target_input_ids.append(encoded_targets['input_ids'])
        self.target_attention_mask.append(encoded_targets['attention_mask'])

    def compute(self):
        score_list = []

        src_tokens = self.src_input_ids[0].to(self.device_)
        src_mask = self.src_attention_mask[0].to(self.device_)

        tgt_tokens = self.target_input_ids[0].to(self.device_)
        tgt_mask = self.target_attention_mask[0]
        tgt_len = tgt_mask.sum(dim=1).to(self.device_)

        # while we do not use the loss computation as a result of labels being provided, the labels also cause
        # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_bart.py#L1320
        # the decoder input id to be shifted to the right for us, which is needed for this to work
        output = self.model(
            input_ids=src_tokens,
            attention_mask=src_mask,
            labels=tgt_tokens
        )

        # loss calculation based on original bart_score
        logits = output.logits.view(-1, self.model.config.vocab_size)
        lsm_output = self.lsm(logits)
        loss = self.loss_fct(lsm_output, tgt_tokens.view(-1))
        loss = loss.view(tgt_tokens.shape[0], -1)
        loss = loss.sum(dim=1) / tgt_len
        curr_score_list = [-x.item() for x in loss]
        score_list += curr_score_list

        return score_list


def main():
    bartscore = BartScore()
    score = bartscore(['This is interesting.', 'This is interesting.'],
                      ['This is very curious.', 'This is incredibly strange.'])
    print(score)


if __name__ == "__main__":
    main()
