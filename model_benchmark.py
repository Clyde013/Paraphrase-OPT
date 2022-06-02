import argparse
import os
from typing import List

from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

from metrics.bart_metric import BartScore
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT
from training_datasets.para_nmt.para_nmt import ParaNMTDataModule
from training_datasets.parabank.parabank import ParabankDataModule
from training_datasets.paracombined import ParaCombinedDataModule

"""
Script for automatically benchmarking model outputs against BartScore, BLEU and ROUGE scores. The file should be .pkl
format of a dataframe where the first column is the source (model predictions) and second column is the target (labels).
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_model(dataset: List[str], batch_size: int, save_path: str, model_name: str, checkpoint: str = None, append_seq: str = "</s>"):
    # init the dataset
    print("Initialising.")
    if checkpoint is None:
        model = ParaphraseOPT(model_name)
    else:
        model = ParaphraseOPT.load_from_custom_save(model_name, checkpoint)

    model = model.eval()
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # pad to the left because the model is autoregressive (anything to the right is ignored)
    tokenizer.padding_side = 'left'

    print("Encoding dataset.")
    # append a sequence to the end of every input (could be </s> token or prompt like "paraphrase:") and encode all
    encoded_inputs = tokenizer([i + append_seq for i in dataset], padding=True, return_tensors='pt')

    print("Generating model predictions.")
    """ Yeah. Don't pass .generate() all the encoded inputs at once.
    RuntimeError: CUDA out of memory. Tried to allocate 17.61 GiB (GPU 0; 39.59 GiB total capacity; 23.04 GiB 
    already allocated; 14.16 GiB free; 23.38 GiB reserved in total by PyTorch) If reserved memory is >> allocated 
    memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and 
    PYTORCH_CUDA_ALLOC_CONF
    """
    output_sequences = list()
    # ensure no intermediate gradient tensors are stored. We need all the memory we can get.
    with torch.no_grad():
        for i in tqdm(range(0, encoded_inputs['input_ids'].size(dim=0), batch_size)):
            batch = encoded_inputs['input_ids'][i:i+batch_size]
            # if use_cache=False is not used there will be dim mismatch as huggingface is cringe
            output_batch = model.model.generate(inputs=batch.to(model.model.device),
                                                max_length=420,
                                                use_cache=False).to('cpu')
            # free the memory (it isn't actually removed from gpu but is able to be overwritten)
            del batch

            # remove the source sentence based on the length of the inputs
            output_batch = output_batch[:, encoded_inputs['attention_mask'].size(dim=-1):]
            output_sequences.append(output_batch)

    print("Decoding model predictions.")
    # concat batches into one tensor, since they are uneven use pad_sequence which will add padding value
    output_sequences = pad_sequence(output_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    print(output_sequences)
    # decode outputs
    outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=False)
    # remove trailing padding
    outputs = [i[:i.find(tokenizer.pad_token)] if i.find(tokenizer.pad_token) else i for i in outputs]

    print("Dataframe saving.")
    df = pd.DataFrame({"preds": outputs, "src": dataset})
    df.to_pickle(save_path)

    print(df)


def benchmark_pairs(filepath, save_path):
    print("Loading for predictions.")
    df = pd.read_pickle(filepath)

    # init metrics
    bart = BartScore()
    rouge = ROUGEScore()
    bleu = BLEUScore()

    # apply the metrics on the source and target sentence
    def score(row):
        src, target = row
        bartscore = bart([src], [target])[0]
        bleuscore = bleu([src], [[target]]).item()
        rougescore = {k: v.item() for k, v in rouge(src, target).items()}
        series = pd.Series([src, target, bartscore, bleuscore], index=["src", "target", "bartscore", "bleuscore"])
        return pd.concat([series, pd.Series(rougescore)])

    # apply score function along each row
    print("Scoring sequence pairs.")
    df = df.apply(score, axis=1)
    print(df)
    df.to_pickle(save_path)


if __name__ == "__main__":
    package_directory = os.path.dirname(os.path.abspath(__file__))

    model_preds_save_path = "metrics/benchmark_runs/model_preds/1.3b-paracombined-5000-samples.pkl"
    benchmark_save_path = "metrics/benchmark_runs/model_benchmarked_results/1.3b-paracombined-5000-samples.pkl"
    checkpoint_path = "training_checkpoints/01-06-2022-1.3b-paracombined/soft-opt-epoch=269-val_loss=1.862.ckpt"

    model_name = "facebook/opt-1.3b"
    dataset_size = 30

    print("Datamodule setup.")
    datamodule = ParaCombinedDataModule(model_name, 1, 1000, [ParabankDataModule, ParaNMTDataModule],
                                        probabilities=[0.35, 0.65], seed=2975, pre_tokenize=False)
    datamodule.setup()

    # get the values from {"source": "...</s>..."} dict and then take only the first as dataset input for model
    dataset = [i["source"].split("</s>")[0] for i in list(datamodule.dataset.take(dataset_size))]

    run_model(dataset=dataset,
              batch_size=5,
              save_path=os.path.join(package_directory, model_preds_save_path),
              model_name=model_name,
              checkpoint=os.path.join(package_directory, checkpoint_path))

    benchmark_pairs(os.path.join(package_directory, model_preds_save_path),
                    save_path=os.path.join(package_directory, benchmark_save_path))


    """
Traceback (most recent call last):
  File "/home/liewweipyn_aisingapore_org/Paraphrase-OPT/model_benchmark.py", line 127, in <module>
    run_model(dataset=dataset,
  File "/home/liewweipyn_aisingapore_org/Paraphrase-OPT/model_benchmark.py", line 71, in run_model
    output_sequences = torch.cat(output_sequences, dim=0)
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 359 but got size 19 for tensor number 1 in the list.


    """

