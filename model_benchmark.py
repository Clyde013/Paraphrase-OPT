import argparse
from typing import List

import pandas as pd
import torch
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


def run_model(dataset: List[str], save_path: str, model_name: str, checkpoint: str = None, append_seq: str = "</s>"):
    # init the dataset
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
    # if use_cache=False is not used there will be dim mismatch as huggingface is cringe
    output_sequences = model.model.generate(inputs=encoded_inputs['input_ids'].to(model.model.device),
                                            max_length=420,
                                            use_cache=False)
    # remove the source sentence based on the length of the inputs
    output_sequences = output_sequences[:, encoded_inputs['attention_mask'].size(dim=-1):]

    print("Decoding model predictions.")
    # decode outputs
    outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=False)
    # remove trailing padding and appended sequence
    outputs = [i[:i.rfind(append_seq)] for i in outputs]

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
    model_name = "facebook/opt-1.3b"
    dataset_size = 1000

    datamodule = ParaCombinedDataModule(model_name, 1, 1000, [ParabankDataModule, ParaNMTDataModule],
                                        probabilities=[0.35, 0.65], seed=1337, pre_tokenize=False)
    datamodule.setup()

    # get the values from {"source": "...</s>..."} dict and then take only the first as dataset input for model
    dataset = [i["source"].split("</s>")[0] for i in list(datamodule.dataset.take(dataset_size))]

    run_model(dataset=dataset,
              save_path=r"metrics/benchmark_runs/model_preds/1.3b-paracombined-5000-samples.pkl",
              model_name="facebook/opt-1.3b",
              checkpoint=r"training_checkpoints/01-06-2022-1.3b-paracombined/soft-opt-epoch=269-val_loss=1.862.ckpt")

    benchmark_pairs(r"metrics/benchmark_runs/model_preds/1.3b-paracombined-5000-samples.pkl",
                    save_path=r"metrics/benchmark_runs/model_benchmarked_results/1.3b-paracombined-5000-samples.pkl")
