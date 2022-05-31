import argparse
from typing import List

import pandas as pd
from transformers import GPT2Tokenizer

from metrics.bart_metric import BartScore
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT

"""
Script for automatically benchmarking model outputs against BartScore, BLEU and ROUGE scores. The file should be .pkl
format of a dataframe where the first column is the source (model predictions) and second column is the target (labels).
"""


def run_model(dataset: List[str], save_path: str, model_name: str, checkpoint: str = None, append_seq: str = "</s>"):
    # init the dataset
    if checkpoint is None:
        model = ParaphraseOPT(model_name)
    else:
        model = ParaphraseOPT.load_from_custom_save(model_name, checkpoint)

    model = model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # pad to the left because the model is autoregressive (anything to the right is ignored)
    tokenizer.padding_side = 'left'

    print("Encoding dataset.")
    # append a sequence to the end of every input (could be </s> token or prompt like "paraphrase:") and encode all
    encoded_inputs = tokenizer([i + append_seq for i in dataset], padding=True, return_tensors='pt')

    print("Generating model predictions.")
    # if use_cache=False is not used there will be dim mismatch as huggingface is cringe
    output_sequences = model.model.generate(inputs=encoded_inputs['input_ids'].to(model.model.device),
                                            max_length=69,
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


def benchmark(filepath):
    print("loading")
    df = pd.read_pickle(filepath)
    print(df)


def main():
    parser = argparse.ArgumentParser(description='Model Benchmark Parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to benchmark')


if __name__ == "__main__":
    # main()
    run_model(dataset=["Why do I always get depressed in the evening?",
                       "What is the most important book you have ever read?",
                       "What is purpose of life?"],
              save_path=r"benchmark_runs/test.pkl",
              model_name="facebook/opt-1.3b",
              checkpoint=r"../training_checkpoints/30-05-2022-1.3b/soft-opt-epoch=179-val_loss=1.397.ckpt")

    benchmark(r"benchmark_runs/test.pkl")
