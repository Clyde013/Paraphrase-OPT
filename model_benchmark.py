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
                                                max_length=100,
                                                use_cache=False).to('cpu')
            # free the memory (it isn't actually removed from gpu but is able to be overwritten)
            del batch

            # remove the source sentence based on the length of the inputs
            output_batch = output_batch[:, encoded_inputs['input_ids'].size(dim=-1):]

            # decode outputs, after removal of source sentence should only remain eos token and padding on the right
            # which are omitted by skip_special_tokens=True
            outputs = tokenizer.batch_decode(output_batch, skip_special_tokens=True)
            output_sequences.extend(outputs)

    print("Dataframe saving.")
    df = pd.DataFrame({"preds": output_sequences, "src": dataset})
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
    dataset_size = 100

    print("Datamodule setup.")
    datamodule = ParaCombinedDataModule(model_name, 1, 1000, [ParabankDataModule, ParaNMTDataModule],
                                        probabilities=[0.35, 0.65], seed=82765, pre_tokenize=False)
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
Encoding dataset.
Generating model predictions.
  0%|                                                                                                             | 0/3 [00:00<?, ?it/s]
  [' bird stopped singing.', ': Implementation of Directive 2002/83/EC', ' want information about agent, please select agent.']
 33%|█████████████████████████████████▋                                                                   | 1/3 [00:00<00:00,  3.16it/s]
 ['terilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is a problem. Sterilization is', ' am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take a large.i am going to take', ' someone who knows what goes on in the inner circle of a corporation like this...... knows what goes on in the inner circle of a corporation like this.']
 67%|███████████████████████████████████████████████████████████████████▎                                 | 2/3 [00:13<00:07,  7.98s/it]
 [' i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir. `` i think i do, sir.`` i think i do, sir. `` i think i do, sir.`` i think i do, sir. `` i think i do, sir. `` i think i do, sir.`` i think i do, sir. `` i think i do, sir.`` i think i do, sir.', ' know, i know.', 'manent. permanent. permanent. permanent.']
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:20<00:00,  6.71s/it]
Dataframe saving.
Traceback (most recent call last):
  File "/home/liewweipyn_aisingapore_org/Paraphrase-OPT/model_benchmark.py", line 124, in <module>
    run_model(dataset=dataset,
  File "/home/liewweipyn_aisingapore_org/Paraphrase-OPT/model_benchmark.py", line 75, in run_model
    df = pd.DataFrame({"preds": outputs, "src": dataset})
  File "/opt/conda/envs/OPT/lib/python3.10/site-packages/pandas/core/frame.py", line 636, in __init__
    mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
  File "/opt/conda/envs/OPT/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 502, in dict_to_mgr
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
  File "/opt/conda/envs/OPT/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 120, in arrays_to_mgr
    index = _extract_index(arrays)
  File "/opt/conda/envs/OPT/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 674, in _extract_index
    raise ValueError("All arrays must be of the same length")
ValueError: All arrays must be of the same length

    """

