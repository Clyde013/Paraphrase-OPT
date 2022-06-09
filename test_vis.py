import os

import torch
import wandb
import numpy as np
import pickle

from transformers import GPT2Tokenizer
from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT

from faerun import Faerun
import tmap as tm

import re

print("Initialising...")

wandb.init(project="test-popt-dump", entity="clyde013", name="test-model", allow_val_change=True)
wandb.config.update({"embedding_n_tokens": 111}, allow_val_change=True)

#checkpoint = r"training_checkpoints/30-05-2022-1.3b/soft-opt-epoch=179-val_loss=1.397.ckpt"
checkpoint = r"training_checkpoints/optimize/soft-opt-epoch=029-val_loss=0.487-optimizer_type=Adam-embedding_n_tokens=111.ckpt"
model_name = "facebook/opt-1.3b"

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

model = ParaphraseOPT.load_from_custom_save(model_name, checkpoint)
model = model.eval()

# default_model = ParaphraseOPT(model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

learned_embeddings = model.model.soft_embedding.learned_embedding.detach()
original_embeddings = model.model.soft_embedding.wte.weight.detach()


def visualise(lf_filename: str, load_lf: bool):
    """
    minhash and lsh forest visualisation
    http://matthewcasperson.blogspot.com/2013/11/minhash-for-dummies.html
    http://infolab.stanford.edu/~bawa/Pub/similarity.pdf
    """
    dims = 1024
    enc = tm.Minhash(learned_embeddings.size(dim=1), 69, dims)
    lf = tm.LSHForest(dims * 2, 128)

    c_labels = np.concatenate([np.ones(original_embeddings.size(dim=0)), np.zeros(learned_embeddings.size(dim=0))])
    print(c_labels)
    print(c_labels.shape)

    # generate labels when you click the points
    labels = []
    # add labels for embeddings to be their decoded tokens
    # iterate through all tokenizer keys, that are stored as unicode, encode them with utf-8 to get their byte like
    # representations, have to repr() the byte string to get b'<string>' and then manually remove the b'', and then use
    # regex to remove the special character that GPTs BPE uses to denote whitespace as well as any ' " \\ that will mess
    # up the javascript source file.
    labels.extend([re.sub(r'(\\xc4|\\xa0)|[\'\"\\]', '', repr(i.encode("utf-8"))[2:-1]) for i in tokenizer.get_vocab().keys()])

    for i in range(learned_embeddings.size(dim=0)):
        labels.append(f"learned embedding {i}")

    if load_lf:
        lf.restore(f"visualisations/{lf_filename}")
    else:
        np_arr = np.concatenate([original_embeddings, learned_embeddings])
        tmp = []
        for i in np_arr:
            tmp.append(tm.VectorFloat(i.tolist()))
        print("batch add")
        lf.batch_add(enc.batch_from_weight_array(tmp))
        print("index")
        lf.index()
        print("saving lf")
        lf.store(f"visualisations/{lf_filename}")

    print("layout")
    config = tm.LayoutConfiguration()
    config.fme_randomize = False
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=config)
    print("faerun")
    legend_labels = [
        (0, "learned embeddings"),
        (255, "default embeddings")
    ]

    faerun = Faerun(clear_color="#111111", view="front", coords=False)
    faerun.add_scatter(
        "Embeddings",
        {"x": x, "y": y, "c": c_labels, "labels": labels},
        colormap="RdYlBu",
        shader="smoothCircle",
        point_scale=3,
        max_point_size=20,
        has_legend=True,
        categorical=True,
        legend_labels=legend_labels,
    )
    faerun.add_tree(
        "Embeddings_tree", {"from": s, "to": t}, point_helper="Embeddings", color="#666666"
    )

    faerun.plot(f"Embeddings_{wandb.config['embedding_n_tokens']}", path="visualisations/")
    print("done")


if __name__ == "__main__":
    visualise(f"lf_{wandb.config['embedding_n_tokens']}_seed=69.dat", False)
