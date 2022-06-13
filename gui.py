import copy

import streamlit as st

import torch
from transformers import GPT2Tokenizer, OPTForCausalLM

import wandb

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT

# init
wandb.init(project="popt-gui", entity="clyde013")
wandb.config.update({"embedding_n_tokens": 111}, allow_val_change=True)
AVAIL_GPUS = min(1, torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fits_on_gpu = True

# we want to define the functions without actually calling them, so we wrap them in lambdas
model_name = "facebook/opt-1.3b"
checkpoint = r"training_checkpoints/optimize/soft-opt-epoch=029-val_loss=0.487-optimizer_type=Adam-embedding_n_tokens=111.ckpt"
model_type_key = {'OPT1.3B Prompt Fine Tuned': lambda: ParaphraseOPT.load_from_custom_save(model_name, checkpoint),
                  'OPT1.3B Base Model': lambda: OPTForCausalLM.from_pretrained(model_name)}
# model specific prompt changes
model_prompt_key = {'OPT1.3B Prompt Fine Tuned': lambda x: x + "</s>",
                    'OPT1.3B Base Model': lambda x: x}


# expensive functions that need caching
@st.cache(hash_funcs={ParaphraseOPT: lambda _: None})
def reconstruct_tokens(model):
    """
    Find the nearest tokens in the embedding space.
    https://stackoverflow.com/questions/64523788/how-to-invert-a-pytorch-embedding
    """
    embeddings = model.model.soft_embedding.wte
    learned_embedding = model.model.soft_embedding.learned_embedding

    reconstructed = list()
    for i in learned_embedding:
        distance = torch.norm(embeddings.weight.detach() - i, dim=1)
        nearest = torch.argmin(distance)
        reconstructed.append(nearest.item())

    return reconstructed


@st.cache(hash_funcs={ParaphraseOPT: lambda _: None, OPTForCausalLM: lambda _: None, GPT2Tokenizer: lambda _: None})
def init_model(selection: str):
    global device
    global fits_on_gpu
    global gpu_counter

    init_model = model_type_key[selection]()
    try:
        init_model.to(device)
    except RuntimeError:
        fits_on_gpu = False
        device = torch.device("cpu")
        init_model.to(device)
        torch.cuda.empty_cache()
        gpu_counter.text("Model does not fit on GPU, using CPU instead.")

    return init_model, GPT2Tokenizer.from_pretrained(model_name)


def tokenize(model_type: str, prompt: str):
    soft_prompt = model_prompt_key[model_type](prompt)
    encoded_inputs = tokenizer(soft_prompt, return_tensors="pt")
    return encoded_inputs


def predict(model_type: str, prompt: str, max_len: int):
    inputs = tokenize(model_type, prompt)
    if model_type == "OPT1.3B Base Model":
        outputs = model.generate(inputs.input_ids, max_length=max_len, use_cache=False)
    else:
        outputs = model.model.generate(inputs.input_ids, max_length=max_len, use_cache=False)
    outputs = outputs[:, inputs['input_ids'].size(dim=-1):]
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded


gpu_counter = st.sidebar.empty()
if fits_on_gpu:
    gpu_counter.text(f"Currently utilising {AVAIL_GPUS} GPU(s).")
else:
    gpu_counter.text("Model does not fit on GPU, using CPU instead.")
cache_warning = st.sidebar.write("Loading models for the first time may take a while. Future selections will "
                                 "be greatly faster as models are cached! :)")

# Add a selectbox for model type to the sidebar:
model_selectbox = st.sidebar.selectbox(
    'Which model would you like to use?',
    model_type_key.keys()
)

# Add slider for max length of the sequence
max_len_slider = st.sidebar.slider("What is the maximum sequence length that you would like the model to generate? "
                                   "(counts input tokens)",
                                   min_value=30, max_value=100, value=45)

model, tokenizer = init_model(model_selectbox)


with st.expander("Model Demo"):
    # input text box
    input_txt = st.text_area("Enter text for the model:", "The quick brown fox jumped over the fence.")

    # token count for input
    seq_len = tokenize(model_selectbox, input_txt)['input_ids'].size(dim=-1)
    token_count = st.caption(f"Token length: {seq_len}")

    # output text box
    with st.spinner(text="Generating..."):
        model_outputs_txt = st.code(predict(model_selectbox, input_txt, max_len_slider), language="markdown")


with st.expander("Learned Embeddings"):
    if model_selectbox == 'OPT1.3B Prompt Fine Tuned':
        tokens = reconstruct_tokens(model)
        outputs = tokenizer.batch_decode(tokens)
        st.text(outputs)
        st.text("".join(outputs))
    else:
        st.text("Default model does not have any prepended learned embeddings.")