# Paraphrase OPT

Training OPT for paraphrasing through prompt engineering.

It seems like GPT3 will perform quite well with just telling it directly to paraphrase the following sentence, and
bumping up the frequency and presence penalties.

However, OPT's smaller variants (125m & 350m) do not seem to be able to understand the prompt "paraphrase:" and instead
attempt to continue the sentence:

```
'Once, a group of frogs was roaming around the forest in search of water. 
Paraphrase: "I\'m thirsty."\n\nThe group of frogs was so thirsty that they were unable 
to find water.\n\nThe group of frogs was'
```

# Current Implementation

Copy pasted soft prompt tuning and created a huggingface model wrapper around OPT for it. It seems to be working. *
seems*.

Might have to alter
the [loss function calculation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L923)
for CLM in OPTs forward pass. Currently it shifts the labels to the right by 1(???) such that the loss applies to
predictions for tokens < n (logits predict what comes after n+1?). Might have to alter the loss function such that only
loss after the <sep> token counts.

TODO:
- [ ] install nvidia-smi drivers on gcloud compute
- [ ] i swear there will be a tensor dim mismatch error caused by [this line](https://github.com/Clyde013/Paraphrase-OPT/blob/main/soft_prompt_tuning/soft_prompt_opt.py#L80)
- [ ] train the model
- [ ] ensure that only embeddings are being trained, and the model weights are fixed
- [ ] profit???

# Ideas

Given that OPT is a decoder only model, how will we get it to perform what is traditionally considered a seq-2-seq task
which involves cross attention from the encoder outputs, transforming the input sequence into an output sequence of a
different phrasing. The function of the encoder output cross attention is for the model to maintain a strong reference
point to the original sequence while predicting the next tokens, which is better than appending the input to the start
of the sequence and referring to the causal mask of the decoder sequence since the encoder and decoder embedding spaces
no longer have to be aligned.

## Soft Prompt Tuning

[Parameter Efficient Soft Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) seems to be the original implementation
that was not referenced in the DART paper. The codebase is much simpler (actually just 1 .py file) and extends the
HuggingFace library in a very simple way, just concatenating the soft prompt embeddings directly to the input.

<img src="https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.png?raw=true" width=300/>

[Github](https://github.com/kipgparker/soft-prompt-tuning) source for soft prompt tuning.

## DART Implementation (???)

Refer to soft prompt tuning. The methodology seems exactly the same, except that DART can be applied to any language
model, and they added fluency constraint objectives to the model training to ensure the differentiable prompts retain
association between template tokens.

Differentiable prompts [DART](https://arxiv.org/pdf/2108.13161.pdf) except we adapt it from MLM to CLM. Instead of
labels based on a the output of a single [MASK] token we generate a whole sequence and evaluate the semantic similarity
of the output sequence.

The input prompt when fed into an MLM model looks like this:
X<sub>prompt</sub> = [CLS] X<sub>in</sub> [SEP] T [SEP]

where T is the template prompt with containing single [MASK] token, of the form:
{h<sub>0</sub>,...,h<sub>i</sub>,w([MASK]),h<sub>i+1</sub>,...,h<sub>m</sub>}

Since OPT as a decoder is autoregressive, we alter T as such (pred<sub>k</sub> are predicted tokens from previous k
iterations):
{h<sub>0</sub>,...,h<sub>i</sub>,pred<sub>0</sub>,...,pred<sub>k</sub>,w([MASK])}

Prompt embeddings that come after w([MASK]) will be masked and ignored anyway, hence we omit them in this
implementation. The input prompt when fed into OPT (formatted similarly to GPT2's tokenizer) will then look like this:
X<sub>prompt</sub> = [EOS] X<sub>in</sub> [BOS/EOS] {h<sub>0</sub>,...,h<sub>i</sub>,pred<sub>0</sub>,...,pred<sub>
k</sub>,w([MASK])}

We then iterate through multiple forward passes until we reach an eos_token output by the model or max length of the
sequence.

# ERROR SHEET

Some errors may pop up when trying to run the program. "But it works on my machine" yeah it will work on your machine
when you do these things.

### protobuf error

Might encounter an error with protobuf apparently one of google's updates broke it so its incompatible with pytorch
lightning. Quick fix is to downgrade it to an older version:

```buildoutcfg
pip install protobuf==3.20.1
```

# GCloud Compute CLI Cheatsheet

```
gcloud compute instances start liewweipyn
gcloud compute ssh liewweipyn
gcloud compute instances stop liewweipyn
```