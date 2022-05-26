# Paraphrase OPT

Training OPT for paraphrasing through prompt engineering.

It seems like GPT3 will perform quite well with just telling it directly to paraphrase the following sentence, and
bumping up the frequency and presence penalties.

# Ideas

Given that OPT is a decoder only model, how will we get it to perform what is traditionally considered a seq-2-seq task
which involves cross attention from the encoder outputs, transforming the input sequence into an output sequence of a
different phrasing. The function of the encoder output cross attention is for the model to maintain a strong reference
point to the original sequence while predicting the next tokens, which is better than appending the input to the start
of the sequence and referring to the causal mask of the decoder sequence since the encoder and decoder embedding spaces
no longer have to be aligned.

Differentiable prompt (DART)[https://arxiv.org/pdf/2108.13161.pdf] except we adapt it from MLM to CLM. Instead of labels
based on a the output of a single \[MASK\] token we generate a whole sequence and evaluate the semantic similarity of
the output sequence. The input template when fed into a MLM model looks like this:

Xprompt = [CLS] Xin [SEP] T [SEP]

where T is the template prompt with containing single [MASK] token, of the form:

{h0,...,hi,w([MASK]),hi+1,...,hm}

# GCloud Compute CLI Cheatsheet

```
gcloud compute ssh liewweipyn
gcloud compute instances stop liewweipyn
```