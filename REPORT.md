# Teaching OPT to Paraphrase through Soft Prompt Tuning

## Table of Contents
1. [Introduction](#introduction)
    1. [Prompt Tuning](#prompt-tuning)
    2. [Soft Prompts](#soft-prompts)
2. [Datasets](#datasets)
3. [Implementation](#implementation)
    1. [HuggingFace Model Wrapper](#HF-wrapper)
    3. [Training](#training)
4. [Results](#results)
    1. [Scoring Metrics](#metrics)
    2. [Visualisation](#visualisation)
5. [Conclusion](#conclusion)


<a id="introduction"></a>
## Introduction 
Open Pre-trained Transformer models are a collection of open source decoder-only pre-trained transformers ranging
from 125M to 175B parameters, with 175B showing comparable performance with GPT3. Such large language models
display remarkable performance in zero and few-shot tasks, making prompting a promising solution for many tasks
due to the capability of coaxing a large model into solving tasks that they were not explicitly trained to do.

The task that we are trying to accomplish here, is to prompt OPT models to paraphrase sentences. The
task of paraphrasing is traditionally a sequence-to-sequence task accomplished using encoder-decoder
transformer architectures (such as BART), however there is still some promise in leveraging the large pretrained
decoder only models like OPT, whose capabilities to model natural language may overcome the architectural
limitations.

However, in the course of this experiment we only work with OPT1.3B, a much smaller variant of the OPT175B model,
hence the results will of course not be incredibly good, as a smaller model is unable to grasp sufficient complexities
of the task at hand.

<a id="prompt-tuning"></a>
### Prompt Tuning
For example, in OpenAI's GPT3 playground, we can use different techniques such as 
[in-context learning](http://ai.stanford.edu/blog/in-context-learning/) and
[chain of thought](https://arxiv.org/pdf/2205.11916.pdf) prompting. An excellent example of chain-of-thought
prompting is provided by the aforementioned paper:

![](images/Screenshot%202022-06-20%20174801.png)
![](images/Screenshot%202022-06-20%20174920.png)

<a id="soft-prompts"></a>
### Soft Prompts
The concept of soft prompts was introduced by Lester et al. in the [paper](https://arxiv.org/pdf/2104.08691.pdf) titled
"The Power of Scale for Parameter-Efficient Prompt Tuning", where they explored prepending learnable tokens to the 
inputs of frozen language models as such:

<img src="https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.png?raw=true" width=300/>

The learnable tokens can be thought of as passing conceptual notions to the model in an attempt to get it to better
understand the task that is requested. As the models represent words as numbers in a high dimensional space,
we need not restrict our prompts to discrete words, and can search in the space between words to find the most
suitable prompts to feed into the model.

These prompts are very efficient in terms of memory and compute, requiring just 0.2% of the size of a complete model
checkpoint which would store fine-tuned model parameters, as well as being capable of achieving good results in
less training time.

<a id="datasets"></a>
## Datasets
Two popular paraphrasic datasets were used in soft prompt training of the models.
[ParaBank 2.0](https://nlp.jhu.edu/parabank/) and [ParaNMT-50M](https://arxiv.org/pdf/1711.05732.pdf),
both datasets generated through automatic translation of large amounts of bilingual textual data, translating a
foreign language to english to obtain english-english paraphrase pairs.

For example, the ParaNMT-50M dataset used Czech-English parallel pairs and applied a Czech to English
pretrained model for translation on the czech pairs.

As the datasets are incredibly large, we utilised HuggingFace's [dataset streaming](https://huggingface.co/docs/datasets/stream)
feature to progressively feed training data to the model.

Initially the baseline 20 token model was trained on a 35%-65% split of Parabank 2.0 and ParaNMT-50M respectively, 
however for parameter optimization, all further models were trained on a 50%-50% split of Parabank 2.0 and ParaNMT-50M
respectively.


<a id="implementation"></a>
## Implementation
The model was implemented using the OPT model provided by the HuggingFace team, organising the
training logic with Pytorch Lightning, tracking the model performance with Weights and Biases, and
multiple visualisations using Streamlit and Graphistry.

<a id="HF-wrapper"></a>
### HuggingFace Model Wrapper
The implementation of the soft prompts follows nearly identical to the 
[Github here](https://github.com/kipgparker/soft-prompt-tuning) where the soft prompts are
simply float tensors duplicated from existing vocabulary and adding them to the module's list of
parameters, to be considered backpropagatable tensors.

The relevant code snippet is shown below, and the full implementation is 
[here](https://github.com/Clyde013/Paraphrase-OPT/blob/fb8f59d6987e3902baf05fa375c856f86e139bb3/soft_prompt_tuning/soft_embedding.py#L26-L44).
```python
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True) -> torch.Tensor:
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
```

We then subclass the HuggingFace's [`OPTForCausalLM`](https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTForCausalLM) 
class, initialise a new soft embedding and 
[override the forward pass](https://github.com/Clyde013/Paraphrase-OPT/blob/fb8f59d6987e3902baf05fa375c856f86e139bb3/soft_prompt_tuning/soft_prompt_opt.py#L44-L64) 
to prepend our learned embeddings in front of the input.

```python
 def forward(self,
             input_ids: torch.LongTensor = None,
             attention_mask: Optional[torch.Tensor] = None,
             labels: Optional[torch.LongTensor] = None,
             **kwargs):
     batch_size = input_ids.shape[0]
     # Note: concatenation of tensors have to happen on the same device
     # concat padding representing our learned embedding tokens for batched inputs
     # inputs come in as (batch_size, seq_len) and are padded to be (batch_size, n_tokens + seq_len)
     input_ids = torch.cat([torch.full((batch_size, self.n_tokens), 50256).to(input_ids.device), input_ids], dim=1)
     attention_mask = torch.cat(
         [torch.full((batch_size, self.n_tokens), 1).to(attention_mask.device), attention_mask], dim=1)
     if labels is not None:
         labels = torch.cat([torch.full((batch_size, self.n_tokens), 50256).to(labels.device), labels], dim=1)

     return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
```

<a id="training"></a>
### Training
Training was done on the OPT1.3B variant, and hyperparameter search for the optimal number of soft tokens
using Optuna. 

All models were trained for 8000 steps per epoch with batch size of 32, and some early stopping applied to
prune under performing models.

It was clear early on that Adam optimizer performed better than Stochastic Gradient Descent, and as such all
further trials were done using the Adam optimizer.

Below are a few selected runs that show a very clear trend.
![](images/W&B%20Chart.png)

<a id="results"></a>
## Results
The models were allowed to run on a small subset of the dataset and their outputs saved, as expected the
results are not fantastic. The model is comparatively small, with only 1.3 billion parameters, and as such
soft prompt tuning will not achieve state of the art performance. Nevertheless, it is observed that at
semantic similarity is maintained, instead of the usual action of OPT continuing to generate the sentence.
Unfortunately the model is unable to comprehend that it should be paraphrasing, and thus changing lexical components
of the input, however as model size increases, it is reasonable to assume that performance will get better. The
following is a selection of some of the better paraphrased results from the model.

| model preds                                                                                           | target                                                                                                |
|:------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| for the movie that's being shot in 1 5 minutes?                                                       | for the movie that we 're shooting in about 1 5 minutes ?                                             |
| adler took a moment to consider that, and nodded.                                                     | adler took a few seconds to consider that , then nodded thoughtfully .                                |
| david schwartz was unaware of the fact that he was only narrowly avoided a permanent dent in his ego. | david schwartz was unaware of how narrowly he had escaped crippling and a permanent dent in his ego . |
| i had no idea you were a stunt performer.                                                             | i had no idea you did stunt work .                                                                    |
| seldon was no longer traveling around only when accompanied.                                          | seldon no longer traveled around only if accompanied .                                                |

The next question that comes to mind is how do we evaluate their predictions?

<a id="metrics"></a>
### Metrics
In order to evaluate our model, we employ multiple different metrics. Traditional metrics such as BLEU and ROUGE might 
not be suitable to evaluate our model directly as good paraphrases usually do not share the same vocabulary, and thus 
would attain a lower ROUGE score, despite being semantically equivalent.

Many alternative metrics are available to tackle this problem, and one of them is 
[BARTScore](https://github.com/neulab/BARTScore). BARTScore leverages a pretrained BART model for
paraphrase generation to score sentence pairs. A generated sentence is scored by the BART model based
on its probability that the model itself would generate the same sentence, gauging the quality
of the paraphrase sentence according to how much the BART model agrees with it.

Below are tabulated results of some selected models, compared to the baselines of OPT1.3B with their weights
directly fine tuned for the task of paraphrasing, and the BART model fine tuned for paraphrasing.

|                     |   soft prompt 20 tokens |   soft prompt 111 tokens |   soft prompt 163 tokens |   fine tuned |   bart fine tuned |
|:--------------------|------------------------:|-------------------------:|-------------------------:|-------------:|------------------:|
| bartscore           |               -3.02511  |                -2.15795  |                -2.19397  |   -4.32509   |        -2.65748   |
| bleuscore           |                0.246091 |                 0.342787 |                 0.316936 |    0.0251696 |         0.0833621 |
| rouge1_fmeasure     |                0.632655 |                 0.835004 |                 0.834778 |    0.315754  |         0.316741  |
| rouge1_precision    |                0.70008  |                 0.856809 |                 0.850439 |    0.304833  |         0.207854  |
| rouge1_recall       |                0.636459 |                 0.838207 |                 0.833884 |    0.374748  |         0.935199  |
| rouge2_fmeasure     |                0.538138 |                 0.737537 |                 0.721758 |    0.140419  |         0.251569  |
| rouge2_precision    |                0.590409 |                 0.756071 |                 0.734675 |    0.130611  |         0.164845  |
| rouge2_recall       |                0.540979 |                 0.743406 |                 0.722555 |    0.178818  |         0.816269  |
| rougeL_fmeasure     |                0.626995 |                 0.83046  |                 0.829546 |    0.300252  |         0.301592  |
| rougeL_precision    |                0.693667 |                 0.852231 |                 0.845049 |    0.288716  |         0.197495  |
| rougeL_recall       |                0.630616 |                 0.83334  |                 0.828588 |    0.358478  |         0.900656  |
| rougeLsum_fmeasure  |                0.626495 |                 0.830814 |                 0.82999  |    0.302298  |         0.309371  |
| rougeLsum_precision |                0.693297 |                 0.852436 |                 0.845449 |    0.290669  |         0.202609  |
| rougeLsum_recall    |                0.629847 |                 0.833801 |                 0.829088 |    0.360918  |         0.920124  |


<a id="visualisation"></a>
### Visualisation
The next step might be to visualise the meanings of the soft prompts with respect to where in the model's
embedding space they end up in, for example in the original paper it was found that clusters of nearest neighbours
maintained high lexical and semantic similarities, and that several prompt tokens end up in the vicinity of each other.

The numerical representation of word tokens are of high dimensionality, and with the specific instance of
OPT1.3B being used, has a hidden size of 2048. That is 2048 dimensions, incomprehensible to the human mind, and
while traditional methods such as PCA and TSNE can produce viable results, lots of information is lost when decomposing
a high dimensional space into 2 dimensions for us to view. In addition, the TSNE algorithm is stochastic and multiple 
restarts with different seeds can yield different embeddings, hence we have no way to directly compare two embedding
spaces.

The visualisation below is produced through the use of a data structure called a locality sensitive hash forest
and a graph visualisation tool graphistry. However, this technique does suffer from information loss
and is even stochastic to a certain extent. We mitigate this issue by utilising the fixed 
embeddings as anchor points, such that they always end up in the same position in the visualisation (determined by an
initial random seed), and then fit the learned embeddings onto the generated anchor points.

If the graph renders as a multicoloured tree, you might need to reload the page as it is a little buggy with 50k
visualisation points :). The visualisation is also available 
[here](https://hub.graphistry.com/graph/graph.html?dataset=05a0c49697bd4a5ebe88c624d709d87f&type=arrow&splashAfter=false&info=False&play=0&menu=True&showArrows=False&pointSize=0.07&edgeCurvature=0.01&edgeSize=1.0&edgeOpacity=0.5&pointOpacity=0.9&lockedX=True&lockedY=True&lockedR=False&linLog=False&strongGravity=False&dissuadeHubs=False&edgeInfluence=1.0&precisionVsSpeed=1.0&gravity=1.0&scalingRatio=0.5&showLabels=True&showLabelOnHover=True&showPointsOfInterest=False&showPointsOfInterestLabel=False&showLabelPropertiesOnHover=True&pointsOfInterestMax=0).
In the graph, red points are the default embeddings, blue points belong to the prompt of 59 prepended tokens, and green
points belong to the prompt of 111 prepended tokens.
<div>
   <iframe src="https://hub.graphistry.com/graph/graph.html?dataset=05a0c49697bd4a5ebe88c624d709d87f&type=arrow&splashAfter=false&info=False&play=0&menu=True&showArrows=False&pointSize=0.07&edgeCurvature=0.01&edgeSize=1.0&edgeOpacity=0.5&pointOpacity=0.9&lockedX=True&lockedY=True&lockedR=False&linLog=False&strongGravity=False&dissuadeHubs=False&edgeInfluence=1.0&precisionVsSpeed=1.0&gravity=1.0&scalingRatio=0.5&showLabels=True&showLabelOnHover=True&showPointsOfInterest=False&showPointsOfInterestLabel=False&showLabelPropertiesOnHover=True&pointsOfInterestMax=0" style="border: 1px solid black; width: 100%; height: 100%; min-height: 400px"></iframe>
</div>

<a id="conclusion"></a>
## Conclusion
We've taught OPT1.3B to paraphrase!

Much of the results of this implementation agree with the conclusions of the original prompt tuning paper authors. 
1. Increasing model size improves soft prompt performance.
2. Increasing the length of the soft prompts improves model performance.
3. This method largely outperforms zero-shot prompting (i.e. "paraphrase the following:"), at least when tested on
OPT1.3B.

Furthermore, some exciting facets of exploration are:
1. Training the full OPT175B model.
2. Distilling the large prepended soft prompt model into a smaller model without need for prepended prompts.
3. Prompting to achieve better chain of thought intermediate responses, thereby improving the final response.

Code is all available publicly on the github here: 
https://github.com/Clyde013/Paraphrase-OPT