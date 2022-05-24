from datasets import Dataset, load_dataset

'''
README from parabank-2.0.zip

The TSV file contains ParaBank 2, a diverse collection of paraphrases generated
through bilingual generation. Details of the dataset and how it's created can
be found here:

Hu, J. E., A. Singh, N. Holzenberger, M. Post, & B. Van Durme. 2019. Large-scale,
Diverse, Paraphrastic Bitexts via Sampling and Clustering. Proceedings of CoNLL 2019,
Hong Kong, Nov 3 – Nov 4, 2019.

Each line of the file contains a bilingual dual-condition score, a reference
sentence, and paraphrases of the same reference sentence. A reference sentence may
have between one to five distinct paraphrases. The lines are in descending
order of the dual-conditioned score, a measurement of the quality of the
original bilingual sentence pair. Within the same line, paraphrases are ranked by
model score as described in the paper - i.e., the first paraphrase from left
to right correspond to the system with subscript "1" in evaluation, and the
last to "5". All sentences are raw text (untokenized). The reference sentences
appear in ascending order of their bidirectional model scores (the lower the
better), which we use to filter the bilingual resource used to generate ParaBank 2.
'''


def init_parabank_dataset(seed=69):
    """Initialises the parabank-2.0 paraphrase dataset in streaming mode.

    Parameters
    ----------
    seed: int
        seed for dataset shuffle order.

    Returns
    -------
    IterableDataset
    """

    url = "http://cs.jhu.edu/~vandurme/data/parabank-2.0.zip"

    def preprocess(examples):
        # list of len batch
        batch = examples['text']
        batch = [x.split('\t')[1:] for x in batch]
        examples['text'] = batch
        return examples

    dataset = load_dataset("text", data_files={"train": url}, streaming=True)['train']
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = dataset.map(preprocess, batched=True)

    return dataset


dataset = init_parabank_dataset()
print(list(dataset.take(5)))