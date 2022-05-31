"""
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
"""
from typing import Optional
from datasets import load_dataset, IterableDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling


class ParabankDataModule(LightningDataModule):
    """
    LightningDataModule for parabank dataset for causal language modelling

    Note on num_workers: https://github.com/huggingface/datasets/pull/4375
    IterableDatasets do not support Dataloaders with num_workers > 0. Watch the PR to see if the fix will be merged.
    """
    parabank_url = "http://cs.jhu.edu/~vandurme/data/parabank-2.0.zip"

    def __init__(self, opt_name, batch_size, steps_per_epoch, num_workers=0, seed=69):
        """

        Parameters
        ----------
        opt_name: str
            name of the OPT model type (i.e. facebook/opt-350m)
        batch_size: int
            batch_size output by dataloader
        steps_per_epoch: int
            dataset_size = steps_per_epoch * batch_size
            Since we do not know the dataset size we simply leave it to the user to determine how many steps per epoch
            we should have.
        num_workers: int
            refer to note above on PR https://github.com/huggingface/datasets/pull/4375
        seed: int
            haha funny number
        """

        super().__init__()
        self.opt_name = opt_name
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.seed = seed

        # init None to make pycharm happy
        self.tokenizer = None
        self.dataset = None

    def prepare_data(self) -> None:
        # download and cache
        GPT2Tokenizer.from_pretrained(self.opt_name)

    def setup(self, stage: Optional[str] = None) -> None:
        # load tokenizer (should be cached)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.opt_name, use_fast=False)

        # preprocess function for the dataset's entries
        def preprocess(examples):
            # list of len batch
            batch = examples['text']
            processed_batch = list()
            for i in batch:
                # split by \t (it is a tsv file) and omit the initial dual-condition score (it is useless)
                i = i.split('\t')[1:]
                # filter entries without paraphrases and split them with a '</s>' token to denote source-target
                if len(i) > 1:
                    processed_batch.append(i[0] + self.tokenizer.eos_token + i[1])

            outputs = self.tokenizer(
                processed_batch,
                truncation=True,
                max_length=69,
            )
            return outputs

        # init dataset in streaming mode
        self.dataset = load_dataset("text", data_files=self.parabank_url, streaming=True)['train']
        # elements within buffer size will be shuffled as they are loaded in
        self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=10_000)
        # preprocessing will take place while being streamed by dataloader
        self.dataset = self.dataset.map(preprocess, batched=True, remove_columns=['text'])
        # ensure pytorch tensors are returned
        self.dataset = self.dataset.with_format("torch")

        # monkeypatch of __len__ function in the dataloader so that the trainer knows how many
        # steps there are per epoch. Sure this violates many programming paradigms but it works.
        n = self.steps_per_epoch
        def __len__(self):
            return n

        IterableDataset.__len__ = __len__

    # dataloaders are basically all the same since we cannot split a streamed dataset
    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                          num_workers=self.num_workers)
