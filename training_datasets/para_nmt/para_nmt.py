import os
from typing import Optional
from datasets import load_dataset, IterableDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling

# current working directory changes when imported from other modules, so to ensure para_nmt_path is correct we store
# the absolute path to the module for reference.
package_directory = os.path.dirname(os.path.abspath(__file__))


class ParaNMTDataModule(LightningDataModule):
    """
    LightningDataModule for para_nmt dataset for causal language modelling

    Note on num_workers: https://github.com/huggingface/datasets/pull/4375
    IterableDatasets do not support Dataloaders with num_workers > 0. Watch the PR to see if the fix will be merged.
    """
    para_nmt_path = os.path.join(package_directory, "para-nmt-5m-processed.zip")

    def __init__(self, opt_name, batch_size, steps_per_epoch, num_workers=0, seed=69, pre_tokenize=True):
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
        pre_tokenize: bool
            should we tokenize the texts (if true: dataset will return tokenized ids instead of source text)
        """

        super().__init__()
        self.opt_name = opt_name
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.seed = seed
        self.pre_tokenize = pre_tokenize

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
                # replace the \t splitting with a '</s>' token to denote source-target
                processed_batch.append(str.replace(i, "\t", self.tokenizer.eos_token))

            if self.pre_tokenize:
                outputs = self.tokenizer(
                    processed_batch,
                    truncation=True,
                    max_length=69,
                )
            else:
                outputs = {"source": processed_batch}
            return outputs

        # init dataset in streaming mode
        self.dataset = load_dataset("text", data_files=self.para_nmt_path, streaming=True)['train']
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
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers)
        if self.pre_tokenize: dataloader.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers)
        if self.pre_tokenize: dataloader.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers)
        if self.pre_tokenize: dataloader.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers)
        if self.pre_tokenize: dataloader.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return dataloader


if __name__ == "__main__":
    model_name = "facebook/opt-1.3b"
    datamodule = ParaNMTDataModule(model_name, 1, 1000, seed=1337)
    datamodule.setup()
    dl = datamodule.val_dataloader()
    it = iter(dl)

    for i in range(10):
        print(datamodule.tokenizer.batch_decode(next(it)['input_ids'])[0])
