from typing import List, Type, Optional

from datasets import IterableDataset, interleave_datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling

from training_datasets.parabank.parabank import ParabankDataModule
from training_datasets.para_nmt.para_nmt import ParaNMTDataModule


class ParaCombinedDataModule(LightningDataModule):
    """
    LightningDataModule for combining different datasets for causal language modelling

    Note on num_workers: https://github.com/huggingface/datasets/pull/4375
    IterableDatasets do not support Dataloaders with num_workers > 0. Watch the PR to see if the fix will be merged.
    """
    def __init__(self, opt_name, batch_size, steps_per_epoch, datamodules: List[Type[LightningDataModule]],
                 probabilities: List[float], num_workers=0, seed=69):
        """

        Parameters
        ----------
        opt_name: str
            Name of model type
        batch_size: int
            batch_size output by dataloader
        steps_per_epoch: int
            dataset_size = steps_per_epoch * batch_size
            Since we do not know the dataset size we simply leave it to the user to determine how many steps per epoch
            we should have.
        datamodules: List[Type[LightningDataModule]]
            List specifying the datamodules whose datasets will be interleaved
        probabilities: List[float]
            List of probabilities for respective datamodules that should sum to 1
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
        self.datamodules = datamodules
        self.probabilities = probabilities
        self.tokenizer = None
        self.dataset = None

        # sanity check
        assert sum(self.probabilities) == 1, "Probabilities for interleaved datasets do not sum to 1.0"

    def prepare_data(self) -> None:
        # download and cache
        GPT2Tokenizer.from_pretrained(self.opt_name)

    def setup(self, stage: Optional[str] = None) -> None:
        # tokenizer is not actually used once instantiated but to stay consistent with other datamodule implementations
        # we instantiate it anyway
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.opt_name, use_fast=False)

        # instantiate all the datamodules and extract the dataset from them
        datasets = list()
        for datamodule in self.datamodules:
            dm = datamodule(self.opt_name, self.batch_size, self.steps_per_epoch, self.seed)
            dm.setup()
            datasets.append(dm.dataset)

        self.dataset = interleave_datasets(datasets, probabilities=self.probabilities, seed=self.seed)
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


if __name__ == "__main__":
    model_name = "facebook/opt-1.3b"
    datamodule = ParaCombinedDataModule(model_name, 1, 1000, [ParabankDataModule, ParaNMTDataModule],
                                        probabilities=[0.35, 0.65], seed=1337)
    datamodule.setup()
    dl = datamodule.val_dataloader()
    it = iter(dl)

    for i in range(10):
        print(datamodule.tokenizer.batch_decode(next(it)['input_ids'])[0])
