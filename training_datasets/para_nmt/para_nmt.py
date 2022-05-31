from datasets import load_dataset, IterableDataset

para_nmt_path = r"para-nmt-5m-processed.zip"

dataset = load_dataset("text", data_files=para_nmt_path, streaming=True)['train']
print(list(dataset.take(10)))
