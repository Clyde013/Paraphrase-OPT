from soft_prompt_opt import SoftOPTModelWrapper
from training_datasets.parabank import init_parabank_dataset

from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, OPTForCausalLM

from tqdm import tqdm

import torch.optim as optim
import torch
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

model = SoftOPTModelWrapper.from_pretrained("facebook/opt-350m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
dataset = init_parabank_dataset(tokenizer).with_format("torch")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

optimizer = optim.Adam(model.soft_embedding.wte.parameters())
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=1)
model.train().to(device)

for epoch in range(3):
    dataset.set_epoch(epoch)
    for i, batch in enumerate(tqdm(data_loader, total=5)):
        if i == 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"loss: {loss}")
