import torch
from pytorch_lightning import Trainer

from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT
from training_datasets.parabank import ParabankDataModule

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

datamodule = ParabankDataModule("facebook/opt-350m", 32)
datamodule.setup()

model = ParaphraseOPT()

print("TRAINING MODEL")
trainer = Trainer(max_epochs=5, gpus=AVAIL_GPUS)
trainer.fit(model, datamodule=datamodule)

print("VALIDATING MODEL")
trainer.validate(model, datamodule.val_dataloader())

'''
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
model = SoftOPTModelWrapper.from_pretrained("facebook/opt-350m")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)

# generate function
print("generate function")
outputs = model.generate(**inputs, max_length=35, use_cache=False)
print(outputs)
print(outputs.shape)
decoded = tokenizer.batch_decode(outputs)
print(decoded)

# manual generation
print("manual generation")
print(inputs)
print(inputs['input_ids'].shape)
generate_ids = model(**inputs)
logits = generate_ids.logits.detach().cpu()
print(logits)
print(logits.shape)
next_token_logits = logits[:, -1, :]
print(next_token_logits)
print(next_token_logits.shape)
next_token = torch.argmax(next_token_logits, dim=-1)
print(next_token)
print(tokenizer.batch_decode(torch.unsqueeze(next_token, 0)))
'''
