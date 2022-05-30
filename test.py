import torch

from transformers import GPT2Tokenizer
from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT
from training_datasets.parabank import ParabankDataModule

checkpoint = r"training_checkpoints/soft-opt-epoch=000-val_loss=14.535.ckpt"

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

model = ParaphraseOPT.load_from_custom_save(checkpoint)
model = model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")


print("-------- MODEL COMPARISON -----------")


# thanks https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


default_model = ParaphraseOPT()
compare_models(default_model.model, model.model)

print("----- MANUAL GENERATION -------")

datamodule = ParabankDataModule("facebook/opt-350m", 1, 1000, seed=986624)
datamodule.setup()
dl = datamodule.val_dataloader()
it = iter(dl)

prompt = tokenizer.batch_decode(next(it).input_ids)[0]
print(prompt)
prompt = str.join("", prompt.split("</s>")[1])
prompt = prompt + "</s>"
print(prompt)

for i in range(10):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu()
    pred_token = torch.argmax(logits[:, -1])
    decoded = tokenizer.batch_decode(torch.unsqueeze(pred_token, 0))
    prompt = prompt + decoded[0]

print(prompt)

print("------ AUTOMATIC GENERATION -------")

prompt = tokenizer.batch_decode(next(it).input_ids)[0]
print(prompt)
prompt = str.join("", prompt.split("</s>")[1])
prompt = prompt + "</s>"
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.model.generate(inputs.input_ids, max_length=40, use_cache=False)
decoded = tokenizer.batch_decode(outputs)[0]
print(decoded)
