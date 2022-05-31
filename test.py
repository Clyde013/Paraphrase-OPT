import torch

from transformers import GPT2Tokenizer
from soft_prompt_tuning.soft_prompt_opt import ParaphraseOPT

print("Initialising...")

checkpoint = r"training_checkpoints/30-05-2022-1.3b/soft-opt-epoch=179-val_loss=1.397.ckpt"
model_name = "facebook/opt-1.3b"

torch.cuda.empty_cache()

AVAIL_GPUS = min(1, torch.cuda.device_count())

model = ParaphraseOPT.load_from_custom_save(model_name, checkpoint)
model = model.eval()

default_model = ParaphraseOPT(model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

"""
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

#compare_models(default_model.model, model.model)

datamodule = ParabankDataModule(model_name, 1, 1000, seed=1337)
datamodule.setup()
dl = datamodule.val_dataloader()
it = iter(dl)
"""
"""
print("----- MANUAL GENERATION -------")



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
"""
while True:
    prompt = input("Prompt: ")

    print("------ SIMPLE PROMPT ---------")
    simple_prompt = prompt + ". paraphrase: "
    print(simple_prompt)
    inputs = tokenizer(simple_prompt, return_tensors="pt")
    outputs = default_model.model.generate(inputs.input_ids, max_length=45, use_cache=False)
    decoded = tokenizer.batch_decode(outputs)[0]
    print(decoded)

    print("------ AUTOMATIC GENERATION -------")
    soft_prompt = prompt + "</s>"
    print(soft_prompt)
    inputs = tokenizer(soft_prompt, return_tensors="pt")
    outputs = model.model.generate(inputs.input_ids, max_length=45, use_cache=False)
    decoded = tokenizer.batch_decode(outputs)[0]
    print(decoded)

    print("-------- END ----------")
