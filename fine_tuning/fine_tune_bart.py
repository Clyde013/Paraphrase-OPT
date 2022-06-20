import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BartForConditionalGeneration, BartTokenizer
import os
import torch
from torch.optim import Adam

# current working directory changes when imported from other modules, so to ensure para_nmt_path is correct we store
# the absolute path to the module for reference.
package_directory = os.path.dirname(os.path.abspath(__file__))


class FineTuneBART(LightningModule):
    bart_path = os.path.join(package_directory, "bart.pth")

    def __init__(self, model_name='facebook/bart-large-cnn'):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.save_hyperparameters()

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = self.bart_path
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def forward(self, **inputs):
        return self.model(**inputs)


if __name__ == "__main__":
    model = FineTuneBART()
    model.load()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    ARTICLE_TO_PARAPHRASE = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )

    while ARTICLE_TO_PARAPHRASE != "":
        inputs = tokenizer([ARTICLE_TO_PARAPHRASE], max_length=1024, truncation=True, return_tensors="pt")

        # Generate Summary
        summary_ids = model.model.generate(inputs["input_ids"], num_beams=2, do_sample=True, min_length=0, max_length=50)
        outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(outputs)

        ARTICLE_TO_PARAPHRASE = input("Enter: ")
