import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizer, RobertaModel, \
    RobertaConfig
import re

print("Setting-up AI Model")
# model_name = 'bert-base-multilingual-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

model_name = 'KoichiYasuoka/roberta-small-belarusian'
config = RobertaConfig.from_pretrained("KoichiYasuoka/roberta-small-belarusian")
config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaModel.from_pretrained(model_name)

model.eval()  # set model to eval mode?

def prepare_text(text):
    text = split_chars(text.lower(), ".,!?'\"-«»‚‘’„“”").strip()
    return re.sub(r'\s+', " ", text).strip()

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=127)
        outputs = model(**inputs)
        # torch_embedding = outputs.pooler_output[0, :]
        torch_embedding =  outputs[-1][0].mean(1)[0, :]
        return torch_embedding.detach().cpu().numpy().tolist()

def split_chars(text, chars):
    for char in chars:
        text = text.replace(char, " " + char + " ")
    return text