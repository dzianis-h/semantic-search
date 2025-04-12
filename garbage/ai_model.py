import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizer, RobertaModel, \
    RobertaConfig
import re

print("Setting-up AI Model")
# model_name = 'bert-base-multilingual-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "KoichiYasuoka/deberta-base-belarusian"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, output_hidden_states=True)

# model_name = 'KoichiYasuoka/roberta-small-belarusian-ud-goeswith'
# config = RobertaConfig.from_pretrained(model_name)
# config.output_hidden_states = True
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaModel.from_pretrained(model_name)

model.eval()  # set model to eval mode?


def prepare_text(text):
    text = split_chars(text.lower(), ".,!?'\"-«»‚‘’„“”").strip()
    return re.sub(r'\s+', " ", text).strip()



# TODO: hidden layer
# import torch.nn.functional as F
#
# # Compute attention scores (importance of each token)
# attention_weights = F.softmax(last_hidden_layer.mean(dim=-1), dim=-1)  # (batch_size, seq_len)
# sentence_embedding = torch.sum(last_hidden_layer * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)



def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        # torch_embedding = outputs.pooler_output[0, :]
        # torch_embedding = outputs[-1][0].mean(1)[0, :]
        torch_embedding = outputs.hidden_states[-1].mean(1)[0, :]
        return torch_embedding.detach().cpu().numpy().tolist()


def split_chars(text, chars):
    for char in chars:
        text = text.replace(char, " " + char + " ")
    return text
