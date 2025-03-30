from transformers import BertTokenizer, BertModel

print("Setting-up AI Model")
model_name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()  # set model to eval mode?


# model_name = 'KoichiYasuoka/roberta-small-belarusian'
# tokenizer=AutoTokenizer.from_pretrained(model_name)
# model=AutoModelForMaskedLM.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text.lower(), return_tensors='pt', padding=True, truncation=True, max_length=2048)
    outputs = model(**inputs)
    torch_embeddings = outputs.last_hidden_state[:, 0, :]
    return torch_embeddings.detach().cpu().numpy()[0].tolist()
