from transformers import BertTokenizer, BertModel

print("Setting-up AI Model")
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()  # set model to eval mode?


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=40)
    outputs = model(**inputs)
    torch_embeddings = outputs.last_hidden_state[:, 0, :]
    return torch_embeddings.detach().cpu().numpy()[0].tolist()
