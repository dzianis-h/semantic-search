import json
import time

import torch
from transformers import BertTokenizer, BertModel

model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()  # set model to eval mode?


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    outputs = model(**inputs)
    torch_embeddings = outputs.last_hidden_state[:, 0, :]
    return torch_embeddings.detach().cpu().numpy()


movies = []
with open('movies.json', 'r') as file:
    movies = json.load(file)

start_time = round(time.time() * 1000)
with torch.no_grad():
    for movie in movies:
        if 'description_short' in movie.keys():
            description_short = movie['description_short']
            print("Calculating embedding for text: {}".format(description_short))
            embeddings = get_embedding(description_short)
            # print(embeddings)

embedding_time = round(time.time() * 1000) - start_time
print("Embedding took {} ms".format(embedding_time))
