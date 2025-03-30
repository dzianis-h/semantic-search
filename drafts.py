import json
import time

import torch

from ai_model import get_embedding
from database import init_db, store_embedding

init_db()

print("Parsing movies JSON")
movies = []
with open('movies.json', 'r') as file:
    movies = json.load(file)

print("Running embedding")
start_time = round(time.time() * 1000)
with torch.no_grad():
    for movie in movies:
        if 'description_short' in movie.keys():
            description_short = movie['description_short']
            # print("Calculating embedding for text: {}".format(description_short))
            store_embedding(description_short, get_embedding(description_short))

embedding_time = round(time.time() * 1000) - start_time
print("Embedding took {} ms".format(embedding_time))
