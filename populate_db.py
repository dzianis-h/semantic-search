import json
import re
import time

from ai_model import get_embedding, prepare_text
from database import init_db, store_embedding

init_db()

print("Parsing movies JSON")
movies = []
with open('movies.json', 'r') as file:
    movies = json.load(file)

print("Running embedding")
start_time = round(time.time() * 1000)
for movie in movies:
    if 'description' in movie.keys():
        text = prepare_text(movie['title_be']) + " [SEP] " + prepare_text(movie['description'])
        text = re.sub(r'\s+', " ", text).strip()
        # print("Calculating embedding for text: {}".format(description))
        embedding = get_embedding(text)
        store_embedding(movie['id'], text, embedding)

embedding_time = round(time.time() * 1000) - start_time
print("Embedding took {} ms".format(embedding_time))
