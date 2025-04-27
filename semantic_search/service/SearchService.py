import json
import os
import time

import requests

from semantic_search.service.EmbeddingService import EmbeddingService
from semantic_search.service.VectorRepository import VectorRepository


class SearchService:
    def __init__(self, embedding_service: EmbeddingService, vector_repository: VectorRepository):
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.movies_link = os.environ['MOVIES_LINK']

    def find_closest(self, prompt, max_results, max_distance):
        prompt_embedding = self.embedding_service.get_embedding(prompt)
        return self.vector_repository.semantic_search(prompt_embedding, max_results, max_distance)

    def reindex_all(self):
        rs = requests.get(self.movies_link)
        if rs.status_code != 200:
            print("Error getting content link, {}".format(rs.status_code))
            return
        movies = json.loads(rs.content)
        start_time = round(time.time() * 1000)
        for movie in movies:
            if 'description' in movie.keys():
                text = movie['description']
                embedding = self.embedding_service.get_embedding(text)
                self.vector_repository.store_embedding(movie['id'], text, embedding)

        embedding_time = round(time.time() * 1000) - start_time
        print("Full reindexing took {} ms".format(embedding_time))



