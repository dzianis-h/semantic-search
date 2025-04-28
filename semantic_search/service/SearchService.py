import json
import os
import time
from datetime import timedelta

import requests

from semantic_search.service.EmbeddingService import EmbeddingService
from semantic_search.service.VectorRepository import VectorRepository


class SearchService:
    def __init__(self, embedding_service: EmbeddingService, vector_repository: VectorRepository):
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.movies_link = os.environ['MOVIES_LINK']

    def find_closest(self, prompt: str, max_results, max_distance):
        prompt_embedding = self.embedding_service.get_embeddings(list(prompt))[0]
        return self.vector_repository.semantic_search(prompt_embedding, max_results, max_distance)

    def reindex_all(self):
        start_time = time.time()
        print('Staring full reindexing....')
        rs = requests.get(self.movies_link)
        if rs.status_code != 200:
            print("Error getting content link, {}".format(rs.status_code))
            return
        movies = json.loads(rs.content)
        self.vector_repository.truncate_embeddings()
        for movie in movies:
            title_keys = ['title_page', 'title_be', 'title_en', 'title_ru']
            titles = ''
            for key in title_keys:
                if key in movie.keys():
                    titles = titles + movie[key] + ' | '
                    break

            keys = ['description', 'description_short']
            prompts = []
            used_keys = []
            for key in keys:
                if key in movie.keys():
                    prompt = movie[key]
                    if len(prompt) > 35:
                        if not prompts.__contains__(prompt):
                            prompts.append(prompt)
                            used_keys.append(key)
                        prompt = titles + prompt
                        if not prompts.__contains__(prompt):
                            prompts.append(prompt)
                            used_keys.append('titles_' + key)

            embeddings = self.embedding_service.get_embeddings(prompts)
            self.vector_repository.store_embeddings(movie['id'], used_keys, prompts, embeddings)

        reindexing_time = timedelta(seconds=time.time() - start_time)
        print("Full reindexing took {}".format(str(reindexing_time)))


