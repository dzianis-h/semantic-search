import os
import threading

from flask import Flask, request

from semantic_search.service.EmbeddingService import EmbeddingService
from semantic_search.service.SearchService import SearchService
from semantic_search.service.VectorRepository import VectorRepository

search_service = SearchService(EmbeddingService(), VectorRepository())

# search_service.reindex_all()

app = Flask(__name__)
@app.route('/semantic-search', methods=['GET'])
def semantic_search():
    prompt = request.args.get('prompt')
    max_results = request.args.get('max_results')
    if max_results is None:
        max_results = 25
    max_distance = request.args.get('max_distance')
    if max_distance is None:
        max_distance = 0.45
    results = search_service.find_closest(prompt, max_results, max_distance)
    return [to_response(x) for x in results]

@app.route('/reindex/all', methods=['GET', 'POST'])
def reindex():
    thread = threading.Thread(target=search_service.reindex_all)
    thread.start()
    return {
        'msg': 'Reindexing has started as a background job. Please, wait.',
    }



def to_response(result):
    link = "{}/movie?id={}".format(os.environ['HOST'], result[1])
    return {
        'distance': result[3],
        'chunk': result[2],
        'link': link
    }

app.run(debug=True)