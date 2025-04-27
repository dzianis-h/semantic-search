import os

from flask import Flask, request

from semantic_search.service.EmbeddingService import EmbeddingService
from semantic_search.service.SearchService import SearchService
from semantic_search.service.VectorRepository import VectorRepository

search_service = SearchService(EmbeddingService(), VectorRepository())

# indexing_service.reindex_all()
# for result in indexing_service.find_closest("увесь свет гэта іллюзія, чалавецтва б'ецца з машынамі"):
#     print("{}/movie?id={} \tcosine similarity = {}".format(os.environ['HOST'], result[1], result[2]))


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


def to_response(result):
    link = "{}/movie?id={}".format(os.environ['HOST'], result[1])
    return {
        'link': link,
        'distance': result[2]
    }

app.run(debug=True)