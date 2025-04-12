import os

from ai_model import get_embedding, prepare_text
from database import semantic_search


def print_search_results(query):
    embedding = get_embedding(prepare_text(query))
    results = semantic_search(embedding)
    for result in results:
        print("https://{}/movie?id={} \tcosine similarity = {}".format(os.environ['HOST'], result[1], result[3]))


print_search_results(
    "усё, што нас акружае - не больш, чым ілюзія, Матрыца, а людзі - усяго толькі крыніца харчавання для штучнага інтэлекту, які заняволіў чалавецтва.")
