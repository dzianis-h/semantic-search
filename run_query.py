from ai_model import get_embedding
from database import semantic_search

embedding = get_embedding("усё, што нас акружае - не больш, чым ілюзія, Матрыца, а людзі - усяго толькі крыніца харчавання для штучнага інтэлекту, які заняволіў чалавецтва.")

results = semantic_search(embedding)
for result in results:
    print("https://site/movie?id={} \tcosine similarity = {}".format(result[1], result[3]))