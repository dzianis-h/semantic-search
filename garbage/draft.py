import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


input_texts = ['query: фільм пра тое што мы жывем у капутарнай самуляцыі',
               'query: усё, што нас акружае - не больш, чым ілюзія, Матрыца, а людзі - усяго толькі крыніца харчавання для штучнага інтэлекту, які заняволіў чалавецтва.',
               "query: матрыца з Нэа",
               "query: жэццё ў беверлі хіллс",
               "query: аднойчы ў менску",
               "query: Жыццё Томаса Андэрсана падзелена на дзве часткі: днём ён самы звычайны офісны працоўнік, а ўначы ператвараецца ў хакера па імені Нэа, і няма месца ў сеціве, куды ён не змог бы дацягнуцца. Але аднойчы ўсё змяняецца - герой, сам таго не жадаючы, пазнае жахлівую праўду: усё, што яго акружае - не больш, чым ілюзія, Матрыца, а людзі - усяго толькі крыніца харчавання для штучнага інтэлекту, які заняволіў чалавецтва. І толькі Нэа можа змяніць расстаноўку сіл у гэтым чужым і жахлівым свеце."]

# input_texts = [ 'query: фильм про то как мы живём в компьютерной симуляции',
#     'query: Все вокруг нас — не более чем иллюзия, Матрица, а люди — всего лишь источник силы для искусственного интеллекта, поработившего человечество.',
#     'query: Матрица с Нео',
#     'query: жизнь в беверли хиллс',
#     'query: однажды в минске',
#     'query: Жизнь Томаса Андерсона делится на две части: днем ​​он обычный офисный работник, а ночью превращается в хакера по имени Нео, и нет такого места в Интернете, куда бы он не смог добраться. Но однажды все меняется — герой, сам того не желая, узнаёт страшную правду: все, что его окружает, — не более чем иллюзия, Матрица, а люди — всего лишь источник силы для искусственного интеллекта, поработившего человечество. И только Нео может изменить расстановку сил в этом странном и страшном мире.']


# [[79.96796417236328], [88.13963317871094], [82.03987884521484], [73.81964111328125], [74.64693450927734]]
# [[80.86103057861328], [85.57841491699219], [82.09412384033203], [77.31060791015625], [79.15103149414062]]
model_name = 'intfloat/multilingual-e5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print('karamba')
# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:5] @ embeddings[5:].T) * 100
print(scores.tolist())
