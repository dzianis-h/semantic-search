import re
from typing import List

from sentence_transformers import SentenceTransformer


# model = SentenceTransformer('intfloat/multilingual-e5-large')
# input_texts = [
#     'query: how much protein should a female eat',
#     'query: 南瓜的家常做法',
#     "query: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 i     s 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or traini     ng for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "query: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮     ,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,     放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油     锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀      6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
# ]
# embeddings = model.encode(input_texts, normalize_embeddings=True)


class EmbeddingService:
    def __init__(self):
        self.model_name = 'intfloat/multilingual-e5-large'
        self.model = SentenceTransformer(self.model_name)
        self.model.eval()

    def get_embeddings(self, text_list: List[str]) -> List[List[float]]:
        text_list = [EmbeddingService.prepare_text(txt) for txt in text_list]
        embeddings = self.model.encode(text_list, normalize_embeddings=True)
        return embeddings.tolist()

    @staticmethod
    def prepare_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', " ", text).strip()
        text = 'query: ' + text
        return text
