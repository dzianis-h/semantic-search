import re

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class EmbeddingService:
    def __init__(self):
        model_name = "KoichiYasuoka/deberta-base-belarusian"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def get_embedding(self, text):
        text = EmbeddingService.prepare_text(text)  # ???
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**inputs)
            torch_embedding = outputs.hidden_states[-1].mean(1)[0, :]
            return torch_embedding.detach().cpu().numpy().tolist()

    @staticmethod
    def prepare_text(text):
        text = text.lower()
        text = EmbeddingService.split_chars(text, ".,!?\"-«»‚‘’„“”").strip()
        return re.sub(r'\s+', " ", text).strip()

    @staticmethod
    def split_chars(text, chars):
        for char in chars:
            text = text.replace(char, " " + char + " ")
        return text
