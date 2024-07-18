import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



class Model:
    def __init__(self):
        model_name = r'bert_model\trained_model'
        tokenizer_name = r'bert_model\trained_tokenizer'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def classify_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        return predicted_class_id

    def classify_dataframe(self, text):
        predictions = []
        predicted_class_id = self.classify_text(text)
        predictions.append(predicted_class_id)
        return predictions