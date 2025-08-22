# src/model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    model = AutoModelForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=1)
    return tokenizer, model
