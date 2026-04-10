from functools import lru_cache
from transformers import pipeline
@lru_cache()
def predict(text):
    classifier = pipeline(task = "sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english",local_files_only=True)
    return classifier(text)
