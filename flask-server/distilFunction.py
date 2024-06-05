import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

model_name = "/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Models/final/Distilbert/genrepredict"  # Path to the saved model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

label_map = {
    0: 'action', 1: 'adult', 2: 'adventure', 3: 'animation', 4: 'biography', 5: 'comedy', 6: 'crime',
    7: 'documentary', 8: 'drama', 9: 'family', 10: 'fantasy', 11: 'game-show', 12: 'history', 13: 'horror',
    14: 'music', 15: 'musical', 16: 'mystery', 17: 'news', 18: 'reality-tv', 19: 'romance', 20: 'sci-fi',
    21: 'short', 22: 'sport', 23: 'talk-show', 24: 'thriller', 25: 'war', 26: 'western'
}

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits).item()
    predicted_class = label_map[predicted_class_idx]
    return predicted_class