from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from transformers import pipeline

# Download stopwords once
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

app = FastAPI()

# Vectorizer
vectorizer = CountVectorizer(
    ngram_range=(2, 2),  # Changed n-gram range to 2 or more (bigrams)
    stop_words='english',
    min_df=1,
    max_features=500
)

# NER Model
ner_model = pipeline('ner', grouped_entities=True)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Generate N-grams
def generate_ngrams(text, top_n=10):
    text = clean_text(text)
    X = vectorizer.fit_transform([text])
    ngram_counts = X.toarray().sum(axis=0)
    ngram_features = vectorizer.get_feature_names_out()
    freq_distribution = dict(zip(ngram_features, ngram_counts))
    sorted_ngrams = sorted(freq_distribution.items(), key=lambda x: x[1], reverse=True)
    return sorted_ngrams[:top_n]

# Perform NER
def perform_ner(text):
    ner_results = ner_model(text)
    return [(entity['entity_group'], entity['word']) for entity in ner_results]

# Request Body Model
class TextRequest(BaseModel):
    text: str
    top_n: Optional[int] = 10

# N-grams Endpoint
@app.post("/ngrams")
async def get_ngrams(request: TextRequest):
    ngrams = generate_ngrams(request.text, request.top_n)
    return {"ngrams": ngrams}

# NER Endpoint
@app.post("/ner")
async def get_ner(request: TextRequest):
    ner = perform_ner(request.text)
    return {"entities": ner}

# Both N-grams and NER
@app.post("/analyze")
async def analyze(request: TextRequest):
    ngrams = generate_ngrams(request.text, request.top_n)
    ner = perform_ner(request.text)
    return {
        "ngrams": ngrams,
        "entities": ner
    }
