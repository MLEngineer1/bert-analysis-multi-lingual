from fastapi import FastAPI, Query
from pydantic import BaseModel
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re
from transformers import pipeline
from typing import List, Dict

# Download stopwords once
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Initialize FastAPI
app = FastAPI()

# Initialize vectorizer and NER model
vectorizer = CountVectorizer(
    ngram_range=(1, 2), stop_words='english', min_df=1, max_features=500
)
ner_model = pipeline('ner', grouped_entities=True)

# Clean text function
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Generate N-grams
def generate_ngrams(text: str, top_n: int = 5) -> List[Dict[str, int]]:
    text = clean_text(text)
    X = vectorizer.fit_transform([text])
    ngram_counts = X.toarray().sum(axis=0)
    ngram_features = vectorizer.get_feature_names_out()
    freq_distribution = dict(zip(ngram_features, ngram_counts))
    sorted_ngrams = sorted(freq_distribution.items(), key=lambda x: x[1], reverse=True)
    return [{"ngram": k, "frequency": v} for k, v in sorted_ngrams[:top_n]]

# Perform NER
def perform_ner(text: str) -> List[Dict[str, str]]:
    ner_results = ner_model(text)
    return [{"entity": entity['entity_group'], "text": entity['word']} for entity in ner_results]

# API Request Model
class AnalyzeRequest(BaseModel):
    text: str
    top_n: int = 5

# API Endpoint
@app.get("/analyze")
def analyze(text: str = Query(..., title="Text to analyze"), top_n: int = 5):
    return {
        "ngrams": generate_ngrams(text, top_n=top_n),
        "ner": perform_ner(text)
    }

