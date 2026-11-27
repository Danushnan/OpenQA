import nltk
from evaluate import load
from sentence_transformers import SentenceTransformer
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

rouge = load("rouge")
meteor = load('meteor')
bleurt = load('bleurt', 'bleurt-base-128')
sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalise_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    words = text.split()
    words = [word for word in words if word not in {"a", "an", "the"}]

    return " ".join(words)

def calculate_token_f1(prediction, reference):
    prediction_tokens = normalise_answer(prediction).split()
    reference_tokens = normalise_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        if prediction_tokens == reference_tokens:
            return 1
        return 0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)