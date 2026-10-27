"""
Text Similarity module for EduExtract AI.
Provides TF-IDF Cosine Similarity, Jaccard Similarity, and Token Overlap
for evidence grounding and algorithm comparison.
"""
import re
from typing import Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp.preprocessing import normalize_text, tokenize_text, STOPWORDS

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity between two text snippets."""
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([text1, text2])
        score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return round(score, 4)
    except Exception:
        # If vocabulary is empty after stopwords
        return calculate_jaccard_similarity(text1, text2)

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token-level Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    tokens1: Set[str] = set(tokenize_text(text1)) - STOPWORDS
    tokens2: Set[str] = set(tokenize_text(text2)) - STOPWORDS
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return round(len(intersection) / len(union), 4)

def calculate_token_overlap(answer: str, source: str) -> float:
    """Compute the fraction of significant answer tokens present in source text."""
    ans_tokens = set(tokenize_text(answer)) - STOPWORDS
    src_tokens = set(tokenize_text(source)) - STOPWORDS
    if not ans_tokens:
        return 1.0 if answer.strip().lower() in source.strip().lower() else 0.0
    overlap = ans_tokens.intersection(src_tokens)
    return round(len(overlap) / len(ans_tokens), 4)
