"""
Unit tests for text preprocessing, tokenization, lemmatization, and stopwords.
"""
from nlp.preprocessing import (
    normalize_text,
    tokenize_text,
    remove_stopwords,
    lemmatize_tokens,
    preprocess_text
)

def test_normalize_text():
    raw = "  “Smart   Notes”  and  ‘NLP’ — a   test\n\n\n\nwith  spaces.  "
    norm = normalize_text(raw)
    assert norm == '"Smart Notes" and \'NLP\' - a test\n\nwith spaces.'

def test_tokenize_text():
    text = "Photosynthesis-driven organisms absorb CO2 and H2O."
    tokens = tokenize_text(text, lowercase=True)
    assert "photosynthesis-driven" in tokens or "photosynthesis" in tokens
    assert "organisms" in tokens
    assert "co2" in tokens

def test_remove_stopwords():
    tokens = ["the", "mitochondria", "is", "the", "powerhouse", "of", "a", "cell"]
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "is" not in filtered
    assert "of" not in filtered
    assert "mitochondria" in filtered
    assert "powerhouse" in filtered
    assert "cell" in filtered

def test_lemmatize_tokens():
    tokens = ["plants", "cells", "processes", "organelles"]
    lemmas = lemmatize_tokens(tokens)
    assert "plant" in lemmas
    assert "cell" in lemmas
    assert "process" in lemmas
    assert "organelle" in lemmas

def test_preprocess_text_stats():
    sample = "Natural Language Processing enables computers to understand human language. It involves syntax and semantics."
    result = preprocess_text(sample)
    assert result["token_count"] > 10
    assert result["filtered_token_count"] < result["token_count"]
    assert "top_vocabulary" in result
    assert result["stopwords_identified_count"] > 0
