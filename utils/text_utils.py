"""
Text utility functions for EduExtract AI.
"""
import re
import math
from typing import Dict, Any, List

def clean_whitespace(text: str) -> str:
    """Normalize multiple spaces, tabs, and newlines into clean readable text."""
    if not text:
        return ""
    # Replace non-breaking spaces and exotic whitespace
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace excessive line breaks (more than 2) with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def compute_text_stats(text: str, tokens: List[str] = None, sentences: List[str] = None) -> Dict[str, Any]:
    """Compute basic readability and descriptive statistics of text."""
    clean = clean_whitespace(text)
    char_count = len(clean)
    words = tokens if tokens is not None else re.findall(r'\b\w+\b', clean.lower())
    word_count = len(words)
    unique_words = len(set(words))
    
    sent_count = len(sentences) if sentences is not None else max(1, len(re.split(r'[.!?]+', clean)))
    avg_sentence_len = round(word_count / max(1, sent_count), 2)
    lexical_diversity = round((unique_words / max(1, word_count)), 3)
    
    # Simple estimate of reading time (200 words per minute)
    reading_time_mins = round(word_count / 200, 1)

    return {
        "character_count": char_count,
        "word_count": word_count,
        "unique_word_count": unique_words,
        "sentence_count": sent_count,
        "average_sentence_length": avg_sentence_len,
        "lexical_diversity": lexical_diversity,
        "estimated_reading_time_minutes": reading_time_mins
    }

def truncate_text(text: str, max_chars: int = 120, suffix: str = "...") -> str:
    """Truncate text safely at word boundary."""
    if not text or len(text) <= max_chars:
        return text or ""
    truncated = text[:max_chars].rsplit(' ', 1)[0]
    return truncated + suffix
