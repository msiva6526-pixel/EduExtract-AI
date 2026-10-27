"""
TF-IDF Extractive Summarization module for EduExtract AI.
Scores sentences based on normalized term-weight importance and selects the most salient sentences,
preserving original document order and source page provenance.
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logging_config import get_logger

logger = get_logger("tfidf_summarizer")

def summarize_tfidf(
    sentences: List[Dict[str, Any]],
    max_sentences: int = 8
) -> List[Dict[str, Any]]:
    """
    Generate an extractive summary using TF-IDF sentence scoring.

    Parameters:
        sentences: List of sentence dictionaries containing:
                   {"sentence_id": int, "text": str, "page_number": int}
        max_sentences: Maximum number of sentences to include in the summary.

    Returns:
        List of selected sentence dictionaries with scores, sorted in original document order.
    """
    if not sentences:
        return []

    # Filter out empty or extremely short sentences
    valid_sentences = [
        s for s in sentences
        if s.get("text") and len(s["text"].split()) >= 4
    ]

    if not valid_sentences:
        valid_sentences = [s for s in sentences if s.get("text") and s["text"].strip()]

    if not valid_sentences:
        return []

    # If document has fewer or equal sentences than target, return all
    if len(valid_sentences) <= max_sentences:
        return [
            {
                **s,
                "score": 1.0,
                "method": "TF-IDF Extractive",
                "original_order": idx
            }
            for idx, s in enumerate(valid_sentences)
        ]

    corpus = [s["text"] for s in valid_sentences]

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Sentence score = sum of TF-IDF weights divided by log-adjusted word count (to prevent favoring solely long sentences)
        scored_sentences = []
        for idx, s in enumerate(valid_sentences):
            row = tfidf_matrix.getrow(idx)
            word_count = len(s["text"].split())
            if row.nnz > 0:
                raw_score = float(row.data.sum())
                # Length normalization penalty/boost
                norm_score = raw_score / (np.log1p(word_count) + 1e-5)
            else:
                norm_score = 0.0

            scored_sentences.append({
                "sentence_id": s.get("sentence_id", idx),
                "text": s["text"],
                "page_number": s.get("page_number", 1),
                "score": round(norm_score, 4),
                "original_order": idx,
                "method": "TF-IDF Extractive"
            })

        # Rank by score descending
        ranked = sorted(scored_sentences, key=lambda x: x["score"], reverse=True)

        # Select top max_sentences
        selected = ranked[:max_sentences]

        # Restore original document order
        selected.sort(key=lambda x: x["original_order"])

        return selected

    except Exception as e:
        logger.error("TF-IDF summarization failed: %s", e)
        # Fallback: take leading sentences
        return [
            {
                **s,
                "score": 0.5,
                "method": "TF-IDF Extractive (Fallback)",
                "original_order": idx
            }
            for idx, s in enumerate(valid_sentences[:max_sentences])
        ]
