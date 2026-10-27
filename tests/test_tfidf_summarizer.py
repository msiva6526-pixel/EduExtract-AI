"""
Unit tests for TF-IDF extractive summarization.
"""
from nlp.tfidf_summarizer import summarize_tfidf

def test_tfidf_summarizer():
    sentences = [
        {"sentence_id": 0, "text": "Photosynthesis is the biological process by which green plants produce glucose.", "page_number": 1},
        {"sentence_id": 1, "text": "Sunlight provides radiant energy required for photochemical reactions.", "page_number": 1},
        {"sentence_id": 2, "text": "Hello there.", "page_number": 1},  # short, should be filtered or low score
        {"sentence_id": 3, "text": "Chloroplast organelles contain chlorophyll pigments that absorb blue and red light.", "page_number": 2},
        {"sentence_id": 4, "text": "Oxygen gas is released into the atmosphere as a metabolic byproduct.", "page_number": 2},
        {"sentence_id": 5, "text": "ATP molecules store chemical potential energy within cellular systems.", "page_number": 3}
    ]

    summary = summarize_tfidf(sentences, max_sentences=3)
    assert len(summary) <= 3
    assert len(summary) > 0

    # Ensure original document order is preserved
    orders = [s["original_order"] for s in summary]
    assert orders == sorted(orders)

    # Ensure page numbers and scores are retained
    for s in summary:
        assert "page_number" in s
        assert s["page_number"] in (1, 2, 3)
        assert "score" in s
        assert s["score"] >= 0.0

def test_tfidf_empty_input():
    assert summarize_tfidf([]) == []
    assert summarize_tfidf([{"sentence_id": 0, "text": "a b c", "page_number": 1}]) != []
