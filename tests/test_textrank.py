"""
Unit tests for TextRank extractive summarization.
"""
from nlp.textrank_summarizer import summarize_textrank

def test_textrank_summarizer():
    sentences = [
        {"sentence_id": 0, "text": "Machine learning algorithms build mathematical models based on training sample data.", "page_number": 1},
        {"sentence_id": 1, "text": "Supervised algorithms use labeled training data to make predictions or decisions.", "page_number": 1},
        {"sentence_id": 2, "text": "Unsupervised learning discovers hidden patterns or data groupings without labeled responses.", "page_number": 2},
        {"sentence_id": 3, "text": "Reinforcement learning agents take actions in an environment to maximize cumulative reward.", "page_number": 2},
        {"sentence_id": 4, "text": "Deep neural networks are a subset of machine learning using multi-layer artificial neurons.", "page_number": 3}
    ]

    summary = summarize_textrank(sentences, max_sentences=2)
    assert len(summary) == 2

    # Check order preservation
    orders = [s["original_order"] for s in summary]
    assert orders == sorted(orders)

    # Check score and provenance
    for s in summary:
        assert s["score"] > 0.0
        assert "page_number" in s
        assert s["method"] == "TextRank Extractive"

def test_textrank_redundancy_filter():
    sentences = [
        {"sentence_id": 0, "text": "Neural networks are trained using stochastic gradient descent optimization.", "page_number": 1},
        {"sentence_id": 1, "text": "Neural networks are trained using stochastic gradient descent optimization.", "page_number": 1},  # duplicate
        {"sentence_id": 2, "text": "Computer vision models classify images into distinct semantic categories.", "page_number": 2}
    ]

    summary = summarize_textrank(sentences, max_sentences=2)
    # The duplicate should be filtered by redundancy check
    selected_texts = [s["text"] for s in summary]
    assert len(set(selected_texts)) == len(selected_texts)
