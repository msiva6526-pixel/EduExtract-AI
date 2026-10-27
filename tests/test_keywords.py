"""
Unit tests for keyword and keyphrase extraction.
"""
from nlp.keyword_extractor import KeywordExtractor

def test_keyword_extraction():
    sentences = [
        {"sentence_id": 0, "text": "Deep learning models require neural networks with many layers.", "page_number": 1},
        {"sentence_id": 1, "text": "Neural networks learn hierarchical representations of training data.", "page_number": 1},
        {"sentence_id": 2, "text": "Supervised learning uses labeled datasets to train models.", "page_number": 2},
        {"sentence_id": 3, "text": "Neural networks achieve state of the art accuracy on computer vision tasks.", "page_number": 2}
    ]

    extractor = KeywordExtractor(max_keywords=5)
    results = extractor.extract_all(sentences)

    tfidf_kws = results["tfidf_keywords"]
    assert len(tfidf_kws) > 0

    # 'neural' or 'networks' should be high ranking
    kw_names = [k["keyword"].lower() for k in tfidf_kws]
    assert any("neural" in k or "network" in k for k in kw_names)

    # Check page tracking
    first_kw = tfidf_kws[0]
    assert "pages" in first_kw
    assert len(first_kw["pages"]) > 0
    assert "evidence_sentences" in first_kw
    assert len(first_kw["evidence_sentences"]) > 0
