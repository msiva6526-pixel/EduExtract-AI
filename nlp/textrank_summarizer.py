"""
TextRank Extractive Summarization module for EduExtract AI.
Constructs a graph of sentence nodes connected by pairwise cosine similarity weights,
applies Google PageRank via NetworkX, and selects key sentences while preserving order and source page provenance.
"""
from typing import List, Dict, Any
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_config import get_logger

logger = get_logger("textrank_summarizer")

def summarize_textrank(
    sentences: List[Dict[str, Any]],
    max_sentences: int = 8,
    similarity_threshold: float = 0.05,
    damping_factor: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Generate an extractive summary using the TextRank graph algorithm.

    Parameters:
        sentences: List of sentence dictionaries containing:
                   {"sentence_id": int, "text": str, "page_number": int}
        max_sentences: Maximum number of sentences to return.
        similarity_threshold: Minimum cosine similarity edge weight to form a graph connection.
        damping_factor: Damping factor for PageRank (standard is 0.85).

    Returns:
        List of selected sentence dictionaries with TextRank scores, restored to original document order.
    """
    if not sentences:
        return []

    valid_sentences = [
        s for s in sentences
        if s.get("text") and len(s["text"].split()) >= 4
    ]

    if not valid_sentences:
        valid_sentences = [s for s in sentences if s.get("text") and s["text"].strip()]

    if not valid_sentences:
        return []

    if len(valid_sentences) <= max_sentences:
        return [
            {
                **s,
                "score": 1.0,
                "method": "TextRank Extractive",
                "original_order": idx
            }
            for idx, s in enumerate(valid_sentences)
        ]

    corpus = [s["text"] for s in valid_sentences]

    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Build NetworkX graph
        graph = nx.Graph()
        n = len(valid_sentences)
        for i in range(n):
            graph.add_node(i)

        for i in range(n):
            for j in range(i + 1, n):
                weight = float(sim_matrix[i][j])
                if weight > similarity_threshold:
                    graph.add_edge(i, j, weight=weight)

        # Compute PageRank
        try:
            scores = nx.pagerank(graph, alpha=damping_factor, weight='weight', max_iter=200)
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality if convergence struggles
            scores = nx.degree_centrality(graph)

        scored_sentences = []
        for idx, s in enumerate(valid_sentences):
            score = scores.get(idx, 0.0)
            scored_sentences.append({
                "sentence_id": s.get("sentence_id", idx),
                "text": s["text"],
                "page_number": s.get("page_number", 1),
                "score": round(float(score), 5),
                "original_order": idx,
                "method": "TextRank Extractive"
            })

        # Rank by score descending
        ranked = sorted(scored_sentences, key=lambda x: x["score"], reverse=True)

        # Select top non-redundant sentences
        selected = []
        selected_indices = set()

        for item in ranked:
            if len(selected) >= max_sentences:
                break
            idx = item["original_order"]
            # Redundancy check against already selected
            is_redundant = False
            for sel_idx in selected_indices:
                if sim_matrix[idx][sel_idx] > 0.80:
                    is_redundant = True
                    break
            if not is_redundant:
                selected.append(item)
                selected_indices.add(idx)

        # Restore original document order
        selected.sort(key=lambda x: x["original_order"])

        return selected

    except Exception as e:
        logger.error("TextRank summarization failed: %s", e)
        # Fallback to initial sentences
        return [
            {
                **s,
                "score": 0.5,
                "method": "TextRank Extractive (Fallback)",
                "original_order": idx
            }
            for idx, s in enumerate(valid_sentences[:max_sentences])
        ]
