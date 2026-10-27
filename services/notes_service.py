"""
Notes Service module for EduExtract AI.
Generates extractive notes using TF-IDF and TextRank, and creates formatted study notes.
"""
from typing import Dict, Any, List
from nlp.tfidf_summarizer import summarize_tfidf
from nlp.textrank_summarizer import summarize_textrank
from utils.text_utils import compute_text_stats

class NotesService:
    """Provides comparative summaries and smart notes generation."""

    @staticmethod
    def generate_notes(
        sentences: List[Dict[str, Any]],
        method: str = "tfidf",
        max_sentences: int = 8
    ) -> Dict[str, Any]:
        """
        Generate smart notes using the specified extractive summarizer.
        """
        if method.lower() == "textrank":
            selected_sents = summarize_textrank(sentences, max_sentences=max_sentences)
            method_label = "TextRank (Graph-based PageRank)"
        else:
            selected_sents = summarize_tfidf(sentences, max_sentences=max_sentences)
            method_label = "TF-IDF (Normalized Term Importance)"

        summary_text = " ".join([s["text"] for s in selected_sents])
        orig_text = " ".join([s["text"] for s in sentences])

        orig_words = len(orig_text.split())
        sum_words = len(summary_text.split())
        compression_ratio = round((1.0 - (sum_words / max(1, orig_words))) * 100, 1)

        # Group sentences by page
        pages_included = sorted(list(set(s["page_number"] for s in selected_sents)))

        return {
            "method": method.lower(),
            "method_label": method_label,
            "sentences": selected_sents,
            "summary_text": summary_text,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(selected_sents),
            "original_word_count": orig_words,
            "summary_word_count": sum_words,
            "compression_ratio_pct": compression_ratio,
            "pages_referenced": pages_included
        }

    @staticmethod
    def compare_summaries(
        sentences: List[Dict[str, Any]],
        max_sentences: int = 8
    ) -> Dict[str, Any]:
        """
        Generate both TF-IDF and TextRank summaries for direct side-by-side comparison.
        """
        tfidf_res = NotesService.generate_notes(sentences, method="tfidf", max_sentences=max_sentences)
        textrank_res = NotesService.generate_notes(sentences, method="textrank", max_sentences=max_sentences)

        # Calculate overlap in selected sentences
        tfidf_ids = set(s["sentence_id"] for s in tfidf_res["sentences"])
        tr_ids = set(s["sentence_id"] for s in textrank_res["sentences"])
        common_ids = tfidf_ids.intersection(tr_ids)
        overlap_pct = round((len(common_ids) / max(1, max_sentences)) * 100, 1)

        return {
            "tfidf": tfidf_res,
            "textrank": textrank_res,
            "sentence_overlap_count": len(common_ids),
            "sentence_overlap_percentage": overlap_pct
        }
