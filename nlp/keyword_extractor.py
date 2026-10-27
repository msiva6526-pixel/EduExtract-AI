"""
Keyword and Keyphrase Extractor module for EduExtract AI.
Implements TF-IDF scoring, linguistic noun-phrase extraction, and RAKE-style multi-word keyphrases.
Tracks keywords by PDF page and identifies surrounding context sentences.
"""
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp.preprocessing import STOPWORDS, get_spacy_nlp
from utils.logging_config import get_logger

logger = get_logger("keyword_extractor")

class KeywordExtractor:
    """Extracts keywords and keyphrases using statistical and linguistic methods."""

    def __init__(self, max_keywords: int = 15):
        self.max_keywords = max_keywords

    def extract_tfidf_keywords(
        self,
        sentences: List[Dict[str, Any]],
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords ranked by TF-IDF scores across document sentences.
        Sentences act as local documents to measure specificity and frequency.
        """
        if not sentences:
            return []

        corpus = [s["text"] for s in sentences if s.get("text")]
        if not corpus:
            return []

        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.85,
                min_df=1,
                token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = tfidf_matrix.mean(axis=0).A1  # average TF-IDF across sentences

            scored_terms = []
            for idx, score in enumerate(mean_scores):
                term = feature_names[idx]
                if score > 0:
                    scored_terms.append((term, float(score)))

            scored_terms.sort(key=lambda x: x[1], reverse=True)

            results: List[Dict[str, Any]] = []
            for term, score in scored_terms[:top_n]:
                # Find matching sentences and pages
                matched_pages = set()
                evidence_sentences = []
                pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)

                for s in sentences:
                    if pattern.search(s.get("text", "")):
                        matched_pages.add(s.get("page_number", 1))
                        if len(evidence_sentences) < 2:
                            evidence_sentences.append(s.get("text", ""))

                results.append({
                    "keyword": term,
                    "score": round(score, 4),
                    "method": "TF-IDF",
                    "occurrences": len(matched_pages),
                    "pages": sorted(list(matched_pages)),
                    "evidence_sentences": evidence_sentences
                })

            return results
        except Exception as e:
            logger.warning("TF-IDF keyword extraction error: %s", e)
            return []

    def extract_linguistic_keyphrases(
        self,
        sentences: List[Dict[str, Any]],
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Extract meaningful keyphrases using noun-phrase patterns:
        (ADJ)* (NOUN)+ or spaCy noun chunks.
        """
        nlp = get_spacy_nlp()
        full_text = " ".join([s.get("text", "") for s in sentences])
        phrase_counter: Counter = Counter()
        phrase_to_sents: Dict[str, List[Dict[str, Any]]] = {}

        if nlp is not None and "parser" in nlp.pipe_names:
            doc = nlp(full_text[:200000])  # limit to safe length
            for chunk in doc.noun_chunks:
                clean_chunk = chunk.text.strip().lower()
                # Exclude pronouns, single-character phrases, and pure stopwords
                words = clean_chunk.split()
                if (len(words) >= 1 and len(words) <= 3 and 
                    clean_chunk not in STOPWORDS and 
                    len(clean_chunk) > 3 and 
                    not clean_chunk.isnumeric()):
                    # Strip leading determiners
                    if words[0] in ("the", "a", "an", "this", "that", "these", "those"):
                        clean_chunk = " ".join(words[1:])
                    if clean_chunk and clean_chunk not in STOPWORDS:
                        phrase_counter[clean_chunk] += 1
        else:
            # Fallback regex pattern for adjective-noun sequences:
            # (Adjective)* Noun+
            words = re.findall(r'\b[a-zA-Z]+\b', full_text)
            for i in range(len(words) - 1):
                w1, w2 = words[i].lower(), words[i+1].lower()
                if w1 not in STOPWORDS and w2 not in STOPWORDS and len(w1) > 2 and len(w2) > 2:
                    phrase = f"{w1} {w2}"
                    phrase_counter[phrase] += 1

        results: List[Dict[str, Any]] = []
        for phrase, freq in phrase_counter.most_common(top_n):
            matched_pages = set()
            evidence = []
            pat = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
            for s in sentences:
                if pat.search(s.get("text", "")):
                    matched_pages.add(s.get("page_number", 1))
                    if len(evidence) < 2:
                        evidence.append(s.get("text", ""))

            results.append({
                "keyword": phrase,
                "score": float(freq),
                "method": "Linguistic Noun Phrase",
                "occurrences": freq,
                "pages": sorted(list(matched_pages)),
                "evidence_sentences": evidence
            })

        return results

    def extract_all(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run all keyword extraction methods and provide a unified ranked view."""
        tfidf_kw = self.extract_tfidf_keywords(sentences, top_n=self.max_keywords)
        ling_kw = self.extract_linguistic_keyphrases(sentences, top_n=self.max_keywords)

        # Keyword distribution by page
        page_distribution: Dict[int, List[str]] = {}
        for item in tfidf_kw + ling_kw:
            kw = item["keyword"]
            for pg in item["pages"]:
                page_distribution.setdefault(pg, []).append(kw)

        # Deduplicate per page
        page_distribution = {pg: sorted(list(set(kws))) for pg, kws in page_distribution.items()}

        return {
            "tfidf_keywords": tfidf_kw,
            "linguistic_keyphrases": ling_kw,
            "page_distribution": page_distribution
        }
