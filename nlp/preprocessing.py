"""
Text Preprocessing module for EduExtract AI.
Provides normalization, tokenization, stopword removal, lemmatization, and POS tagging.
"""
import re
import unicodedata
from typing import List, Dict, Any, Optional
from utils.logging_config import get_logger

logger = get_logger("preprocessing")

# Standard English stopwords fallback list
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

_spacy_nlp = None

def get_spacy_nlp():
    """Lazily load spacy pipeline."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("en_core_web_sm not found, loading blank English model")
                _spacy_nlp = spacy.blank("en")
        except ImportError:
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None


def normalize_text(text: str) -> str:
    """
    Normalize text encoding, strip excessive whitespace, and standardize quotes/hyphens
    without destroying mathematical expressions, formulas, or abbreviations.
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Replace stylized quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace non-breaking spaces
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Normalize tabs and spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize newline sequences
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into alphanumeric words and meaningful terms.
    Preserves words with internal hyphens or apostrophes.
    """
    if not text:
        return []
    target = text.lower() if lowercase else text
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:['_-][a-zA-Z0-9]+)*\b", target)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out stopwords from a token list while preserving word case."""
    if not tokens:
        return []
    return [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize tokens using spaCy if available, or lightweight morphological suffix rules.
    """
    if not tokens:
        return []

    nlp = get_spacy_nlp()
    if nlp is not None and "tagger" in nlp.pipe_names:
        doc = nlp(" ".join(tokens))
        return [tok.lemma_.lower() for tok in doc if not tok.is_space]

    # Rule-based fallback lemmatization
    lemmatized = []
    for token in tokens:
        w = token.lower()
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("ves") and len(w) > 4:
            w = w[:-3] + "f"
        elif w.endswith("es") and len(w) > 4 and w[-3] in "shxz":
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        elif w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        lemmatized.append(w)
    return lemmatized


def pos_tag_text(text: str) -> List[Dict[str, str]]:
    """
    Return POS tags for words in the text.
    """
    if not text:
        return []

    nlp = get_spacy_nlp()
    if nlp is not None and "tagger" in nlp.pipe_names:
        doc = nlp(text)
        return [{"token": tok.text, "lemma": tok.lemma_, "pos": tok.pos_, "tag": tok.tag_} for tok in doc if not tok.is_space]

    # Fallback heuristic tagging
    tokens = tokenize_text(text, lowercase=False)
    results = []
    for t in tokens:
        pos = "NOUN"
        low = t.lower()
        if low in STOPWORDS:
            pos = "DET" if low in ("the", "a", "an") else "PRON" if low in ("he", "she", "it", "they") else "ADP"
        elif low.endswith("ly"):
            pos = "ADV"
        elif low.endswith(("able", "ible", "ous", "ful", "less", "ive", "al")):
            pos = "ADJ"
        elif low.endswith(("ize", "ise", "ate", "ify")):
            pos = "VERB"
        results.append({"token": t, "lemma": low, "pos": pos, "tag": pos})
    return results


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Full preprocessing pipeline.
    Preserves original text, computes normalized text, tokens, clean tokens, lemmas,
    POS tags, and vocabulary statistics.
    """
    normalized = normalize_text(text)
    raw_tokens = tokenize_text(normalized, lowercase=False)
    lower_tokens = [t.lower() for t in raw_tokens]
    filtered_tokens = remove_stopwords(lower_tokens)
    lemmas = lemmatize_tokens(filtered_tokens)
    pos_tags = pos_tag_text(normalized)

    # Vocabulary frequency
    vocab_freq: Dict[str, int] = {}
    for lemma in lemmas:
        vocab_freq[lemma] = vocab_freq.get(lemma, 0) + 1

    sorted_vocab = sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True)

    return {
        "original_text": text,
        "normalized_text": normalized,
        "token_count": len(raw_tokens),
        "unique_token_count": len(set(lower_tokens)),
        "filtered_token_count": len(filtered_tokens),
        "tokens": raw_tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags[:100],  # sample for inspection
        "top_vocabulary": sorted_vocab[:30],
        "stopwords_identified_count": len(lower_tokens) - len(filtered_tokens)
    }
