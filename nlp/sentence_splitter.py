"""
Sentence Splitter module for EduExtract AI.
Splits text into sentences while preserving page numbers, abbreviations, and citations.
"""
import re
from typing import List, Dict, Any

# Common abbreviations to prevent premature sentence splitting
COMMON_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.",
    "fig.", "figs.", "al.", "inc.", "ltd.", "co.", "dept.", "approx.", "est.", "vol.",
    "no.", "p.", "pp.", "u.s.", "u.k.", "u.n.", "e.u."
}

def split_into_sentences_raw(text: str) -> List[str]:
    """
    Split text into clean sentence strings using regex with abbreviation guards.
    """
    if not text or not text.strip():
        return []

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace single line-breaks within paragraphs with spaces, preserve double line-breaks
    paragraphs = re.split(r'\n\s*\n', text)
    sentences: List[str] = []

    for p in paragraphs:
        cleaned_para = re.sub(r'\s+', ' ', p).strip()
        if not cleaned_para:
            continue

        # Regex split on period, exclamation, or question mark followed by space and capital letter or quote
        tokens = re.split(r'([.?!]+["\'\s]+)', cleaned_para)
        current_sentence = ""
        
        i = 0
        while i < len(tokens):
            chunk = tokens[i]
            current_sentence += chunk
            
            # Check if this chunk contains terminal punctuation
            if i + 1 < len(tokens):
                delimiter = tokens[i + 1]
                current_sentence += delimiter
                
                # Check for abbreviation or decimal number
                words = current_sentence.strip().split()
                last_word = words[-1].lower() if words else ""
                
                # If last word is an abbreviation or ends in a number with dot, don't break
                is_abbrev = (last_word in COMMON_ABBREVIATIONS or 
                             re.search(r'\b[A-Za-z]\.$', last_word) or 
                             re.search(r'\d+\.$', last_word))
                
                if not is_abbrev:
                    sent = current_sentence.strip()
                    if len(sent) > 5 and re.search(r'[a-zA-Z]', sent):
                        sentences.append(sent)
                    current_sentence = ""
                i += 2
            else:
                i += 1

        if current_sentence.strip():
            sent = current_sentence.strip()
            if len(sent) > 5 and re.search(r'[a-zA-Z]', sent):
                sentences.append(sent)

    return sentences


def split_document_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split a list of page dicts [{"page_number": int, "original_text": str}]
    into a list of sentence dicts with page provenance:
    [
        {
            "sentence_id": int,
            "text": str,
            "page_number": int,
            "char_length": int,
            "word_count": int
        }
    ]
    """
    all_sentences: List[Dict[str, Any]] = []
    sentence_counter = 0

    for page in pages:
        page_num = page.get("page_number", 1)
        text = page.get("original_text", "")
        if not text:
            continue

        raw_sents = split_into_sentences_raw(text)
        for s in raw_sents:
            words = re.findall(r'\b\w+\b', s)
            if len(words) < 3:
                # Skip trivial fragments like table headers or stray numbers
                continue

            all_sentences.append({
                "sentence_id": sentence_counter,
                "text": s,
                "page_number": page_num,
                "char_length": len(s),
                "word_count": len(words)
            })
            sentence_counter += 1

    return all_sentences
