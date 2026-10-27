"""
Named Entity Recognition module for EduExtract AI.
Uses spaCy en_core_web_sm pipeline to extract and categorize entities with page provenance.
"""
from typing import List, Dict, Any, Set
from nlp.preprocessing import get_spacy_nlp
from utils.logging_config import get_logger

logger = get_logger("entity_extractor")

# Friendly labels for entity categories
ENTITY_CATEGORY_MAP = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations / Geopolitics",
    "LOC": "Locations",
    "DATE": "Dates & Events",
    "TIME": "Time",
    "MONEY": "Financial",
    "PERCENT": "Percentages",
    "CARDINAL": "Numerical Quantities",
    "QUANTITY": "Measurements",
    "EVENT": "Historical Events",
    "WORK_OF_ART": "Works of Art / Publications",
    "LAW": "Laws & Treaties",
    "FAC": "Facilities / Buildings",
    "NORP": "Nationalities & Groups"
}

class EntityExtractor:
    """Extracts named entities from document sentences and organizes by category."""

    def __init__(self):
        self.nlp = get_spacy_nlp()

    def extract_entities(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract named entities from sentences.
        Returns a dictionary of categorized entities, entity frequencies, and page references.
        """
        if not self.nlp or "ner" not in self.nlp.pipe_names:
            logger.warning("spaCy NER model not available.")
            return {
                "available": False,
                "message": "spaCy English NER model (en_core_web_sm) is not loaded.",
                "entities_by_category": {},
                "unique_entities": [],
                "entity_page_map": {}
            }

        entities_by_category: Dict[str, Dict[str, Dict[str, Any]]] = {}
        entity_page_map: Dict[str, Set[int]] = {}

        for sent in sentences:
            text = sent.get("text", "")
            page_num = sent.get("page_number", 1)
            doc = self.nlp(text)

            for ent in doc.ents:
                clean_ent = ent.text.strip()
                label = ent.label_

                # Skip single punctuation or trivial fragments
                if len(clean_ent) < 2 or clean_ent.isspace():
                    continue

                category = ENTITY_CATEGORY_MAP.get(label, label)
                cat_dict = entities_by_category.setdefault(category, {})

                if clean_ent not in cat_dict:
                    cat_dict[clean_ent] = {
                        "text": clean_ent,
                        "label": label,
                        "category": category,
                        "count": 1,
                        "pages": {page_num},
                        "evidence_sentences": [text]
                    }
                else:
                    cat_dict[clean_ent]["count"] += 1
                    cat_dict[clean_ent]["pages"].add(page_num)
                    if len(cat_dict[clean_ent]["evidence_sentences"]) < 3 and text not in cat_dict[clean_ent]["evidence_sentences"]:
                        cat_dict[clean_ent]["evidence_sentences"].append(text)

                entity_page_map.setdefault(clean_ent, set()).add(page_num)

        # Convert sets to sorted lists for serialization
        formatted_categories: Dict[str, List[Dict[str, Any]]] = {}
        all_unique: List[Dict[str, Any]] = []

        for category, ents in entities_by_category.items():
            ent_list = []
            for item in ents.values():
                entry = {
                    "text": item["text"],
                    "label": item["label"],
                    "category": item["category"],
                    "count": item["count"],
                    "pages": sorted(list(item["pages"])),
                    "evidence_sentences": item["evidence_sentences"]
                }
                ent_list.append(entry)
                all_unique.append(entry)

            ent_list.sort(key=lambda x: x["count"], reverse=True)
            formatted_categories[category] = ent_list

        all_unique.sort(key=lambda x: x["count"], reverse=True)

        return {
            "available": True,
            "message": "NER model loaded successfully.",
            "entities_by_category": formatted_categories,
            "unique_entities": all_unique,
            "entity_page_map": {k: sorted(list(v)) for k, v in entity_page_map.items()}
        }
