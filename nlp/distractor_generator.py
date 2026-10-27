"""
Distractor Generator module for EduExtract AI.
Generates plausible, evidence-grounded incorrect options for Multiple Choice Questions (MCQs)
using entity matching, concept clusters, and document vocabulary.
"""
import random
import re
from typing import List, Dict, Any, Optional
from nlp.preprocessing import STOPWORDS

class DistractorGenerator:
    """Generates contextually grounded distractors for MCQs without hallucinating arbitrary options."""

    def __init__(self, entities_by_category: Dict[str, List[Dict[str, Any]]] = None, keywords: List[Dict[str, Any]] = None):
        self.entities_by_category = entities_by_category or {}
        self.keywords = [k.get("keyword", "") for k in (keywords or []) if k.get("keyword")]

    def generate_distractors(
        self,
        correct_answer: str,
        target_count: int = 3,
        entity_type: Optional[str] = None,
        candidate_pool: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate plausible distractors for a given correct answer.
        Ensures distractors are distinct, non-empty, and not equal to the correct answer.
        """
        correct_clean = correct_answer.strip().lower()
        distractors: List[str] = []
        seen = {correct_clean}

        # 1. Strategy 1: Match by entity category (highest semantic plausibility)
        if entity_type and entity_type in self.entities_by_category:
            cat_candidates = [
                e["text"] for e in self.entities_by_category[entity_type]
                if e["text"].strip().lower() not in seen and len(e["text"].strip()) > 1
            ]
            for cand in cat_candidates:
                cand_clean = cand.strip()
                if cand_clean.lower() not in seen:
                    distractors.append(cand_clean)
                    seen.add(cand_clean.lower())
                if len(distractors) >= target_count:
                    break

        # 2. Strategy 2: Use caller provided candidate pool
        if len(distractors) < target_count and candidate_pool:
            for cand in candidate_pool:
                cand_clean = cand.strip()
                if cand_clean and cand_clean.lower() not in seen and len(cand_clean) > 1:
                    distractors.append(cand_clean)
                    seen.add(cand_clean.lower())
                if len(distractors) >= target_count:
                    break

        # 3. Strategy 3: Check across all extracted entities
        if len(distractors) < target_count:
            for cat, ent_list in self.entities_by_category.items():
                for e in ent_list:
                    cand = e["text"].strip()
                    if cand and cand.lower() not in seen and len(cand) > 2:
                        distractors.append(cand)
                        seen.add(cand.lower())
                    if len(distractors) >= target_count:
                        break
                if len(distractors) >= target_count:
                    break

        # 4. Strategy 4: Use extracted document keywords
        if len(distractors) < target_count and self.keywords:
            for kw in self.keywords:
                cand = kw.strip().title()
                if cand and cand.lower() not in seen and len(cand) > 2:
                    distractors.append(cand)
                    seen.add(cand.lower())
                if len(distractors) >= target_count:
                    break

        # Return only up to target count
        return distractors[:target_count]
