"""
Question Generator module for EduExtract AI.
Generates Multiple Choice Questions (MCQs), Fill-in-the-Blank (Cloze),
and Short Answer questions using explainable rule-based pattern matching and entity cues.
"""
import re
import uuid
import random
from typing import List, Dict, Any, Optional
from nlp.distractor_generator import DistractorGenerator
from utils.logging_config import get_logger

logger = get_logger("question_generator")

class QuestionGenerator:
    """Generates educational questions grounded in source sentences and page provenance."""

    def __init__(self, distractor_gen: Optional[DistractorGenerator] = None):
        self.distractor_gen = distractor_gen or DistractorGenerator()

    def generate_definition_questions(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect definition patterns such as:
        - 'X is defined as Y'
        - 'X refers to Y'
        - 'X is the / a / an process that / by which Y'
        - 'X is a / an Y that...'
        """
        questions: List[Dict[str, Any]] = []

        patterns = [
            # Pattern 1: X is defined as Y
            (r'^(?:The\s+)?([A-Z][a-zA-Z0-9\s_-]{2,35})\s+is\s+defined\s+as\s+(.+)$', "definition"),
            # Pattern 2: X refers to Y
            (r'^(?:The\s+)?([A-Z][a-zA-Z0-9\s_-]{2,35})\s+refers\s+to\s+(.+)$', "reference"),
            # Pattern 3: X is a/an/the ... that/which ...
            (r'^(?:The\s+)?([A-Z][a-zA-Z0-9\s_-]{2,35})\s+is\s+(the\s+(?:process|method|system|framework|concept|technique|property|component|structure|organelle|algorithm)\s+(?:by\s+which|that|which)\s+.+)$', "concept_is"),
            # Pattern 4: X consists of Y
            (r'^(?:The\s+)?([A-Z][a-zA-Z0-9\s_-]{2,35})\s+consists\s+of\s+(.+)$', "consists_of")
        ]

        for s in sentences:
            text = s.get("text", "").strip()
            page_num = s.get("page_number", 1)

            for regex, pat_type in patterns:
                m = re.match(regex, text, re.IGNORECASE)
                if m:
                    subject = m.group(1).strip()
                    definition = m.group(2).strip().rstrip('.')

                    if len(subject) < 3 or len(definition) < 10 or subject.lower() in ("this", "it", "they", "these", "there"):
                        continue

                    # 1. Multiple Choice version
                    distractors = self.distractor_gen.generate_distractors(
                        correct_answer=subject,
                        target_count=3,
                        entity_type="Concepts"
                    )

                    if len(distractors) >= 2:
                        options = [subject] + distractors
                        random.shuffle(options)
                        questions.append({
                            "question_id": f"mcq-def-{uuid.uuid4().hex[:8]}",
                            "question_type": "mcq",
                            "question": f"Which term is described as: \"{definition.capitalize()}\"?",
                            "options": options,
                            "correct_answer": subject,
                            "explanation": f"According to page {page_num}: \"{text}\"",
                            "source_page": page_num,
                            "source_sentence": text,
                            "generation_method": f"definition_pattern ({pat_type})"
                        })

                    # 2. Short answer version
                    questions.append({
                        "question_id": f"sa-def-{uuid.uuid4().hex[:8]}",
                        "question_type": "short_answer",
                        "question": f"What is {subject}?",
                        "options": [],
                        "correct_answer": definition,
                        "explanation": f"Source reference (Page {page_num}): \"{text}\"",
                        "source_page": page_num,
                        "source_sentence": text,
                        "generation_method": f"definition_inversion ({pat_type})"
                    })
                    break

        return questions

    def generate_fill_in_the_blank(
        self,
        sentences: List[Dict[str, Any]],
        keywords: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create cloze/fill-in-the-blank questions by blanking out key extracted concepts or entities.
        """
        questions: List[Dict[str, Any]] = []
        target_words = [k["keyword"].strip() for k in keywords if len(k.get("keyword", "")) > 3]

        for s in sentences:
            text = s.get("text", "").strip()
            page_num = s.get("page_number", 1)

            # Avoid very long or short sentences
            words = text.split()
            if len(words) < 7 or len(words) > 35:
                continue

            for kw in target_words:
                pat = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
                match = pat.search(text)
                if match:
                    matched_term = match.group(1)
                    # Replace first occurrence with blank line
                    blanked_sentence = pat.sub("_______", text, count=1)

                    # Distractors for an MCQ variation of FITB
                    distractors = self.distractor_gen.generate_distractors(
                        correct_answer=matched_term,
                        target_count=3
                    )
                    options = [matched_term] + distractors if len(distractors) >= 2 else []
                    if options:
                        random.shuffle(options)

                    questions.append({
                        "question_id": f"fitb-{uuid.uuid4().hex[:8]}",
                        "question_type": "fill_in_the_blank",
                        "question": f"Fill in the blank: {blanked_sentence}",
                        "options": options,
                        "correct_answer": matched_term,
                        "explanation": f"Source sentence from page {page_num}: \"{text}\"",
                        "source_page": page_num,
                        "source_sentence": text,
                        "generation_method": "keyword_cloze"
                    })
                    break  # One question per sentence

        return questions

    def generate_entity_questions(
        self,
        sentences: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate questions around recognized named entities (People, Locations, Dates, Orgs).
        """
        questions: List[Dict[str, Any]] = []

        for ent in entities:
            ent_text = ent.get("text", "").strip()
            label = ent.get("label", "")
            cat = ent.get("category", "")
            evidence_list = ent.get("evidence_sentences", [])

            if not evidence_list or len(ent_text) < 2:
                continue

            sentence = evidence_list[0]
            # Find page for this sentence
            page_num = 1
            for s in sentences:
                if s.get("text") == sentence:
                    page_num = s.get("page_number", 1)
                    break

            # Template selection based on entity type
            template_q = None
            if label == "PERSON":
                template_q = f"Which notable person is associated with the following statement: \"{sentence}\"?"
            elif label in ("GPE", "LOC"):
                template_q = f"Which location or place is referenced in this context: \"{sentence}\"?"
            elif label == "ORG":
                template_q = f"Which organization is identified in this passage: \"{sentence}\"?"
            elif label in ("DATE", "TIME"):
                template_q = f"When did the following take place: \"{sentence}\"?"

            if template_q:
                distractors = self.distractor_gen.generate_distractors(
                    correct_answer=ent_text,
                    target_count=3,
                    entity_type=cat
                )
                if len(distractors) >= 2:
                    options = [ent_text] + distractors
                    random.shuffle(options)
                    questions.append({
                        "question_id": f"mcq-ent-{uuid.uuid4().hex[:8]}",
                        "question_type": "mcq",
                        "question": template_q,
                        "options": options,
                        "correct_answer": ent_text,
                        "explanation": f"Directly mentioned on page {page_num}: \"{sentence}\"",
                        "source_page": page_num,
                        "source_sentence": sentence,
                        "generation_method": f"entity_recognition ({label})"
                    })

        return questions

    def generate_all(
        self,
        sentences: List[Dict[str, Any]],
        keywords: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        mcq_count: int = 5,
        fitb_count: int = 5,
        short_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate a balanced mix of question types grounded in evidence."""
        def_qs = self.generate_definition_questions(sentences)
        fitb_qs = self.generate_fill_in_the_blank(sentences, keywords)
        ent_qs = self.generate_entity_questions(sentences, entities)

        all_mcqs = [q for q in def_qs + ent_qs if q["question_type"] == "mcq"]
        all_short = [q for q in def_qs if q["question_type"] == "short_answer"]
        all_fitb = fitb_qs

        # Randomize selection
        random.shuffle(all_mcqs)
        random.shuffle(all_fitb)
        random.shuffle(all_short)

        selected: List[Dict[str, Any]] = (
            all_mcqs[:mcq_count] +
            all_fitb[:fitb_count] +
            all_short[:short_count]
        )

        return selected
