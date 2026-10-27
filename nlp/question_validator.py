"""
Question Validator and Evidence Grounding module for EduExtract AI.
Validates generated questions for correctness, source evidence grounding,
distractor uniqueness, and answer plausibility.
"""
from typing import List, Dict, Any, Tuple
from nlp.text_similarity import calculate_token_overlap, calculate_cosine_similarity
from utils.logging_config import get_logger

logger = get_logger("question_validator")

class QuestionValidator:
    """Performs rigorous multi-point validation on all generated questions."""

    def __init__(self, min_token_overlap: float = 0.60):
        self.min_token_overlap = min_token_overlap

    def validate_question(self, question_item: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        Validate a single question dictionary.
        Returns: (is_valid: bool, status: 'validated' | 'needs_review' | 'rejected', reason: str)
        """
        q_text = question_item.get("question", "").strip()
        ans = question_item.get("correct_answer", "").strip()
        source = question_item.get("source_sentence", "").strip()
        q_type = question_item.get("question_type", "")
        options = question_item.get("options", [])
        page = question_item.get("source_page", 1)

        # 1. Non-empty check
        if not q_text or len(q_text) < 5:
            return False, "rejected", "Question text is empty or too short."
        if not ans or len(ans) < 1:
            return False, "rejected", "Correct answer is empty."

        # 2. Source page validity
        if not isinstance(page, int) or page < 1:
            return False, "rejected", "Invalid source page number."

        # 3. Source evidence grounding
        # Check if answer is in source text or has high token overlap
        ans_in_source = ans.lower() in source.lower()
        overlap_score = calculate_token_overlap(ans, source)

        if not ans_in_source and overlap_score < self.min_token_overlap:
            return False, "rejected", f"Answer not grounded in evidence sentence (overlap={overlap_score:.2f})."

        # 4. MCQ-specific checks
        if q_type == "mcq":
            if not options or len(options) < 2:
                return False, "rejected", "MCQ must have at least 2 options."

            # Check for duplicate options (case-insensitive)
            cleaned_options = [opt.strip().lower() for opt in options if opt.strip()]
            if len(cleaned_options) != len(set(cleaned_options)):
                return False, "rejected", "MCQ options contain duplicate entries."

            # Check that correct answer is in options
            if ans.lower() not in cleaned_options:
                return False, "rejected", "Correct answer is missing from MCQ options."

            # Check distractors are not trivial or empty
            for opt in options:
                if len(opt.strip()) < 1:
                    return False, "rejected", "MCQ contains empty option."

        # 5. Review check (warning conditions)
        if overlap_score < 0.85 and not ans_in_source:
            return True, "needs_review", "Answer has partial evidence grounding."

        return True, "validated", "Passed all structural and evidence validations."

    def validate_batch(
        self,
        questions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate a batch of questions, deduplicate, and split into:
        - validated
        - needs_review
        - rejected
        """
        validated: List[Dict[str, Any]] = []
        needs_review: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        seen_questions = set()

        for q in questions:
            norm_q = " ".join(q.get("question", "").lower().split())

            # Deduplication
            if norm_q in seen_questions:
                rejected.append({
                    **q,
                    "validation_status": "rejected",
                    "validation_reason": "Duplicate question text."
                })
                continue
            seen_questions.add(norm_q)

            is_valid, status, reason = self.validate_question(q)
            q_copy = {
                **q,
                "validation_status": status,
                "validation_reason": reason,
                "is_validated": (status == "validated")
            }

            if status == "validated":
                validated.append(q_copy)
            elif status == "needs_review":
                needs_review.append(q_copy)
            else:
                rejected.append(q_copy)

        return {
            "validated": validated,
            "needs_review": needs_review,
            "rejected": rejected,
            "summary": {
                "total": len(questions),
                "validated_count": len(validated),
                "review_count": len(needs_review),
                "rejected_count": len(rejected)
            }
        }
