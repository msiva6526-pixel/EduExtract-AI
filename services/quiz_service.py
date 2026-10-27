"""
Quiz Service module for EduExtract AI.
Coordinates question generation, rigorous validation, interactive scoring, and progress tracking.
"""
import uuid
from typing import Dict, Any, List, Optional
from nlp.distractor_generator import DistractorGenerator
from nlp.question_generator import QuestionGenerator
from nlp.question_validator import QuestionValidator
from database.repository import QuizRepository
from utils.logging_config import get_logger

logger = get_logger("quiz_service")

class QuizService:
    """Manages quiz generation, validation, scoring, and history."""

    def __init__(self):
        self.validator = QuestionValidator()

    def generate_and_validate_quiz(
        self,
        doc_id: str,
        title: str,
        sentences: List[Dict[str, Any]],
        keywords: List[Dict[str, Any]],
        entities: Dict[str, Any],
        mcq_count: int = 5,
        fitb_count: int = 5,
        short_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate candidate questions, filter through validator, and save valid questions to database.
        """
        distractor_gen = DistractorGenerator(
            entities_by_category=entities.get("entities_by_category", {}),
            keywords=keywords.get("tfidf_keywords", [])
        )
        generator = QuestionGenerator(distractor_gen=distractor_gen)

        all_unique_entities = entities.get("unique_entities", [])
        kw_list = keywords.get("tfidf_keywords", []) + keywords.get("linguistic_keyphrases", [])

        # Generate candidates
        raw_questions = generator.generate_all(
            sentences=sentences,
            keywords=kw_list,
            entities=all_unique_entities,
            mcq_count=mcq_count,
            fitb_count=fitb_count,
            short_count=short_count
        )

        # Validate candidates
        validation_results = self.validator.validate_batch(raw_questions)
        valid_questions = validation_results["validated"] + validation_results["needs_review"]

        # Persist quiz if valid questions exist
        quiz_id = str(uuid.uuid4())
        if valid_questions:
            QuizRepository.save_quiz(
                quiz_id=quiz_id,
                doc_id=doc_id,
                title=f"Quiz on {title}",
                questions=valid_questions
            )

        return {
            "quiz_id": quiz_id,
            "document_id": doc_id,
            "title": f"Quiz on {title}",
            "questions": valid_questions,
            "validation_results": validation_results
        }

    def score_quiz(
        self,
        quiz_id: str,
        doc_id: str,
        questions: List[Dict[str, Any]],
        user_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate user answers, calculate score and percentage, and record attempt.
        """
        correct_count = 0
        total_questions = len(questions)
        detailed_results: List[Dict[str, Any]] = []

        for q in questions:
            qid = q["question_id"]
            correct_ans = q["correct_answer"].strip()
            user_ans = user_answers.get(qid, "").strip()

            is_correct = False
            # Check for exact or normalized match
            if q["question_type"] in ("mcq", "fill_in_the_blank"):
                is_correct = (user_ans.lower() == correct_ans.lower())
            else:
                # Short answer: flexible match (containment or token overlap)
                user_clean = user_ans.lower()
                corr_clean = correct_ans.lower()
                is_correct = (
                    user_clean == corr_clean or
                    corr_clean in user_clean or
                    (len(user_clean) > 3 and user_clean in corr_clean)
                )

            if is_correct:
                correct_count += 1

            detailed_results.append({
                "question_id": qid,
                "question": q["question"],
                "question_type": q["question_type"],
                "user_answer": user_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
                "source_page": q.get("source_page", 1),
                "source_sentence": q.get("source_sentence", "")
            })

        percentage = round((correct_count / max(1, total_questions)) * 100, 1)

        # Save attempt in database
        attempt_id = QuizRepository.save_quiz_attempt(
            quiz_id=quiz_id,
            doc_id=doc_id,
            score=correct_count,
            total_score=total_questions,
            answers=user_answers
        )

        return {
            "attempt_id": attempt_id,
            "quiz_id": quiz_id,
            "score": correct_count,
            "total_score": total_questions,
            "percentage": percentage,
            "detailed_results": detailed_results
        }

    def get_history(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve historical quiz attempts."""
        return QuizRepository.list_attempts(doc_id=doc_id)
