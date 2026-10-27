"""
Unit tests for question validation and evidence grounding.
"""
from nlp.question_validator import QuestionValidator

def test_valid_question():
    validator = QuestionValidator()
    q = {
        "question": "What is photosynthesis?",
        "correct_answer": "conversion of light energy into chemical energy",
        "source_sentence": "Photosynthesis involves the conversion of light energy into chemical energy in plants.",
        "source_page": 1,
        "question_type": "short_answer"
    }
    is_valid, status, reason = validator.validate_question(q)
    assert is_valid is True
    assert status == "validated"

def test_ungrounded_answer_rejection():
    validator = QuestionValidator()
    q = {
        "question": "Who invented calculus?",
        "correct_answer": "Isaac Newton",
        "source_sentence": "The cell membrane controls cellular osmotic pressure.",
        "source_page": 1,
        "question_type": "short_answer"
    }
    is_valid, status, reason = validator.validate_question(q)
    assert is_valid is False
    assert status == "rejected"
    assert "not grounded" in reason.lower()

def test_mcq_duplicate_options_rejection():
    validator = QuestionValidator()
    q = {
        "question": "Which organelle generates ATP?",
        "options": ["Mitochondria", "mitochondria", "Ribosome", "Nucleus"],
        "correct_answer": "Mitochondria",
        "source_sentence": "Mitochondria generates ATP cellular energy.",
        "source_page": 2,
        "question_type": "mcq"
    }
    is_valid, status, reason = validator.validate_question(q)
    assert is_valid is False
    assert status == "rejected"
    assert "duplicate" in reason.lower()

def test_batch_validation_and_deduplication():
    validator = QuestionValidator()
    questions = [
        {
            "question": "What is DNA?",
            "correct_answer": "genetic material",
            "source_sentence": "DNA is the genetic material of living organisms.",
            "source_page": 1,
            "question_type": "short_answer"
        },
        {
            # Duplicate question text
            "question": "What is DNA?",
            "correct_answer": "genetic material",
            "source_sentence": "DNA is the genetic material of living organisms.",
            "source_page": 1,
            "question_type": "short_answer"
        }
    ]
    batch = validator.validate_batch(questions)
    assert len(batch["validated"]) == 1
    assert len(batch["rejected"]) == 1
    assert "Duplicate" in batch["rejected"][0]["validation_reason"]
