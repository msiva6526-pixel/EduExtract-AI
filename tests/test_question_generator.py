"""
Unit tests for rule-based question generation and distractors.
"""
from nlp.question_generator import QuestionGenerator
from nlp.distractor_generator import DistractorGenerator

def test_definition_question_generation():
    sentences = [
        {
            "sentence_id": 0,
            "text": "Mitochondria is the organelle that generates most of the chemical energy needed by cells.",
            "page_number": 2
        }
    ]
    distractor_gen = DistractorGenerator(
        keywords=[{"keyword": "Ribosome"}, {"keyword": "Nucleus"}, {"keyword": "Chloroplast"}]
    )
    gen = QuestionGenerator(distractor_gen=distractor_gen)
    questions = gen.generate_definition_questions(sentences)

    assert len(questions) >= 1
    q = questions[0]
    assert "Mitochondria" in q["correct_answer"] or "organelle" in q["question"]
    assert q["source_page"] == 2
    assert q["source_sentence"] == sentences[0]["text"]

def test_fill_in_the_blank():
    sentences = [
        {
            "sentence_id": 0,
            "text": "The ribosome synthesizes proteins using messenger RNA genetic instructions.",
            "page_number": 4
        }
    ]
    keywords = [{"keyword": "ribosome"}, {"keyword": "proteins"}]
    distractor_gen = DistractorGenerator(
        keywords=[{"keyword": "lipids"}, {"keyword": "enzymes"}, {"keyword": "carbohydrates"}]
    )
    gen = QuestionGenerator(distractor_gen=distractor_gen)
    fitb_qs = gen.generate_fill_in_the_blank(sentences, keywords)

    assert len(fitb_qs) >= 1
    fitb = fitb_qs[0]
    assert "_______" in fitb["question"]
    assert fitb["correct_answer"].lower() in ("ribosome", "proteins")
    assert fitb["source_page"] == 4

def test_distractor_generator_safety():
    distractor_gen = DistractorGenerator(
        entities_by_category={
            "People": [
                {"text": "Alan Turing"},
                {"text": "Ada Lovelace"},
                {"text": "John von Neumann"}
            ]
        }
    )
    distractors = distractor_gen.generate_distractors(
        correct_answer="Alan Turing",
        target_count=3,
        entity_type="People"
    )

    # Correct answer MUST NOT be in distractors
    assert "Alan Turing" not in distractors
    # Should include Ada Lovelace and John von Neumann
    assert "Ada Lovelace" in distractors
    assert "John von Neumann" in distractors
