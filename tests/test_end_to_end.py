"""
End-to-End integration tests for EduExtract AI.
Tests DocumentService, NotesService, QuizService, and ExportService.
"""
import pytest
import pymupdf
from services.document_service import DocumentService
from services.notes_service import NotesService
from services.quiz_service import QuizService
from services.export_service import ExportService

def create_rich_sample_pdf() -> bytes:
    doc = pymupdf.open()
    # Page 1
    p1 = doc.new_page()
    t1 = (
        "Cellular Biology Overview\n\n"
        "Photosynthesis is defined as the biological process by which green plants transform light energy into chemical energy. "
        "Chloroplasts are the cellular organelles that contain chlorophyll pigments. "
        "Dr. Melvin Calvin discovered the light-independent chemical cycles of carbon fixation in 1950."
    )
    p1.insert_textbox(pymupdf.Rect(50, 50, 550, 700), t1)

    # Page 2
    p2 = doc.new_page()
    t2 = (
        "Energy Production and Mitochondria\n\n"
        "Mitochondria is the organelle that produces adenosine triphosphate for cellular metabolism. "
        "Glucose and oxygen undergo cellular respiration to release metabolic water and carbon dioxide. "
        "The United Nations Environment Programme monitors atmospheric carbon dioxide emissions globally."
    )
    p2.insert_textbox(pymupdf.Rect(50, 50, 550, 700), t2)

    data = doc.tobytes()
    doc.close()
    return data

def test_full_document_to_quiz_pipeline():
    pdf_bytes = create_rich_sample_pdf()
    doc_service = DocumentService()

    # 1. Ingest document
    doc_result = doc_service.process_pdf(pdf_bytes, filename="Cell_Biology_Test.pdf")
    doc_id = doc_result["document_id"]
    assert doc_id is not None
    assert doc_result["page_count"] == 2
    assert len(doc_result["sentences"]) >= 4

    # 2. Generate Notes
    notes_service = NotesService()
    notes = notes_service.generate_notes(doc_result["sentences"], method="tfidf", max_sentences=3)
    assert notes["summary_sentence_count"] <= 3
    assert len(notes["summary_text"]) > 10

    comp = notes_service.compare_summaries(doc_result["sentences"], max_sentences=3)
    assert "tfidf" in comp and "textrank" in comp

    # 3. Generate and Validate Quiz
    quiz_service = QuizService()
    quiz = quiz_service.generate_and_validate_quiz(
        doc_id=doc_id,
        title="Cell Biology",
        sentences=doc_result["sentences"],
        keywords=doc_result["keywords"],
        entities=doc_result["entities"],
        mcq_count=3,
        fitb_count=3,
        short_count=2
    )
    assert quiz["quiz_id"] is not None
    questions = quiz["questions"]
    assert len(questions) > 0

    # 4. Interactive Quiz Scoring
    user_answers = {}
    for q in questions:
        user_answers[q["question_id"]] = q["correct_answer"]  # Simulate perfect attempt

    score_result = quiz_service.score_quiz(
        quiz_id=quiz["quiz_id"],
        doc_id=doc_id,
        questions=questions,
        user_answers=user_answers
    )
    assert score_result["score"] == len(questions)
    assert score_result["percentage"] == 100.0

    # 5. Export Services
    kw_list = doc_result["keywords"]["tfidf_keywords"]
    md_notes = ExportService.export_notes_markdown(doc_result["title"], notes, kw_list)
    assert "# Smart Study Notes" in md_notes

    study_pack = ExportService.export_study_pack_markdown(doc_result["title"], notes, kw_list, questions)
    assert "Complete Study Pack" in study_pack
    assert "Part 4: Answer Key" in study_pack

    json_qs = ExportService.export_questions_json(questions)
    assert len(json_qs) > 0

    csv_qs = ExportService.export_questions_csv(questions)
    assert "Question Type,Question" in csv_qs

    # 6. Database Verification
    retrieved_doc = doc_service.load_document(doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc["filename"] == "Cell_Biology_Test.pdf"

    history = quiz_service.get_history(doc_id)
    assert len(history) >= 1
    assert history[0]["percentage"] == 100.0
