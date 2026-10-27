"""
Unit tests for PDF extraction and metadata preservation.
"""
import pytest
import pymupdf
from nlp.pdf_extractor import PDFExtractor, PDFExtractionError
from utils.validation import ValidationError

def create_sample_pdf_bytes(page_texts: list[str]) -> bytes:
    """Helper to create an in-memory PDF with known text."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 700), text)
    doc_bytes = doc.tobytes()
    doc.close()
    return doc_bytes

def test_pdf_extraction_success():
    texts = [
        "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy.",
        "During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen."
    ]
    pdf_bytes = create_sample_pdf_bytes(texts)
    extractor = PDFExtractor(enable_ocr=False)
    result = extractor.extract(pdf_bytes, filename="biology_sample.pdf")

    assert result["filename"] == "biology_sample.pdf"
    assert result["page_count"] == 2
    assert len(result["pages"]) == 2
    assert result["pages"][0]["page_number"] == 1
    assert "Photosynthesis" in result["pages"][0]["original_text"]
    assert result["pages"][1]["page_number"] == 2
    assert "minerals" in result["pages"][1]["original_text"]

def test_empty_pdf_bytes():
    extractor = PDFExtractor()
    with pytest.raises(ValidationError):
        extractor.extract(b"")

def test_corrupt_pdf_bytes():
    extractor = PDFExtractor()
    with pytest.raises(PDFExtractionError):
        extractor.extract(b"Not a valid PDF header or stream")
