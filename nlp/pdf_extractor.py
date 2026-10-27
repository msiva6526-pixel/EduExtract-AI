"""
PDF Extractor module for EduExtract AI.
Extracts text page-by-page using PyMuPDF with metadata preservation and OCR fallback.
"""
import io
import uuid
from typing import Dict, Any, List, Optional
import pymupdf  # PyMuPDF
from utils.validation import validate_pdf_file_size, validate_page_count, ValidationError
from utils.logging_config import get_logger
from config import ENABLE_OCR, TESSERACT_CMD

logger = get_logger("pdf_extractor")

class PDFExtractionError(Exception):
    """Exception raised when PDF extraction encounters an error."""
    pass

class PDFExtractor:
    """Extracts text and metadata page-by-page from PDF streams or bytes."""

    def __init__(self, enable_ocr: bool = ENABLE_OCR, tesseract_cmd: Optional[str] = TESSERACT_CMD):
        self.enable_ocr = enable_ocr
        self.tesseract_cmd = tesseract_cmd

    def extract(self, file_bytes: bytes, filename: str = "document.pdf", doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text, metadata, and per-page structures from a PDF.
        Preserves original text and page numbers.
        """
        if not file_bytes:
            raise ValidationError("Uploaded file is empty (0 bytes).")

        # Validate file size
        validate_pdf_file_size(file_bytes)

        if not doc_id:
            doc_id = str(uuid.uuid4())

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF document: {e}")

        try:
            if doc.is_encrypted:
                raise PDFExtractionError("PDF is encrypted/password protected and cannot be processed.")

            page_count = len(doc)
            validate_page_count(page_count)

            # Metadata extraction
            meta = doc.metadata or {}
            title = meta.get("title")
            if not title or not title.strip():
                title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()

            pages_data: List[Dict[str, Any]] = []
            ocr_used_total = False
            total_extracted_chars = 0

            for page_idx in range(page_count):
                page_num = page_idx + 1
                page = doc[page_idx]
                text = page.get_text("text") or ""
                page_ocr_used = False

                # Check if page is empty or likely scanned
                if len(text.strip()) < 20 and self.enable_ocr:
                    ocr_text = self._try_ocr_page(page)
                    if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        page_ocr_used = True
                        ocr_used_total = True

                clean_text = text.strip()
                total_extracted_chars += len(clean_text)

                pages_data.append({
                    "document_id": doc_id,
                    "page_number": page_num,
                    "original_text": clean_text
                })

            if total_extracted_chars == 0:
                warning_msg = "No readable text found in the PDF. The document may be scanned or image-only."
                if not self.enable_ocr:
                    warning_msg += " (OCR is currently disabled or unavailable)."
                raise PDFExtractionError(warning_msg)

            return {
                "document_id": doc_id,
                "filename": filename,
                "title": title,
                "page_count": page_count,
                "file_size_bytes": len(file_bytes),
                "pages": pages_data,
                "ocr_used": ocr_used_total,
                "metadata": {
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "creator": meta.get("creator", ""),
                    "producer": meta.get("producer", "")
                }
            }
        finally:
            doc.close()

    def _try_ocr_page(self, page: pymupdf.Page) -> Optional[str]:
        """Attempt OCR on a page using pytesseract if installed."""
        try:
            import pytesseract
            from PIL import Image
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Render page to pixmap
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            return ocr_text
        except ImportError:
            logger.warning("pytesseract or PIL is not installed; OCR fallback unavailable.")
            return None
        except Exception as e:
            logger.warning("OCR attempt failed for page %s: %s", page.number, e)
            return None
