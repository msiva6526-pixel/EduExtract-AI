"""
Document Service module for EduExtract AI.
Coordinates PDF extraction, sentence splitting, preprocessing, keywords, and database persistence.
"""
import uuid
from typing import Dict, Any, List, Optional
from nlp.pdf_extractor import PDFExtractor
from nlp.sentence_splitter import split_document_pages
from nlp.preprocessing import preprocess_text
from nlp.keyword_extractor import KeywordExtractor
from nlp.entity_extractor import EntityExtractor
from database.repository import DocumentRepository
from database.db import init_db
from utils.logging_config import get_logger

logger = get_logger("document_service")

class DocumentService:
    """Coordinates document ingestion, NLP analysis, and caching."""

    def __init__(self):
        init_db()
        self.extractor = PDFExtractor()
        self.keyword_extractor = KeywordExtractor()
        self.entity_extractor = EntityExtractor()

    def process_pdf(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Full document processing pipeline:
        1. Extract text and page structures from PDF
        2. Split text into sentences preserving page numbers
        3. Preprocess text (tokens, lemmas, statistics)
        4. Extract keywords (TF-IDF & linguistic)
        5. Extract named entities
        6. Persist metadata and pages to SQLite
        """
        doc_id = str(uuid.uuid4())

        # 1. Extraction
        doc_data = self.extractor.extract(file_bytes=file_bytes, filename=filename, doc_id=doc_id)
        pages = doc_data["pages"]

        # 2. Sentence Splitting with page tracking
        sentences = split_document_pages(pages)

        # 3. Preprocessing
        full_text = " ".join([p["original_text"] for p in pages])
        preprocessed = preprocess_text(full_text)

        # 4. Keyword Extraction
        keywords_data = self.keyword_extractor.extract_all(sentences)

        # 5. Entity Extraction
        entities_data = self.entity_extractor.extract_entities(sentences)

        # 6. Database persistence
        DocumentRepository.save_document(
            doc_id=doc_id,
            filename=filename,
            title=doc_data.get("title", filename),
            page_count=doc_data.get("page_count", len(pages)),
            file_size_bytes=doc_data.get("file_size_bytes", len(file_bytes)),
            pages=pages,
            ocr_used=doc_data.get("ocr_used", False)
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "title": doc_data.get("title", filename),
            "page_count": doc_data.get("page_count", len(pages)),
            "pages": pages,
            "sentences": sentences,
            "preprocessed": preprocessed,
            "keywords": keywords_data,
            "entities": entities_data,
            "ocr_used": doc_data.get("ocr_used", False),
            "metadata": doc_data.get("metadata", {})
        }

    def load_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load document and re-construct sentence structures from database."""
        doc = DocumentRepository.get_document(doc_id)
        if not doc:
            return None

        pages = DocumentRepository.get_document_pages(doc_id)
        sentences = split_document_pages(pages)
        full_text = " ".join([p["original_text"] for p in pages])
        preprocessed = preprocess_text(full_text)
        keywords_data = self.keyword_extractor.extract_all(sentences)
        entities_data = self.entity_extractor.extract_entities(sentences)

        return {
            "document_id": doc_id,
            "filename": doc["filename"],
            "title": doc.get("title", doc["filename"]),
            "page_count": doc["page_count"],
            "pages": pages,
            "sentences": sentences,
            "preprocessed": preprocessed,
            "keywords": keywords_data,
            "entities": entities_data,
            "ocr_used": bool(doc.get("ocr_used", False)),
            "created_at": doc.get("created_at")
        }

    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List metadata for all stored documents."""
        return DocumentRepository.list_documents()

    def remove_document(self, doc_id: str) -> bool:
        """Delete document from database."""
        return DocumentRepository.delete_document(doc_id)
