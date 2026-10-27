"""
Validation and security utilities.
Validates uploaded files, file sizes, page limits, and text content.
"""
import re
from pathlib import Path
from config import MAX_FILE_SIZE_MB, MAX_PAGE_COUNT, MAX_TEXT_LENGTH_CHARACTERS

class ValidationError(Exception):
    """Custom exception raised when input validation fails."""
    pass

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or unsafe characters."""
    base = Path(filename).name
    # Keep only alphanumeric, underscores, hyphens, and periods
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return sanitized or "document.pdf"

def validate_pdf_file_size(file_bytes: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> None:
    """Ensure PDF size does not exceed the allowed maximum."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb} MB."
        )

def validate_page_count(page_count: int, max_pages: int = MAX_PAGE_COUNT) -> None:
    """Ensure page count is within acceptable bounds."""
    if page_count <= 0:
        raise ValidationError("Document has no readable pages.")
    if page_count > max_pages:
        raise ValidationError(
            f"Document has {page_count} pages, which exceeds maximum limit of {max_pages} pages."
        )

def validate_extracted_text(text: str) -> None:
    """Ensure extracted text is nonempty and within reasonable length."""
    stripped = text.strip()
    if not stripped:
        raise ValidationError("Extracted text is empty. PDF may be scanned or corrupted.")
    if len(stripped) > MAX_TEXT_LENGTH_CHARACTERS:
        raise ValidationError(
            f"Extracted text exceeds safety limit of {MAX_TEXT_LENGTH_CHARACTERS} characters."
        )
