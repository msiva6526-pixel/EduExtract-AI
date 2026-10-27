"""
Configuration module for EduExtract AI.
Loads settings from environment variables with safe defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Database
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "eduextract.db")))

# PDF and extraction limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_PAGE_COUNT = int(os.getenv("MAX_PAGE_COUNT", "100"))
MAX_TEXT_LENGTH_CHARACTERS = 2_000_000

# OCR settings
ENABLE_OCR = os.getenv("ENABLE_OCR", "False").lower() in ("true", "1", "yes")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# Generative model settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local_fallback")  # 'local_fallback', 'huggingface', 'openai'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "sshleifer/distilbart-cnn-6-6")

# Summarization defaults
DEFAULT_MAX_SUMMARY_SENTENCES = 8
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Quiz defaults
DEFAULT_MCQ_COUNT = 5
DEFAULT_FITB_COUNT = 5
DEFAULT_SHORT_ANSWER_COUNT = 3
DEFAULT_DISTRACTOR_COUNT = 3
