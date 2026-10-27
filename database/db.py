"""
Database connection and initialization module for EduExtract AI using SQLite.
"""
import sqlite3
import os
from pathlib import Path
from config import DATABASE_PATH
from utils.logging_config import get_logger

logger = get_logger("database")

def get_db_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database with Row factory enabled."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db() -> None:
    """Initialize database tables and indexes safely."""
    conn = get_db_connection()
    try:
        with conn:
            # 1. Documents table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                page_count INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ocr_used BOOLEAN DEFAULT 0,
                status TEXT DEFAULT 'processed',
                raw_text_length INTEGER DEFAULT 0
            );
            """)

            # 2. Document Pages table (preserves source page metadata)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS document_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text_content TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                UNIQUE(document_id, page_number)
            );
            """)

            # 3. Document Summaries table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                method TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                sentence_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
            """)

            # 4. Quizzes table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                total_questions INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
            """)

            # 5. Quiz Questions table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id TEXT PRIMARY KEY,
                quiz_id TEXT NOT NULL,
                question_type TEXT NOT NULL,
                question TEXT NOT NULL,
                options_json TEXT,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                source_page INTEGER,
                source_sentence TEXT,
                generation_method TEXT,
                is_validated BOOLEAN DEFAULT 1,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (id) ON DELETE CASCADE
            );
            """)

            # 6. Quiz Attempts table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                score INTEGER NOT NULL,
                total_score INTEGER NOT NULL,
                percentage REAL NOT NULL,
                answers_json TEXT NOT NULL,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (id) ON DELETE CASCADE
            );
            """)

            # Indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_pages ON document_pages(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quiz_doc ON quizzes(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_quiz ON quiz_attempts(quiz_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_doc ON quiz_attempts(document_id);")

        logger.info("Database initialized successfully at %s", DATABASE_PATH)
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    finally:
        conn.close()
