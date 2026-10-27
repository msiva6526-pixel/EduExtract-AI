"""
Repository module for database CRUD operations.
Ensures parameterized SQL queries and safe data access.
"""
import json
import sqlite3
from typing import List, Dict, Any, Optional
from database.db import get_db_connection
from utils.logging_config import get_logger

logger = get_logger("repository")

class DocumentRepository:
    """Manages persistence for documents, pages, and summaries."""

    @staticmethod
    def save_document(
        doc_id: str,
        filename: str,
        title: Optional[str],
        page_count: int,
        file_size_bytes: int,
        pages: List[Dict[str, Any]],
        ocr_used: bool = False
    ) -> str:
        conn = get_db_connection()
        try:
            with conn:
                raw_len = sum(len(p.get("original_text", "")) for p in pages)
                conn.execute(
                    """
                    INSERT INTO documents (id, filename, title, page_count, file_size_bytes, ocr_used, raw_text_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, filename, title or filename, page_count, file_size_bytes, 1 if ocr_used else 0, raw_len)
                )

                for p in pages:
                    conn.execute(
                        """
                        INSERT INTO document_pages (document_id, page_number, text_content)
                        VALUES (?, ?, ?)
                        """,
                        (doc_id, p["page_number"], p["original_text"])
                    )
            return doc_id
        except Exception as e:
            logger.error("Error saving document %s: %s", doc_id, e)
            raise
        finally:
            conn.close()

    @staticmethod
    def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @staticmethod
    def list_documents() -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def get_document_pages(doc_id: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT page_number, text_content FROM document_pages WHERE document_id = ? ORDER BY page_number ASC",
                (doc_id,)
            )
            rows = cursor.fetchall()
            return [{"page_number": r["page_number"], "original_text": r["text_content"]} for r in rows]
        finally:
            conn.close()

    @staticmethod
    def delete_document(doc_id: str) -> bool:
        """Safely delete document and cascade to pages, quizzes, and attempts."""
        conn = get_db_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                return cursor.rowcount > 0
        finally:
            conn.close()


class QuizRepository:
    """Manages persistence for quizzes, questions, and attempts."""

    @staticmethod
    def save_quiz(quiz_id: str, doc_id: str, title: str, questions: List[Dict[str, Any]]) -> str:
        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO quizzes (id, document_id, title, total_questions) VALUES (?, ?, ?, ?)",
                    (quiz_id, doc_id, title, len(questions))
                )
                for q in questions:
                    options_str = json.dumps(q.get("options", [])) if q.get("options") else None
                    conn.execute(
                        """
                        INSERT INTO quiz_questions 
                        (id, quiz_id, question_type, question, options_json, correct_answer, explanation, source_page, source_sentence, generation_method, is_validated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            q["question_id"],
                            quiz_id,
                            q["question_type"],
                            q["question"],
                            options_str,
                            q["correct_answer"],
                            q.get("explanation", ""),
                            q.get("source_page", 1),
                            q.get("source_sentence", ""),
                            q.get("generation_method", "rule_based"),
                            1 if q.get("is_validated", True) else 0
                        )
                    )
            return quiz_id
        finally:
            conn.close()

    @staticmethod
    def get_quiz(quiz_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,))
            quiz_row = cursor.fetchone()
            if not quiz_row:
                return None
            
            quiz = dict(quiz_row)
            cursor.execute("SELECT * FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
            q_rows = cursor.fetchall()
            questions = []
            for qr in q_rows:
                qd = dict(qr)
                if qd.get("options_json"):
                    qd["options"] = json.loads(qd["options_json"])
                else:
                    qd["options"] = []
                questions.append(qd)
            quiz["questions"] = questions
            return quiz
        finally:
            conn.close()

    @staticmethod
    def save_quiz_attempt(quiz_id: str, doc_id: str, score: int, total_score: int, answers: Dict[str, Any]) -> int:
        conn = get_db_connection()
        try:
            percentage = round((score / max(1, total_score)) * 100, 1)
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO quiz_attempts (quiz_id, document_id, score, total_score, percentage, answers_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (quiz_id, doc_id, score, total_score, percentage, json.dumps(answers))
                )
                return cursor.lastrowid
        finally:
            conn.close()

    @staticmethod
    def list_attempts(doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute(
                    """
                    SELECT a.*, q.title as quiz_title, d.filename as doc_filename
                    FROM quiz_attempts a
                    JOIN quizzes q ON a.quiz_id = q.id
                    JOIN documents d ON a.document_id = d.id
                    WHERE a.document_id = ?
                    ORDER BY a.attempted_at DESC
                    """,
                    (doc_id,)
                )
            else:
                cursor.execute(
                    """
                    SELECT a.*, q.title as quiz_title, d.filename as doc_filename
                    FROM quiz_attempts a
                    JOIN quizzes q ON a.quiz_id = q.id
                    JOIN documents d ON a.document_id = d.id
                    ORDER BY a.attempted_at DESC
                    """
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("answers_json"):
                    item["answers"] = json.loads(item["answers_json"])
                results.append(item)
            return results
        finally:
            conn.close()
