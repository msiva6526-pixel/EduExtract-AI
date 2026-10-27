"""
Settings & System Diagnostics Page for EduExtract AI.
Displays NLP dependencies, model availability, database status, and configuration limits.
"""
import streamlit as st
import os
from config import DATABASE_PATH, MAX_FILE_SIZE_MB, MAX_PAGE_COUNT, ENABLE_OCR
from services.document_service import DocumentService
from models.model_loader import ModelLoader

st.set_page_config(page_title="Settings — EduExtract AI", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings & NLP Diagnostics")
st.markdown("Inspect local NLP backend readiness, runtime hardware, database management, and processing limits.")

doc_service = DocumentService()

tab_diag, tab_limits, tab_db = st.tabs(["🩺 Diagnostics & Models", "📏 Safety Limits", "🗄️ Database Management"])

with tab_diag:
    st.subheader("NLP Library & Hardware Diagnostics")

    # Hardware detection
    hw_info = ModelLoader.get_hardware_info()
    col1, col2, col3 = st.columns(3)
    col1.metric("Compute Device", hw_info["device"].upper())
    col2.metric("PyTorch Available", "Yes ✅" if hw_info["torch_available"] else "No ❌")
    col3.metric("CUDA Acceleration", "Yes 🚀" if hw_info["cuda_available"] else "CPU Only")

    st.markdown("---")
    st.subheader("Installed NLP Packages")

    pkg_status = []
    # Check PyMuPDF
    try:
        import pymupdf
        pkg_status.append({"Module": "PyMuPDF (fitz)", "Status": "Installed ✅", "Version": pymupdf.__version__, "Role": "PDF Text Extraction"})
    except ImportError:
        pkg_status.append({"Module": "PyMuPDF", "Status": "Missing ❌", "Version": "N/A", "Role": "PDF Text Extraction"})

    # Check spaCy
    try:
        import spacy
        nlp_loaded = False
        try:
            spacy.load("en_core_web_sm")
            nlp_loaded = True
        except Exception:
            pass
        status_str = "Installed & Model Ready ✅" if nlp_loaded else "Installed (Blank model fallback) ⚠️"
        pkg_status.append({"Module": "spaCy (en_core_web_sm)", "Status": status_str, "Version": spacy.__version__, "Role": "NER & Lemmatization"})
    except ImportError:
        pkg_status.append({"Module": "spaCy", "Status": "Missing ❌", "Version": "N/A", "Role": "NER"})

    # Check scikit-learn
    try:
        import sklearn
        pkg_status.append({"Module": "scikit-learn", "Status": "Installed ✅", "Version": sklearn.__version__, "Role": "TF-IDF & Cosine Similarity"})
    except ImportError:
        pkg_status.append({"Module": "scikit-learn", "Status": "Missing ❌", "Version": "N/A", "Role": "TF-IDF"})

    # Check networkx
    try:
        import networkx
        pkg_status.append({"Module": "networkx", "Status": "Installed ✅", "Version": networkx.__version__, "Role": "TextRank Graph PageRank"})
    except ImportError:
        pkg_status.append({"Module": "networkx", "Status": "Missing ❌", "Version": "N/A", "Role": "TextRank"})

    # Check pytesseract
    try:
        import pytesseract
        pkg_status.append({"Module": "pytesseract", "Status": "Installed ✅", "Version": pytesseract.__version__, "Role": "OCR Scanned PDF Fallback"})
    except ImportError:
        pkg_status.append({"Module": "pytesseract (OCR)", "Status": "Optional / Not Installed ℹ️", "Version": "N/A", "Role": "OCR Fallback"})

    st.table(pkg_status)

with tab_limits:
    st.subheader("Configured Safety & Processing Limits")
    st.markdown("These limits are configured via `.env` or defaults to ensure robust, memory-safe execution:")
    
    st.write(f"- **Maximum File Size:** `{MAX_FILE_SIZE_MB} MB`")
    st.write(f"- **Maximum Page Count:** `{MAX_PAGE_COUNT} pages`")
    st.write(f"- **OCR Fallback Enabled:** `{'Enabled' if ENABLE_OCR else 'Disabled (Standard Text Extraction)'}`")
    st.write(f"- **Database Path:** `{DATABASE_PATH}`")

with tab_db:
    st.subheader("Stored Document Management")
    stored_docs = doc_service.list_all_documents()

    if stored_docs:
        st.write(f"Currently managing **{len(stored_docs)}** stored documents in SQLite.")
        del_target = st.selectbox(
            "Select Document to Remove:",
            options=[d["id"] for d in stored_docs],
            format_func=lambda x: next(f"{d['title'] or d['filename']} (ID: {d['id'][:8]}...)" for d in stored_docs if d["id"] == x)
        )

        if st.button("🗑️ Permanently Delete Document", type="primary"):
            success = doc_service.remove_document(del_target)
            if success:
                st.success("Document and associated quizzes/attempts deleted successfully.")
                if st.session_state.get("current_doc", {}).get("document_id") == del_target:
                    st.session_state["current_doc"] = None
                st.rerun()
            else:
                st.error("Failed to delete document.")
    else:
        st.info("Database is currently empty.")
