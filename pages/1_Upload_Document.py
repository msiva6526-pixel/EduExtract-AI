"""
Upload Document Page for EduExtract AI.
Handles PDF uploading, page-by-page extraction, metadata inspection, and preview.
"""
import streamlit as st
from services.document_service import DocumentService
from utils.validation import ValidationError
from nlp.pdf_extractor import PDFExtractionError
from config import MAX_FILE_SIZE_MB, MAX_PAGE_COUNT

st.set_page_config(page_title="Upload Document — EduExtract AI", page_icon="📄", layout="wide")

st.title("📄 PDF Document Upload & Extraction")
st.markdown("Upload any academic paper, lecture slide deck, textbook chapter, or article in PDF format.")

doc_service = DocumentService()

with st.sidebar:
    st.markdown("### Document Constraints")
    st.markdown(f"- **Max File Size:** {MAX_FILE_SIZE_MB} MB")
    st.markdown(f"- **Max Page Count:** {MAX_PAGE_COUNT} pages")
    st.markdown("- **Supported Formats:** `.pdf`")
    st.markdown("---")
    st.markdown("### Stored Documents")
    stored_docs = doc_service.list_all_documents()
    if stored_docs:
        doc_options = {d["id"]: f"{d['title'] or d['filename']} ({d['page_count']} pages)" for d in stored_docs}
        selected_id = st.selectbox(
            "Load existing document:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x]
        )
        if st.button("Load Selected Document", type="secondary"):
            with st.spinner("Loading document from database..."):
                doc_obj = doc_service.load_document(selected_id)
                if doc_obj:
                    st.session_state["current_doc"] = doc_obj
                    st.success(f"Loaded '{doc_obj['title']}'!")
                    st.rerun()

uploaded_file = st.file_uploader(
    "Choose a PDF file to process",
    type=["pdf"],
    help=f"Upload a standard or academic PDF document (max {MAX_FILE_SIZE_MB} MB)"
)

if uploaded_file is not None:
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Document Information")
        st.write(f"**Filename:** `{uploaded_file.name}`")
        st.write(f"**Uploaded Size:** `{round(uploaded_file.size / (1024 * 1024), 2)} MB`")

    with col2:
        process_btn = st.button("🚀 Process & Extract Text", type="primary", use_container_width=True)

    if process_btn:
        progress_bar = st.progress(0, text="Reading PDF bytes...")
        try:
            file_bytes = uploaded_file.read()
            progress_bar.progress(30, text="Extracting page text and metadata...")

            processed_doc = doc_service.process_pdf(file_bytes=file_bytes, filename=uploaded_file.name)
            progress_bar.progress(70, text="Performing linguistic preprocessing and entity detection...")

            st.session_state["current_doc"] = processed_doc
            progress_bar.progress(100, text="Processing complete!")
            st.success(f"Successfully processed **{processed_doc['title']}** ({processed_doc['page_count']} pages)!")

        except ValidationError as ve:
            progress_bar.empty()
            st.error(f"Validation Error: {ve}")
        except PDFExtractionError as pe:
            progress_bar.empty()
            st.error(f"PDF Extraction Failed: {pe}")
        except Exception as e:
            progress_bar.empty()
            st.error(f"An unexpected error occurred during processing: {str(e)}")

# Display current document details if active
current_doc = st.session_state.get("current_doc")
if current_doc:
    st.markdown("---")
    st.subheader(f"📑 Active Document: {current_doc['title']}")

    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Total Pages", current_doc["page_count"])
    stat2.metric("Extracted Sentences", len(current_doc.get("sentences", [])))
    pre = current_doc.get("preprocessed", {})
    stat3.metric("Word Tokens", pre.get("token_count", 0))
    stat4.metric("Unique Lemmas", pre.get("unique_token_count", 0))

    tab_preview, tab_pages, tab_meta = st.tabs(["📄 Full Text Preview", "📖 Page-by-Page Inspection", "ℹ️ Metadata"])

    with tab_preview:
        full_text = "\n\n".join([f"--- Page {p['page_number']} ---\n{p['original_text']}" for p in current_doc["pages"]])
        st.text_area("Extracted Document Text", full_text, height=350, disabled=True)

    with tab_pages:
        page_num_sel = st.selectbox(
            "Select Page to Inspect:",
            options=[p["page_number"] for p in current_doc["pages"]]
        )
        selected_page = next((p for p in current_doc["pages"] if p["page_number"] == page_num_sel), None)
        if selected_page:
            st.markdown(f"**Page {selected_page['page_number']} Content ({len(selected_page['original_text'].split())} words):**")
            st.info(selected_page["original_text"] or "[Empty Page]")

    with tab_meta:
        st.json({
            "Document ID": current_doc["document_id"],
            "Filename": current_doc["filename"],
            "Page Count": current_doc["page_count"],
            "OCR Fallback Used": current_doc.get("ocr_used", False),
            "Metadata": current_doc.get("metadata", {})
        })

    st.markdown("---")
    st.markdown("### 🎯 What would you like to do next?")
    next_c1, next_c2, next_c3 = st.columns(3)
    if next_c1.button("📝 Generate Smart Notes ➔", use_container_width=True):
        st.switch_page("pages/2_Smart_Notes.py")
    if next_c2.button("🎯 Generate Practice Quiz ➔", use_container_width=True):
        st.switch_page("pages/3_Quiz_Generator.py")
    if next_c3.button("🔬 Inspect in NLP Lab ➔", use_container_width=True):
        st.switch_page("pages/4_NLP_Lab.py")
