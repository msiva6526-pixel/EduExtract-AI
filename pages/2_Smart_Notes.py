"""
Smart Notes Page for EduExtract AI.
Provides extractive summarization via TF-IDF and TextRank,
displays keywords & entities, and offers study exports.
"""
import streamlit as st
from services.notes_service import NotesService
from services.export_service import ExportService
from services.document_service import DocumentService

st.set_page_config(page_title="Smart Notes — EduExtract AI", page_icon="📝", layout="wide")

st.title("📝 Smart Notes & Extractive Summarization")
st.markdown("Generate concise, evidence-based study notes using mathematically grounded NLP summarization algorithms.")

doc_service = DocumentService()
current_doc = st.session_state.get("current_doc")

# If no active document in session, let user select from database
if not current_doc:
    stored = doc_service.list_all_documents()
    if stored:
        st.info("No active document loaded in current session. Please select one below:")
        sel_id = st.selectbox(
            "Select document:",
            options=[d["id"] for d in stored],
            format_func=lambda x: next(d["title"] or d["filename"] for d in stored if d["id"] == x)
        )
        if st.button("Load Document"):
            current_doc = doc_service.load_document(sel_id)
            st.session_state["current_doc"] = current_doc
            st.rerun()
    else:
        st.warning("Please upload a PDF document first on the Upload page.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/1_Upload_Document.py")
        st.stop()

st.sidebar.markdown(f"**Active Document:**\n`{current_doc['title']}`")
st.sidebar.markdown(f"**Total Sentences:** {len(current_doc.get('sentences', []))}")
st.sidebar.markdown("---")

# Controls
st.sidebar.header("Summarization Settings")
summary_method = st.sidebar.radio(
    "Summarization Algorithm:",
    options=["TF-IDF Extractive", "TextRank (PageRank Graph)", "Side-by-Side Comparison"],
    help="TF-IDF ranks by term significance; TextRank ranks by graph centrality of sentence similarity."
)

max_sentences = st.sidebar.slider(
    "Summary Length (Sentences):",
    min_value=3,
    max_value=min(25, max(3, len(current_doc.get("sentences", [])))),
    value=min(7, max(3, len(current_doc.get("sentences", []))))
)

sentences = current_doc.get("sentences", [])
keywords_data = current_doc.get("keywords", {})
entities_data = current_doc.get("entities", {})

# Generate Summaries based on selection
if summary_method == "Side-by-Side Comparison":
    st.subheader("⚖️ Side-by-Side Summary Comparison")
    comparison = NotesService.compare_summaries(sentences, max_sentences=max_sentences)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 TF-IDF Summary")
        tfidf_res = comparison["tfidf"]
        st.caption(f"Compression: **{tfidf_res['compression_ratio_pct']}%** ({tfidf_res['summary_word_count']} words)")
        for idx, s in enumerate(tfidf_res["sentences"], 1):
            st.markdown(f"**{idx}.** {s['text']}  \n<small style='color:#6B7280;'>Page {s['page_number']} | Term Score: {s['score']}</small>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🕸️ TextRank Summary")
        tr_res = comparison["textrank"]
        st.caption(f"Compression: **{tr_res['compression_ratio_pct']}%** ({tr_res['summary_word_count']} words)")
        for idx, s in enumerate(tr_res["sentences"], 1):
            st.markdown(f"**{idx}.** {s['text']}  \n<small style='color:#6B7280;'>Page {s['page_number']} | Graph Score: {s['score']}</small>", unsafe_allow_html=True)

    st.info(f"**Algorithm Agreement:** Both algorithms selected **{comparison['sentence_overlap_count']}** identical sentences ({comparison['sentence_overlap_percentage']}% overlap).")

    notes_result = tfidf_res

else:
    method_key = "tfidf" if "TF-IDF" in summary_method else "textrank"
    notes_result = NotesService.generate_notes(sentences, method=method_key, max_sentences=max_sentences)

    st.subheader(f"📑 {notes_result['method_label']}")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original Words", notes_result["original_word_count"])
    m2.metric("Summary Words", notes_result["summary_word_count"])
    m3.metric("Information Compression", f"{notes_result['compression_ratio_pct']}%")
    m4.metric("Pages Cited", len(notes_result["pages_referenced"]))

    st.markdown("---")
    st.markdown("### 📌 Executive Key Takeaways")
    for idx, s in enumerate(notes_result["sentences"], 1):
        with st.container():
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 12px 16px; margin-bottom: 10px; border-radius: 4px;">
                <span style="font-weight: 600; color: #1E3A8A;">Point {idx} (Page {s['page_number']})</span><br/>
                {s['text']}
            </div>
            """, unsafe_allow_html=True)

# Tabs for Keywords, Entities, and Source Evidence
st.markdown("---")
tab_kw, tab_ent, tab_export = st.tabs(["🏷️ Key Concepts & Terms", "👥 Named Entities", "📥 Export Study Materials"])

with tab_kw:
    st.subheader("Key Concepts & Vocabulary")
    tfidf_kws = keywords_data.get("tfidf_keywords", [])
    ling_kws = keywords_data.get("linguistic_keyphrases", [])

    kw_col1, kw_col2 = st.columns(2)
    with kw_col1:
        st.markdown("**Top Statistical Keywords (TF-IDF):**")
        for k in tfidf_kws[:10]:
            pages = ", ".join(map(str, k.get("pages", [])))
            st.markdown(f"- **{k['keyword']}** `(Score: {k['score']})` — Pages: {pages}")

    with kw_col2:
        st.markdown("**Linguistic Keyphrases (Noun Phrases):**")
        for k in ling_kws[:10]:
            pages = ", ".join(map(str, k.get("pages", [])))
            st.markdown(f"- **{k['keyword']}** `(Freq: {int(k['score'])})` — Pages: {pages}")

with tab_ent:
    st.subheader("Extracted Named Entities")
    cats = entities_data.get("entities_by_category", {})
    if cats:
        for cat_name, ent_list in cats.items():
            if ent_list:
                with st.expander(f"{cat_name} ({len(ent_list)} found)", expanded=(cat_name in ("People", "Organizations", "Locations / Geopolitics"))):
                    for ent in ent_list[:8]:
                        pages = ", ".join(map(str, ent.get("pages", [])))
                        st.markdown(f"- **{ent['text']}** (Occurrences: {ent['count']}) — Page(s): {pages}")
    else:
        st.info("No named entities identified in document.")

with tab_export:
    st.subheader("Download Formatted Study Materials")
    all_kws = keywords_data.get("tfidf_keywords", [])
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        md_content = ExportService.export_notes_markdown(current_doc["title"], notes_result, all_kws)
        st.download_button(
            "📥 Download Notes (Markdown .md)",
            data=md_content,
            file_name=f"{current_doc['filename']}_notes.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_d2:
        txt_content = ExportService.export_notes_txt(current_doc["title"], notes_result)
        st.download_button(
            "📥 Download Notes (Plain Text .txt)",
            data=txt_content,
            file_name=f"{current_doc['filename']}_notes.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col_d3:
        # Create minimal study pack
        pack_content = ExportService.export_study_pack_markdown(
            doc_title=current_doc["title"],
            notes_data=notes_result,
            keywords=all_kws,
            questions=[]
        )
        st.download_button(
            "📚 Download Full Study Pack (.md)",
            data=pack_content,
            file_name=f"{current_doc['filename']}_studypack.md",
            mime="text/markdown",
            use_container_width=True
        )
