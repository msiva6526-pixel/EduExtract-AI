"""
EduExtract AI - Main Application and Home Dashboard
"""
import streamlit as st
from services.document_service import DocumentService
from services.quiz_service import QuizService
from database.db import init_db

# Initialize application database
init_db()

st.set_page_config(
    page_title="EduExtract AI — Smart Notes & Quiz Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished educational interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease-in-out;
    }
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Services
doc_service = DocumentService()
quiz_service = QuizService()

# Sidebar Navigation & Quick Info
with st.sidebar:
    st.markdown("### 🎓 EduExtract AI")
    st.markdown("**PDF to Smart Notes & Quizzes**")
    st.markdown("---")
    st.info("💡 **Local NLP Mode Active**\n\nAll summaries, keywords, entities, and quizzes operate locally without requiring third-party API keys.")
    
    docs = doc_service.list_all_documents()
    st.metric("Total Documents Stored", len(docs))
    attempts = quiz_service.get_history()
    st.metric("Total Quiz Attempts", len(attempts))

# Main Header
st.markdown('<div class="main-header">EduExtract AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Educational Knowledge Extractor, Smart Notes Summarizer, and Adaptive Quiz Generator</div>', unsafe_allow_html=True)

# Quick Action Banner
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#1E40AF;">📄 Upload PDF</h4>
        <p style="font-size:0.85rem; color:#6B7280; margin-top:5px;">Extract structured pages, verify text readability, and catalog metadata.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Upload ➔", key="btn_upload"):
        st.switch_page("pages/1_Upload_Document.py")

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#059669;">📝 Smart Notes</h4>
        <p style="font-size:0.85rem; color:#6B7280; margin-top:5px;">Generate TF-IDF & TextRank extractive summaries and key concept lists.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Notes ➔", key="btn_notes"):
        st.switch_page("pages/2_Smart_Notes.py")

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#7C3AED;">🎯 Quiz Generator</h4>
        <p style="font-size:0.85rem; color:#6B7280; margin-top:5px;">Practice with validated MCQs, fill-in-the-blanks, and short answers.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Take Quiz ➔", key="btn_quiz"):
        st.switch_page("pages/3_Quiz_Generator.py")

with col4:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#D97706;">🔬 NLP Lab</h4>
        <p style="font-size:0.85rem; color:#6B7280; margin-top:5px;">Inspect tokenization, PageRank, TF-IDF weights, and neural decoding.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open NLP Lab ➔", key="btn_lab"):
        st.switch_page("pages/4_NLP_Lab.py")

st.markdown("---")

# Recent Documents and Performance Section
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📚 Recent Documents")
    if docs:
        for d in docs[:5]:
            with st.container():
                st.markdown(f"""
                <div style="border-left: 4px solid #3B82F6; padding: 8px 12px; margin-bottom: 8px; background: #F9FAFB; border-radius: 4px;">
                    <strong>{d['title'] or d['filename']}</strong><br/>
                    <span style="font-size:0.85rem; color:#6B7280;">
                        Pages: {d['page_count']} | Size: {round(d['file_size_bytes']/1024, 1)} KB | Uploaded: {d['created_at'][:19]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No documents uploaded yet. Upload your first PDF to generate smart notes and practice quizzes.")

with right_col:
    st.subheader("📊 Recent Quiz Performance")
    if attempts:
        for att in attempts[:5]:
            pct = att["percentage"]
            color = "#059669" if pct >= 75 else "#D97706" if pct >= 50 else "#DC2626"
            st.markdown(f"""
            <div style="border: 1px solid #E5E7EB; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{att.get('quiz_title', 'Quiz Attempt')}</strong><br/>
                    <small style="color: #6B7280;">{att.get('attempted_at', '')[:19]}</small>
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">
                    {pct}% ({att['score']}/{att['total_score']})
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No quiz attempts yet. Generate a quiz from an uploaded document to track your progress.")

st.markdown("---")

# Explanatory System Architecture Overview
st.subheader("⚙️ Real NLP Algorithms Under the Hood")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    **1. Linguistic & Statistical Preprocessing**
    - Abbreviation-guarded sentence segmentation
    - Morphological token lemmatization
    - spaCy Named Entity Recognition (NER)
    - Stopword filtering preserving evidence context
    """)
with c2:
    st.markdown("""
    **2. Explainable Extractive Summarization**
    - Normalized Term-Weight TF-IDF ranking
    - NetworkX Graph-based TextRank (PageRank)
    - Pairwise sentence cosine similarity graph
    - Dynamic redundancy suppression
    """)
with c3:
    st.markdown("""
    **3. Evidence-Grounded Question Synthesis**
    - Definition and concept pattern extraction
    - Category-matched distractor clustering
    - Multi-point evidence grounding validation
    - Neural decoding algorithms (Greedy, Beam, Top-K, Top-P)
    """)
