"""
Quiz History Page for EduExtract AI.
Displays stored quiz attempts, score analytics, and attempt reviews.
"""
import streamlit as st
import pandas as pd
from services.quiz_service import QuizService
from services.document_service import DocumentService

st.set_page_config(page_title="Quiz History — EduExtract AI", page_icon="📊", layout="wide")

st.title("📊 Practice Quiz History & Learning Analytics")
st.markdown("Track your performance trends across past quizzes and review answers with source evidence citations.")

quiz_service = QuizService()
doc_service = DocumentService()

stored_docs = doc_service.list_all_documents()
doc_filter = None

with st.sidebar:
    st.header("Filter by Document")
    if stored_docs:
        options = ["All Documents"] + [d["id"] for d in stored_docs]
        name_map = {d["id"]: d["title"] or d["filename"] for d in stored_docs}
        name_map["All Documents"] = "All Documents"

        sel_doc = st.selectbox(
            "Select Document:",
            options=options,
            format_func=lambda x: name_map[x]
        )
        if sel_doc != "All Documents":
            doc_filter = sel_doc

attempts = quiz_service.get_history(doc_id=doc_filter)

if not attempts:
    st.info("No quiz attempts recorded yet. Head over to the Quiz Generator to test your knowledge!")
    if st.button("Go to Quiz Generator ➔"):
        st.switch_page("pages/3_Quiz_Generator.py")
    st.stop()

# Aggregate Metrics
total_attempts = len(attempts)
avg_score = round(sum(a["percentage"] for a in attempts) / max(1, total_attempts), 1)
highest_score = max(a["percentage"] for a in attempts)
total_questions_answered = sum(a["total_score"] for a in attempts)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Quizzes Taken", total_attempts)
m2.metric("Average Score", f"{avg_score}%")
m3.metric("Highest Score", f"{highest_score}%")
m4.metric("Questions Answered", total_questions_answered)

st.markdown("---")
st.subheader("📋 Completed Attempts")

# Table of attempts
table_data = [
    {
        "Attempt ID": a["id"],
        "Quiz Title": a.get("quiz_title", "Quiz"),
        "Document": a.get("doc_filename", "Document"),
        "Score": f"{a['score']} / {a['total_score']}",
        "Percentage": f"{a['percentage']}%",
        "Date & Time": a.get("attempted_at", "")[:19]
    }
    for a in attempts
]
df_attempts = pd.DataFrame(table_data)
st.dataframe(df_attempts, use_container_width=True)

# Detailed inspection of an attempt
st.markdown("---")
st.subheader("🔍 Review Past Attempt Answers")

attempt_ids = [a["id"] for a in attempts]
selected_attempt_id = st.selectbox(
    "Choose Attempt to Review:",
    options=attempt_ids,
    format_func=lambda x: f"Attempt #{x} — {next(a['quiz_title'] for a in attempts if a['id'] == x)} ({next(a['percentage'] for a in attempts if a['id'] == x)}%)"
)

selected_attempt = next((a for a in attempts if a["id"] == selected_attempt_id), None)
if selected_attempt and selected_attempt.get("answers"):
    st.markdown(f"**Submitted Answers for Attempt #{selected_attempt['id']}:**")
    for qid, ans in selected_attempt["answers"].items():
        st.markdown(f"- **Question ID `{qid}`:** `{ans}`")
