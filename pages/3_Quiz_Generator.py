"""
Quiz Generator Page for EduExtract AI.
Generates validated practice quizzes with interactive test-taking, scoring,
and grounded evidence review.
"""
import streamlit as st
from services.quiz_service import QuizService
from services.export_service import ExportService
from services.document_service import DocumentService

st.set_page_config(page_title="Quiz Generator — EduExtract AI", page_icon="🎯", layout="wide")

st.title("🎯 Adaptive Practice Quiz Generator")
st.markdown("Test your comprehension with questions automatically synthesized and validated from your document.")

quiz_service = QuizService()
doc_service = DocumentService()
current_doc = st.session_state.get("current_doc")

if not current_doc:
    stored = doc_service.list_all_documents()
    if stored:
        st.info("No document loaded in session. Please select one below:")
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
        st.warning("Please upload a PDF document first.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/1_Upload_Document.py")
        st.stop()

# Sidebar: Quiz Configuration
with st.sidebar:
    st.markdown(f"**Active Document:**\n`{current_doc['title']}`")
    st.markdown("---")
    st.header("Quiz Settings")

    mode = st.radio(
        "Generation Engine:",
        options=["Local NLP Rule-Based (Verified)", "Generative Model"],
        help="Local NLP uses entity recognition and definition patterns without LLM hallucination."
    )

    q_types = st.multiselect(
        "Question Types to Include:",
        options=["Multiple Choice (MCQ)", "Fill-in-the-Blank", "Short Answer"],
        default=["Multiple Choice (MCQ)", "Fill-in-the-Blank", "Short Answer"]
    )

    mcq_count = st.slider("MCQ Count", 0, 10, 4) if "Multiple Choice (MCQ)" in q_types else 0
    fitb_count = st.slider("Fill-in-the-Blank Count", 0, 10, 3) if "Fill-in-the-Blank" in q_types else 0
    short_count = st.slider("Short Answer Count", 0, 10, 2) if "Short Answer" in q_types else 0

    gen_btn = st.button("⚡ Generate New Quiz", type="primary", use_container_width=True)

# Generate Quiz
if gen_btn:
    with st.spinner("Synthesizing questions and validating against source evidence..."):
        quiz_data = quiz_service.generate_and_validate_quiz(
            doc_id=current_doc["document_id"],
            title=current_doc["title"],
            sentences=current_doc.get("sentences", []),
            keywords=current_doc.get("keywords", {}),
            entities=current_doc.get("entities", {}),
            mcq_count=mcq_count,
            fitb_count=fitb_count,
            short_count=short_count
        )
        st.session_state["active_quiz"] = quiz_data
        st.session_state["quiz_submitted"] = False
        st.session_state["quiz_score_data"] = None
        st.rerun()

active_quiz = st.session_state.get("active_quiz")

if not active_quiz or not active_quiz.get("questions"):
    st.info("👈 Configure quiz parameters on the left and click **'Generate New Quiz'** to begin.")
    st.stop()

questions = active_quiz["questions"]
val_summary = active_quiz.get("validation_results", {}).get("summary", {})

# Validation Badge Banner
st.markdown(f"""
<div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;">
    <strong>✅ Question Validation Status:</strong> {len(questions)} questions generated and verified grounded in source text.
    <span style="font-size: 0.85rem; color: #065F46; margin-left: 10px;">
        (Validated: {val_summary.get('validated_count', len(questions))} | Filtered Out: {val_summary.get('rejected_count', 0)})
    </span>
</div>
""", unsafe_allow_html=True)

# Interactive Quiz Form
if not st.session_state.get("quiz_submitted", False):
    st.subheader(f"📝 {active_quiz['title']} ({len(questions)} Questions)")

    with st.form("quiz_form"):
        user_responses = {}

        for idx, q in enumerate(questions, 1):
            qid = q["question_id"]
            q_type = q["question_type"]

            st.markdown(f"#### Question {idx}")
            st.markdown(f"**{q['question']}**")

            if q_type == "mcq" and q.get("options"):
                user_responses[qid] = st.radio(
                    f"Select your answer for Q{idx}:",
                    options=q["options"],
                    key=f"resp_{qid}",
                    label_visibility="collapsed"
                )
            elif q_type == "fill_in_the_blank":
                if q.get("options"):
                    user_responses[qid] = st.radio(
                        f"Choose the missing term for Q{idx}:",
                        options=q["options"],
                        key=f"resp_{qid}",
                        label_visibility="collapsed"
                    )
                else:
                    user_responses[qid] = st.text_input(
                        f"Your answer for Q{idx}:",
                        key=f"resp_{qid}",
                        placeholder="Type answer here..."
                    )
            else:
                user_responses[qid] = st.text_input(
                    f"Your answer for Q{idx}:",
                    key=f"resp_{qid}",
                    placeholder="Type concise answer here..."
                )

            st.caption(f"Source: Page {q.get('source_page', 1)} | Pattern: {q.get('generation_method')}")
            st.markdown("---")

        submit_btn = st.form_submit_button("Submit Quiz for Evaluation ➔", type="primary")

        if submit_btn:
            score_data = quiz_service.score_quiz(
                quiz_id=active_quiz["quiz_id"],
                doc_id=current_doc["document_id"],
                questions=questions,
                user_answers=user_responses
            )
            st.session_state["quiz_score_data"] = score_data
            st.session_state["quiz_submitted"] = True
            st.rerun()

else:
    # Quiz Results Screen
    score_data = st.session_state.get("quiz_score_data")
    if score_data:
        st.subheader("🎉 Quiz Results & Evidence Breakdown")

        pct = score_data["percentage"]
        score_color = "#059669" if pct >= 75 else "#D97706" if pct >= 50 else "#DC2626"

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Score", f"{score_data['score']} / {score_data['total_score']}")
        c2.metric("Percentage", f"{pct}%")
        c3.metric("Status", "Passed ✅" if pct >= 60 else "Needs Review ⚠️")

        st.markdown("---")
        st.markdown("### Detailed Question Review")

        for idx, res in enumerate(score_data["detailed_results"], 1):
            is_corr = res["is_correct"]
            box_color = "#ECFDF5" if is_corr else "#FEF2F2"
            border_color = "#10B981" if is_corr else "#EF4444"
            icon = "✅ Correct" if is_corr else "❌ Incorrect"

            st.markdown(f"""
            <div style="background-color: {box_color}; border-left: 5px solid {border_color}; padding: 14px 18px; margin-bottom: 12px; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>Question {idx}: {res['question']}</strong>
                    <span style="font-weight: 700; color: {border_color};">{icon}</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.95rem;">
                    <strong>Your Answer:</strong> <code>{res['user_answer'] or '[No Answer Provided]'}</code><br/>
                    <strong>Correct Answer:</strong> <code>{res['correct_answer']}</code>
                </div>
                <div style="margin-top: 6px; font-size: 0.85rem; color: #4B5563;">
                    📖 <em>{res['explanation']}</em> (Source Page {res['source_page']})
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("🔄 Retake Quiz", use_container_width=True):
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_score_data"] = None
                st.rerun()

        with action_col2:
            csv_results = ExportService.export_quiz_results_csv(score_data)
            st.download_button(
                "📥 Download Results (.csv)",
                data=csv_results,
                file_name=f"{current_doc['filename']}_quiz_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        with action_col3:
            json_questions = ExportService.export_questions_json(questions)
            st.download_button(
                "📥 Export Questions (.json)",
                data=json_questions,
                file_name=f"{current_doc['filename']}_questions.json",
                mime="application/json",
                use_container_width=True
            )
