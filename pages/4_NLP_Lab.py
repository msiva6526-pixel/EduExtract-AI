"""
NLP Lab Page for EduExtract AI.
An interactive educational laboratory demonstrating text preprocessing, TF-IDF,
TextRank graph algorithms, question synthesis, and neural decoding step-by-step.
"""
import streamlit as st
import pandas as pd
import numpy as np
from services.document_service import DocumentService
from models.generation_interface import GenerationInterface
from decoding.decoder import DecoderConfig, stable_softmax
from decoding.temperature import apply_temperature
from nlp.text_similarity import calculate_cosine_similarity, calculate_jaccard_similarity

st.set_page_config(page_title="NLP Lab — EduExtract AI", page_icon="🔬", layout="wide")

st.title("🔬 Interactive NLP Laboratory")
st.markdown("Inspect and experiment with the core algorithms powering modern text processing, extractive summarization, and neural token decoding.")

doc_service = DocumentService()
current_doc = st.session_state.get("current_doc")

if not current_doc:
    stored = doc_service.list_all_documents()
    if stored:
        st.info("Select a document to inspect its computed NLP representations:")
        sel_id = st.selectbox(
            "Select document:",
            options=[d["id"] for d in stored],
            format_func=lambda x: next(d["title"] or d["filename"] for d in stored if d["id"] == x)
        )
        if st.button("Load into Lab"):
            current_doc = doc_service.load_document(sel_id)
            st.session_state["current_doc"] = current_doc
            st.rerun()
        st.stop()
    else:
        st.warning("Please upload a PDF document first.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/1_Upload_Document.py")
        st.stop()

# Lab Tabs
tab_pipe, tab_tfidf, tab_graph, tab_sim, tab_decoding = st.tabs([
    "1️⃣ Preprocessing & Tokens",
    "2️⃣ TF-IDF Scoring",
    "3️⃣ TextRank Graph (PageRank)",
    "4️⃣ Semantic Similarity",
    "5️⃣ Neural Decoding Simulator"
])

# ----------------- Tab 1: Preprocessing & Tokens -----------------
with tab_pipe:
    st.subheader("Text Normalization, Lemmatization & Stopword Filtering")
    st.markdown("""
    **Concept:** Preprocessing cleans raw document characters, separates punctuation, maps varied inflections
    to base lemmas (e.g., *cells* ➔ *cell*), and isolates non-informative stop words while preserving source evidence.
    """)

    pre = current_doc.get("preprocessed", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Tokens", pre.get("token_count", 0))
    col2.metric("Filtered Non-Stopwords", pre.get("filtered_token_count", 0))
    col3.metric("Stopwords Removed", pre.get("stopwords_identified_count", 0))

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**Top Lemma Vocabulary Frequency:**")
        vocab = pre.get("top_vocabulary", [])
        if vocab:
            df_vocab = pd.DataFrame(vocab[:15], columns=["Lemma", "Count"])
            st.dataframe(df_vocab, use_container_width=True)

    with c_right:
        st.markdown("**Part-of-Speech (POS) Sample:**")
        pos_list = pre.get("pos_tags", [])
        if pos_list:
            df_pos = pd.DataFrame(pos_list[:15])
            st.dataframe(df_pos, use_container_width=True)

# ----------------- Tab 2: TF-IDF Scoring -----------------
with tab_tfidf:
    st.subheader("Term Frequency - Inverse Document Frequency (TF-IDF)")
    st.markdown("""
    **Concept:** TF-IDF balances how frequently a term appears in a specific sentence against how common it is across the entire document.
    $$\\text{TF-IDF}(t, s, D) = \\text{TF}(t, s) \\times \\log\\left(\\frac{|D|}{1 + |\\{s' \\in D : t \\in s'\\}|}\\right)$$
    """)

    from nlp.tfidf_summarizer import summarize_tfidf
    sents = current_doc.get("sentences", [])
    if sents:
        scored_sents = summarize_tfidf(sents, max_sentences=len(sents))
        df_tfidf = pd.DataFrame([
            {
                "Order": s.get("original_order", idx),
                "Page": s.get("page_number", 1),
                "TF-IDF Score": s.get("score", 0),
                "Sentence": s.get("text", "")
            }
            for idx, s in enumerate(scored_sents)
        ])
        st.dataframe(df_tfidf.sort_values(by="TF-IDF Score", ascending=False), use_container_width=True)

# ----------------- Tab 3: TextRank Graph -----------------
with tab_graph:
    st.subheader("TextRank Graph Centrality (Google PageRank on Sentences)")
    st.markdown("""
    **Concept:** TextRank constructs a complete or thresholded graph where each node is a sentence, and edge weights represent
    pairwise cosine similarity. Sentences that share vocabulary with many other central sentences receive higher PageRank scores.
    """)

    from nlp.textrank_summarizer import summarize_textrank
    if sents:
        tr_sents = summarize_textrank(sents, max_sentences=min(10, len(sents)))
        df_tr = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Page": s.get("page_number", 1),
                "PageRank Score": s.get("score", 0),
                "Sentence": s.get("text", "")
            }
            for idx, s in enumerate(tr_sents)
        ])
        st.dataframe(df_tr, use_container_width=True)

# ----------------- Tab 4: Semantic Similarity -----------------
with tab_sim:
    st.subheader("Pairwise Sentence Similarity Calculator")
    st.markdown("Test cosine similarity and Jaccard token overlap between any two extracted sentences:")

    if len(sents) >= 2:
        sent_options = [f"[{s['page_number']}] {s['text'][:90]}..." for s in sents]
        s1_idx = st.selectbox("Sentence A:", range(len(sent_options)), format_func=lambda x: sent_options[x], key="sim_a")
        s2_idx = st.selectbox("Sentence B:", range(len(sent_options)), index=min(1, len(sent_options)-1), format_func=lambda x: sent_options[x], key="sim_b")

        text_a = sents[s1_idx]["text"]
        text_b = sents[s2_idx]["text"]

        cos_sim = calculate_cosine_similarity(text_a, text_b)
        jac_sim = calculate_jaccard_similarity(text_a, text_b)

        sc1, sc2 = st.columns(2)
        sc1.metric("TF-IDF Cosine Similarity", f"{cos_sim:.4f}")
        sc2.metric("Jaccard Token Overlap", f"{jac_sim:.4f}")

# ----------------- Tab 5: Neural Decoding Simulator -----------------
with tab_decoding:
    st.subheader("Neural Autoregressive Text Decoding Simulator")
    st.markdown("""
    **Concept:** Autoregressive language models predict next-token probability distributions.
    This interactive simulator demonstrates **Greedy**, **Beam Search**, **Top-K**, **Top-P (Nucleus)**,
    and **Temperature scaling** operating directly on next-token logit vectors.
    """)

    dec_col1, dec_col2 = st.columns([1, 2])

    with dec_col1:
        st.markdown("#### Decoder Configuration")
        strategy = st.selectbox("Strategy:", ["greedy", "beam_search", "top_k", "top_p"])
        temp = st.slider("Temperature (T > 0):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        k_val = st.slider("Top-K:", min_value=1, max_value=15, value=3) if strategy == "top_k" else 3
        p_val = st.slider("Top-P Threshold:", min_value=0.1, max_value=1.0, value=0.85, step=0.05) if strategy == "top_p" else 0.9
        b_width = st.slider("Beam Width:", min_value=1, max_value=6, value=3) if strategy == "beam_search" else 3
        seed_val = st.number_input("Random Seed:", min_value=0, max_value=9999, value=42)

        run_dec_btn = st.button("🚀 Run Decoding Step", type="primary", use_container_width=True)

    with dec_col2:
        st.markdown("#### Interactive Logit Distribution & Generation")

        toy_env = GenerationInterface.create_educational_vocab()
        vocab = toy_env["vocab"]
        model_fn = toy_env["model_fn"]
        tokenizer = toy_env["tokenizer"]

        # Sample initial context
        initial_tokens = [toy_env["token2id"]["Photosynthesis"]]

        # Display current raw vs temperature scaled logits
        raw_logits = model_fn(initial_tokens)
        scaled_logits = apply_temperature(raw_logits, temperature=temp)
        probs = stable_softmax(scaled_logits)

        # Plot top token probabilities
        top_indices = np.argsort(probs)[::-1][:10]
        df_probs = pd.DataFrame({
            "Token": [vocab[i] for i in top_indices],
            "Probability": [probs[i] for i in top_indices],
            "Scaled Logit": [scaled_logits[i] for i in top_indices]
        })
        st.dataframe(df_probs, use_container_width=True)

        if run_dec_btn:
            cfg = DecoderConfig(
                strategy=strategy,
                max_new_tokens=10,
                temperature=temp,
                top_k=k_val,
                top_p=p_val,
                beam_width=b_width,
                seed=seed_val,
                eos_token_id=toy_env["eos_id"]
            )
            gen_interface = GenerationInterface(cfg)
            result = gen_interface.generate_with_decoder(model_fn, initial_tokens, tokenizer)

            st.success(f"**Generated Text:** `{result.generated_text}`")
            st.json({
                "Strategy": result.strategy,
                "Token IDs": result.generated_token_ids,
                "Effective Settings": result.effective_settings,
                "Metadata": result.metadata
            })
