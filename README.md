# EduExtract AI — PDF to Smart Notes & Quiz Generator with NLP Lab

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20scikit--learn%20%7C%20NetworkX-green)
![Tests](https://img.shields.io/badge/tests-pytest%20passing-brightgreen)

**EduExtract AI** is an educational, production-ready Natural Language Processing (NLP) web application designed for students and educators. It transforms standard academic PDF documents into structured study notes, comparative extractive summaries, and validated interactive practice quizzes.

Additionally, EduExtract AI features a comprehensive **NLP Lab** where users can dissect every algorithm step by step—from tokenization and lemmatization to PageRank graph centrality and autoregressive neural text decoding.

---

## 🌟 Key Features

1. **Robust PDF Text & Metadata Ingestion**:
   - Page-by-page text extraction with PyMuPDF (`pymupdf`).
   - Exact page provenance preserved across all notes, questions, and evidence citations.
   - Protection against corrupted, encrypted, or oversized documents.
   - Optional OCR fallback for scanned materials.

2. **Core Linguistic Preprocessing**:
   - Whitespace and Unicode normalization preserving scientific symbols and abbreviations.
   - Abbreviation-guarded sentence segmentation (handles "e.g.", "i.e.", "Dr.", "vs.").
   - Word tokenization, stopword filtering, and morphological lemmatization.
   - Named Entity Recognition (NER) powered by spaCy's `en_core_web_sm` pipeline.

3. **Explainable Extractive Summarization**:
   - **TF-IDF Summarizer**: Scores sentences using length-normalized term importance.
   - **TextRank Summarizer**: Builds a graph of sentences connected by pairwise cosine similarity and computes Google PageRank via NetworkX.
   - **Side-by-Side Comparison**: Direct contrast of compression ratios, sentence overlap, and score distributions.

4. **Evidence-Grounded Question Synthesis**:
   - **Multiple Choice Questions (MCQs)**: Dynamic distractor generation matching semantic categories.
   - **Fill-in-the-Blank (Cloze)**: Important concepts masked out for recall practice.
   - **Short-Answer Questions**: Inverted definition and factual queries.
   - **Multi-Point Validation**: Every question is verified for non-emptiness, distractor uniqueness, and strict evidence grounding before display.

5. **Complete Autoregressive Neural Decoding Package**:
   - Clean, modular implementation of:
     - **Greedy Decoding**: Deterministic argmax next-token selection.
     - **Beam Search**: Hypothesis tracking with cumulative log probabilities and length penalty.
     - **Top-K Sampling**: Filters tail logits to top $K$ candidates with temperature scaling.
     - **Top-P (Nucleus) Sampling**: Dynamic cutoff selecting the smallest subset reaching cumulative probability $P$.
     - **Temperature Scaling**: Controls distribution sharpness without invalidating rank ordering.

6. **Interactive NLP Lab**:
   - Inspect tokens, lemmas, POS tags, and vocabulary statistics.
   - View TF-IDF sentence weights and TextRank graph convergence.
   - Experiment with next-token logit distributions and decoding hyperparameters in real time.

7. **Persistence & Comprehensive Exports**:
   - Embedded SQLite database storing documents, pages, quizzes, and score attempts.
   - Download notes as Markdown (`.md`) or Plain Text (`.txt`).
   - Export questions as JSON or CSV.
   - Download unified **Study Packs** combining notes, practice questions, and answer keys with page evidence.

---

## 🏗️ Architecture & Module Structure

```text
EduExtract-AI/
├── app.py                      # Main Streamlit entrypoint and Home Dashboard
├── config.py                   # Environment configuration & safety limits
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules
│
├── pages/                      # Multi-page Streamlit application
│   ├── 1_Upload_Document.py    # PDF upload, extraction, and text preview
│   ├── 2_Smart_Notes.py        # TF-IDF & TextRank summaries, entities, notes
│   ├── 3_Quiz_Generator.py     # Interactive quiz taking, validation & scoring
│   ├── 4_NLP_Lab.py            # Educational algorithm laboratory & decoder lab
│   ├── 5_Quiz_History.py       # Performance analytics and previous attempts
│   └── 6_Settings.py           # NLP dependencies, limits, and document deletion
│
├── nlp/                        # Core NLP algorithms
│   ├── pdf_extractor.py        # Page-by-page extraction via PyMuPDF
│   ├── sentence_splitter.py    # Abbreviation-guarded sentence segmentation
│   ├── preprocessing.py        # Tokenization, lemmatization, and stopwords
│   ├── keyword_extractor.py    # TF-IDF and linguistic noun-phrase extraction
│   ├── entity_extractor.py     # spaCy Named Entity Recognition (NER)
│   ├── tfidf_summarizer.py     # Term-weight extractive summarizer
│   ├── textrank_summarizer.py  # NetworkX PageRank graph summarizer
│   ├── text_similarity.py      # Cosine similarity and Jaccard token overlap
│   ├── distractor_generator.py # Category-matched distractor generator
│   ├── question_generator.py   # Pattern-based question generation
│   └── question_validator.py   # Evidence grounding & structural verification
│
├── decoding/                   # Neural autoregressive text decoding
│   ├── decoder.py              # BaseDecoder interface, stable softmax, config
│   ├── greedy.py               # Greedy search decoder
│   ├── beam_search.py          # Beam search decoder with log-probs
│   ├── top_k_sampling.py       # Top-K sampling with temperature
│   ├── top_p_sampling.py       # Nucleus (Top-P) sampling
│   └── temperature.py          # Logit temperature scaling
│
├── models/                     # Language model interfaces
│   ├── model_loader.py         # Hardware detection (CPU/CUDA) & model loading
│   └── generation_interface.py # Generation interface & educational simulation
│
├── services/                   # Business logic layer
│   ├── document_service.py     # PDF ingestion pipeline & caching
│   ├── notes_service.py        # Summarization and note compilation
│   ├── quiz_service.py         # Quiz generation, validation & evaluation
│   └── export_service.py       # Markdown, TXT, JSON, and CSV exporters
│
├── database/                   # SQLite persistence
│   ├── db.py                   # Schema initialization and connection factory
│   └── repository.py           # Parameterized SQL CRUD operations
│
├── utils/                      # Helper utilities
│   ├── text_utils.py           # Whitespace cleaning & readability metrics
│   ├── validation.py           # Safety limits & filename sanitization
│   └── logging_config.py       # Structured application logging
│
└── tests/                      # Automated pytest suite
    ├── test_pdf_extractor.py
    ├── test_preprocessing.py
    ├── test_keywords.py
    ├── test_tfidf_summarizer.py
    ├── test_textrank.py
    ├── test_question_generator.py
    ├── test_question_validator.py
    ├── test_decoding.py
    └── test_end_to_end.py
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/EduExtract-AI.git
cd EduExtract-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install spaCy Linguistic Model
EduExtract AI uses spaCy's English pipeline for Named Entity Recognition and lemmatization:
```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Default parameters in `.env` are pre-tuned for local operation.

---

## 💻 Running the Application

### 1. Launch the New Web Application (Custom Frontend)
```bash
python server.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

### 2. Alternative: Streamlit UI
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite covering all NLP modules, decoders, and services:
```bash
pytest -v
```

All 27 unit and integration tests run in seconds without requiring heavy neural model downloads.

---

## 📖 Educational NLP Guide: Core Concepts Explained

### 1. Extractive vs. Generative Summarization
- **Extractive Summarization (Implemented via TF-IDF and TextRank)**:
  Selects the most informative, central sentences verbatim from the source document.
  *Advantages:* Factually grounded, eliminates hallucination, retains source page citations.
- **Generative Summarization (Abstractive)**:
  Uses an encoder-decoder or causal language model to generate novel sentences paraphrasing the text.
  *Limitations:* Prone to factual hallucination without strict retrieval grounding.

### 2. TF-IDF vs. TextRank Extractive Methods
- **TF-IDF Summarizer**:
  Weights words by their frequency in a sentence divided by their frequency across the whole document. Sentences with the highest cumulative term importance score are extracted.
- **TextRank (Graph-based PageRank)**:
  Models sentences as nodes in a graph. An edge exists if two sentences share similar vocabulary (cosine similarity). PageRank computes eigenvector centrality—identifying sentences that are most connected to other central sentences.

### 3. Neural Token Decoding Strategies
- **Greedy Search**: Selects $\arg\max_w P(w \mid \text{context})$ at every step. Extremely fast and deterministic, but easily trapped in repetitive loops.
- **Beam Search**: Maintains the top $B$ candidate sequences scored by cumulative log-probabilities $\sum \log P(w_t)$. Effective for structured translation, but can yield generic responses.
- **Top-K Sampling**: Truncates the vocabulary distribution to the top $K$ most probable words, then samples probabilistically. Prevents nonsensical tail tokens.
- **Top-P (Nucleus) Sampling**: Dynamically expands or contracts the candidate pool to the smallest set of words whose cumulative probability $\ge P$. Adapts naturally to high-confidence and low-confidence prediction contexts.
- **Temperature ($T > 0$)**: Computes $z_i' = z_i / T$ before softmax. Lower temperature ($T < 1.0$) concentrates probability mass on top candidates; higher temperature ($T > 1.0$) flattens the distribution for increased diversity.

---

## 🔒 Limitations & Edge Cases

1. **Scanned / Image-Only PDFs**: Standard PDF extraction requires embedded digital text. If a PDF is a pure image scan, the application will provide a clear notice and suggest enabling OCR (requires system-level Tesseract binary).
2. **Grammatical Variations**: Rule-based question synthesis relies on explicit grammatical cues (such as definition copulas). Passages without clear definitional statements will produce fewer definition questions, falling back on entity and cloze questions.
3. **Hardware Acceleration**: Neural decoding operates on CPU or GPU (CUDA/MPS) seamlessly. When large local language models are loaded, GPU acceleration is recommended.

---

## 📄 License
This project is licensed under the MIT License.
