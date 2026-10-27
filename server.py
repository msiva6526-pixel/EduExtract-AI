"""
EduExtract AI — Flask REST API Backend
Exposes all NLP and quiz capabilities for the standalone HTML/CSS/JS frontend.
"""
import json
import os
import sys
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from database.db import init_db
from services.document_service import DocumentService
from services.notes_service import NotesService
from services.quiz_service import QuizService
from services.export_service import ExportService
from models.generation_interface import GenerationInterface
from models.model_loader import ModelLoader
from decoding.decoder import DecoderConfig
from decoding.temperature import apply_temperature
from decoding.decoder import stable_softmax
import numpy as np

# Initialise DB
init_db()

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

doc_service  = DocumentService()
quiz_service = QuizService()

# ──────────────────────────────────────────────────────────
#  Static frontend serving
# ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ──────────────────────────────────────────────────────────
#  Documents
# ──────────────────────────────────────────────────────────
@app.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify(doc_service.list_all_documents())


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    try:
        doc = doc_service.process_pdf(f.read(), f.filename)
        # Return lightweight summary (omit full text blobs to keep payload small)
        return jsonify({
            "document_id":  doc["document_id"],
            "title":        doc["title"],
            "filename":     doc["filename"],
            "page_count":   doc["page_count"],
            "ocr_used":     doc["ocr_used"],
            "sentence_count": len(doc["sentences"]),
            "token_count":  doc["preprocessed"].get("token_count", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 422


@app.route("/api/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    doc = doc_service.load_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    # Send lightweight metadata + sentences (text only)
    return jsonify({
        "document_id":  doc["document_id"],
        "title":        doc["title"],
        "filename":     doc["filename"],
        "page_count":   doc["page_count"],
        "ocr_used":     doc["ocr_used"],
        "sentence_count": len(doc["sentences"]),
        "pages": [{"page_number": p["page_number"], "text": p["original_text"][:400]} for p in doc["pages"]],
        "preprocessed": {
            "token_count":           doc["preprocessed"].get("token_count", 0),
            "unique_token_count":    doc["preprocessed"].get("unique_token_count", 0),
            "filtered_token_count":  doc["preprocessed"].get("filtered_token_count", 0),
            "stopwords_identified_count": doc["preprocessed"].get("stopwords_identified_count", 0),
            "top_vocabulary": doc["preprocessed"].get("top_vocabulary", [])[:20],
        },
    })


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    ok = doc_service.remove_document(doc_id)
    return jsonify({"deleted": ok})


# ──────────────────────────────────────────────────────────
#  Notes / Summarization
# ──────────────────────────────────────────────────────────
@app.route("/api/documents/<doc_id>/notes", methods=["POST"])
def generate_notes(doc_id):
    doc = doc_service.load_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    body = request.get_json(silent=True) or {}
    method       = body.get("method", "tfidf")          # "tfidf" | "textrank" | "compare"
    max_sentences = int(body.get("max_sentences", 7))

    sentences = doc["sentences"]
    keywords  = doc["keywords"]
    entities  = doc["entities"]

    if method == "compare":
        result = NotesService.compare_summaries(sentences, max_sentences=max_sentences)
        return jsonify(result)
    else:
        result = NotesService.generate_notes(sentences, method=method, max_sentences=max_sentences)

    # Attach keywords and entities
    result["keywords"] = {
        "tfidf": keywords.get("tfidf_keywords", [])[:12],
        "linguistic": keywords.get("linguistic_keyphrases", [])[:12],
    }
    result["entities"] = entities.get("entities_by_category", {})
    return jsonify(result)


# ──────────────────────────────────────────────────────────
#  Quizzes
# ──────────────────────────────────────────────────────────
@app.route("/api/documents/<doc_id>/quiz", methods=["POST"])
def generate_quiz(doc_id):
    doc = doc_service.load_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    body = request.get_json(silent=True) or {}
    mcq_count   = int(body.get("mcq_count", 4))
    fitb_count  = int(body.get("fitb_count", 3))
    short_count = int(body.get("short_count", 2))

    quiz = quiz_service.generate_and_validate_quiz(
        doc_id=doc_id,
        title=doc["title"],
        sentences=doc["sentences"],
        keywords=doc["keywords"],
        entities=doc["entities"],
        mcq_count=mcq_count,
        fitb_count=fitb_count,
        short_count=short_count,
    )
    return jsonify(quiz)


@app.route("/api/quiz/<quiz_id>/submit", methods=["POST"])
def submit_quiz(quiz_id):
    body = request.get_json(silent=True) or {}
    doc_id   = body.get("doc_id")
    questions = body.get("questions", [])
    answers   = body.get("answers", {})

    if not doc_id or not questions:
        return jsonify({"error": "doc_id and questions are required"}), 400

    result = quiz_service.score_quiz(
        quiz_id=quiz_id,
        doc_id=doc_id,
        questions=questions,
        user_answers=answers,
    )
    return jsonify(result)


@app.route("/api/quiz/history", methods=["GET"])
def quiz_history():
    doc_id = request.args.get("doc_id")
    return jsonify(quiz_service.get_history(doc_id=doc_id))


# ──────────────────────────────────────────────────────────
#  NLP Lab endpoints
# ──────────────────────────────────────────────────────────
@app.route("/api/documents/<doc_id>/nlp", methods=["GET"])
def nlp_analysis(doc_id):
    doc = doc_service.load_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    pre  = doc["preprocessed"]
    kws  = doc["keywords"]
    ents = doc["entities"]

    from nlp.tfidf_summarizer  import summarize_tfidf
    from nlp.textrank_summarizer import summarize_textrank

    sents = doc["sentences"]
    tfidf_scored = summarize_tfidf(sents, max_sentences=len(sents))
    tr_scored    = summarize_textrank(sents, max_sentences=min(15, len(sents)))

    return jsonify({
        "preprocessing": {
            "token_count":          pre.get("token_count", 0),
            "unique_token_count":   pre.get("unique_token_count", 0),
            "filtered_token_count": pre.get("filtered_token_count", 0),
            "stopwords_removed":    pre.get("stopwords_identified_count", 0),
            "top_vocabulary":       pre.get("top_vocabulary", [])[:20],
            "pos_tags":             pre.get("pos_tags", [])[:15],
        },
        "tfidf_sentences": [
            {"text": s["text"][:120], "score": s["score"], "page": s["page_number"]}
            for s in sorted(tfidf_scored, key=lambda x: x["score"], reverse=True)[:8]
        ],
        "textrank_sentences": [
            {"text": s["text"][:120], "score": s["score"], "page": s["page_number"]}
            for s in sorted(tr_scored, key=lambda x: x["score"], reverse=True)[:8]
        ],
        "keywords": kws.get("tfidf_keywords", [])[:10],
        "entities": {cat: ents_list[:5] for cat, ents_list in ents.get("entities_by_category", {}).items()},
    })


@app.route("/api/nlp/decode", methods=["POST"])
def run_decoder():
    body = request.get_json(silent=True) or {}
    strategy   = body.get("strategy", "greedy")
    temperature = float(body.get("temperature", 1.0))
    top_k       = int(body.get("top_k", 5))
    top_p       = float(body.get("top_p", 0.9))
    beam_width  = int(body.get("beam_width", 3))
    seed        = int(body.get("seed", 42))
    max_tokens  = int(body.get("max_tokens", 10))

    try:
        cfg = DecoderConfig(
            strategy=strategy,
            max_new_tokens=max_tokens,
            temperature=max(0.1, temperature),
            top_k=max(1, top_k),
            top_p=min(1.0, max(0.05, top_p)),
            beam_width=max(1, beam_width),
            seed=seed,
        )

        toy = GenerationInterface.create_educational_vocab()
        cfg.eos_token_id = toy["eos_id"]

        gen = GenerationInterface(cfg)
        result = gen.generate_with_decoder(toy["model_fn"], [toy["bos_id"]], toy["tokenizer"])

        # Build logit table for first step
        raw_logits = toy["model_fn"]([toy["bos_id"]])
        scaled     = apply_temperature(raw_logits, temperature)
        probs      = stable_softmax(scaled)
        vocab      = toy["vocab"]
        top_indices = np.argsort(probs)[::-1][:10]
        logit_table = [
            {"token": vocab[i], "logit": round(float(raw_logits[i]), 3),
             "scaled_logit": round(float(scaled[i]), 3),
             "probability": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        return jsonify({
            "generated_text": result.generated_text,
            "token_ids": result.generated_token_ids,
            "tokens": [vocab[i] if i < len(vocab) else f"[{i}]" for i in result.generated_token_ids],
            "strategy": result.strategy,
            "settings": result.effective_settings,
            "notes": result.metadata.get("notes", ""),
            "logit_table": logit_table,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 422


# ──────────────────────────────────────────────────────────
#  Exports
# ──────────────────────────────────────────────────────────
@app.route("/api/documents/<doc_id>/export/notes", methods=["POST"])
def export_notes(doc_id):
    doc = doc_service.load_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    body = request.get_json(silent=True) or {}
    method = body.get("method", "tfidf")
    fmt    = body.get("format", "markdown")   # markdown | txt
    notes  = NotesService.generate_notes(doc["sentences"], method=method, max_sentences=8)
    kws    = doc["keywords"].get("tfidf_keywords", [])
    if fmt == "txt":
        content = ExportService.export_notes_txt(doc["title"], notes)
        mime, ext = "text/plain", "txt"
    else:
        content = ExportService.export_notes_markdown(doc["title"], notes, kws)
        mime, ext = "text/markdown", "md"
    return app.response_class(content, mimetype=mime,
                              headers={"Content-Disposition": f'attachment; filename="notes.{ext}"'})


@app.route("/api/documents/<doc_id>/export/studypack", methods=["POST"])
def export_studypack(doc_id):
    doc = doc_service.load_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    body      = request.get_json(silent=True) or {}
    questions = body.get("questions", [])
    notes     = NotesService.generate_notes(doc["sentences"], method="tfidf", max_sentences=8)
    kws       = doc["keywords"].get("tfidf_keywords", [])
    content   = ExportService.export_study_pack_markdown(doc["title"], notes, kws, questions)
    return app.response_class(content, mimetype="text/markdown",
                              headers={"Content-Disposition": 'attachment; filename="studypack.md"'})


# ──────────────────────────────────────────────────────────
#  System info
# ──────────────────────────────────────────────────────────
@app.route("/api/system", methods=["GET"])
def system_info():
    hw = ModelLoader.get_hardware_info()
    pkgs = {}
    for name, mod in [("pymupdf","pymupdf"),("spacy","spacy"),("sklearn","sklearn"),
                      ("networkx","networkx"),("nltk","nltk")]:
        try:
            m = __import__(mod)
            pkgs[name] = getattr(m, "__version__", "installed")
        except ImportError:
            pkgs[name] = None
    # spaCy model check
    spacy_model = False
    try:
        import spacy; spacy.load("en_core_web_sm"); spacy_model = True
    except Exception:
        pass
    return jsonify({**hw, "packages": pkgs, "spacy_model_loaded": spacy_model})


if __name__ == "__main__":
    print("\n[EduExtract AI] Flask API + Frontend Server")
    print("   Open  http://127.0.0.1:5000  in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
