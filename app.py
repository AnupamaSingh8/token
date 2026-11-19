"""
Flask Web Application for LLM Pipeline Explorer
===============================================
Run with:
    python3 app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
from flask import Flask, jsonify, request, send_from_directory
from pipeline_explorer import LLMPipeline

app = Flask(__name__, static_folder=".")
pipeline = LLMPipeline("file.txt")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/vocab", methods=["GET"])
def get_vocab():
    return jsonify({
        "sentences": pipeline.sentences,
        "vocab_size": len(pipeline.vocab),
        "vocab": [
            {
                "id": idx,
                "word": w,
                "category": pipeline.semantic_categories.get(w, "Special / Out of Vocab"),
                "spelling_coords": pipeline.spelling_coords.get(w, [0.0, 0.0]),
                "semantic_coords": pipeline.semantic_coords.get(w, [0.0, 0.0])
            }
            for w, idx in sorted(pipeline.vocab.items(), key=lambda x: x[1])
        ]
    })

@app.route("/api/tokenize", methods=["GET", "POST"])
def tokenize():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        sentence = data.get("sentence", "")
        mode = data.get("mode", "meaning")
    else:
        sentence = request.args.get("sentence", "The cat sat on the mat.")
        mode = request.args.get("mode", "meaning")

    tokens = pipeline.tokenize(sentence)
    result_tokens = []
    for i, t in enumerate(tokens, 1):
        spelling_coords = pipeline.get_token_embedding(t["token"], mode="spelling")
        semantic_coords = pipeline.get_token_embedding(t["token"], mode="meaning")
        result_tokens.append({
            "step": i,
            "token": t["token"],
            "token_id": t["token_id"],
            "is_oov": t["is_oov"],
            "spelling_coords": spelling_coords,
            "semantic_coords": semantic_coords,
            "active_coords": spelling_coords if mode == "spelling" else semantic_coords
        })

    return jsonify({
        "sentence": sentence,
        "mode": mode,
        "token_count": len(result_tokens),
        "tokens": result_tokens
    })

if __name__ == "__main__":
    print("=" * 60)
    print("Starting LLM Pipeline Explorer Web App on http://127.0.0.1:5000")
    print("Open this URL in your web browser to explore!")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
