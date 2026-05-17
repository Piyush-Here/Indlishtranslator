# backend/app.py — Bidirectional EN ↔ HI Seq2Seq Translator API
#
# Folder structure expected:
#   models/
#     model_config_v3.json
#     eng_to_hin/
#       att_encoder_inference.keras
#       att_decoder_inference.keras
#       eng_tokenizer.pkl
#       hin_tokenizer.pkl
#     hin_to_eng/
#       rev_encoder_inference.keras
#       rev_decoder_inference.keras
#       rev_hin_tokenizer.pkl
#       rev_eng_tokenizer.pkl
#
# Run:  python backend/app.py   (from project root)
#  OR   python app.py           (from inside backend/)
# API:  POST http://localhost:5000/translate

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle, json, re, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# ── Resolve model root (works whether run from project root or backend/) ──────
_here = os.path.dirname(os.path.abspath(__file__))  # directory of this file

def model_path(*parts):
    """Return absolute path under models/, resolved relative to this file."""
    return os.path.join(_here, "..", "models", *parts)

# ── Load config ───────────────────────────────────────────────────────────────
_cfg_candidates = [
    model_path("model_config_v3.json"),
    model_path("model_config_v2.json"),
    model_path("model_config.json"),
]
cfg_file = next((f for f in _cfg_candidates if os.path.exists(f)), None)
if not cfg_file:
    raise FileNotFoundError(
        "No model_config*.json found inside models/. Run the notebook first."
    )

with open(cfg_file) as f:
    config = json.load(f)

# Support both flat (v1/v2) and nested (v3) config formats
if "EN_TO_HI" in config:
    cfg_en_hi = config["EN_TO_HI"]
    cfg_hi_en = config["HI_TO_EN"]
else:
    cfg_en_hi = config
    cfg_hi_en = None

MAX_ENG_ENC = cfg_en_hi["max_eng_seq_len"]
MAX_HIN_DEC = cfg_en_hi["max_hin_seq_len"]
LATENT_ENHI = cfg_en_hi["LATENT_DIM"]

if cfg_hi_en:
    MAX_HIN_ENC = cfg_hi_en["max_hin_seq_len"]
    MAX_ENG_DEC = cfg_hi_en["max_eng_seq_len"]
    LATENT_HIEN = cfg_hi_en["LATENT_DIM"]

# ── Load tokenizers ───────────────────────────────────────────────────────────
print("Loading tokenizers...")

def load_pkl(rel_path):
    with open(model_path(rel_path), "rb") as f:
        return pickle.load(f)

eng_tok = load_pkl(os.path.join("eng_to_hin", "eng_tokenizer.pkl"))
hin_tok = load_pkl(os.path.join("eng_to_hin", "hin_tokenizer.pkl"))
hin_i2w = {i: w for w, i in hin_tok.word_index.items()}

rev_hin_tok = rev_eng_tok = None
rev_eng_i2w = {}

_rev_hin_path = model_path("hin_to_eng", "rev_hin_tokenizer.pkl")
_rev_eng_path = model_path("hin_to_eng", "rev_eng_tokenizer.pkl")
if os.path.exists(_rev_hin_path) and os.path.exists(_rev_eng_path):
    rev_hin_tok = load_pkl(os.path.join("hin_to_eng", "rev_hin_tokenizer.pkl"))
    rev_eng_tok = load_pkl(os.path.join("hin_to_eng", "rev_eng_tokenizer.pkl"))
    rev_eng_i2w = {i: w for w, i in rev_eng_tok.word_index.items()}
    print("✅ Reverse tokenizers loaded")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models (may take 20–40 s)...")

att_enc = att_dec = None
rev_enc = rev_dec = None

try:
    att_enc = tf.keras.models.load_model(model_path("eng_to_hin", "att_encoder_inference.keras"), safe_mode=False)
    att_dec = tf.keras.models.load_model(model_path("eng_to_hin", "att_decoder_inference.keras"), safe_mode=False)
    print("✅ EN→HI attention models loaded")
except Exception as e:
    print(f"⚠️  EN→HI models not found: {e}")

try:
    rev_enc = tf.keras.models.load_model(model_path("hin_to_eng", "rev_encoder_inference.keras"), safe_mode=False)
    rev_dec = tf.keras.models.load_model(model_path("hin_to_eng", "rev_decoder_inference.keras"), safe_mode=False)
    print("✅ HI→EN models loaded")
except Exception as e:
    print(f"⚠️  HI→EN models not found: {e}")

print("✅ Ready!\n")

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_english(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_hindi(text):
    text = str(text).strip()
    text = re.sub(r"[^\u0900-\u097F0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# ── Greedy decode ─────────────────────────────────────────────────────────────
def greedy_decode(enc_model, dec_model, padded, start_idx, end_idx, i2w, max_out_len):
    enc_outs, h, c = enc_model.predict(padded, verbose=0)
    token = np.zeros((1, 1))
    token[0, 0] = start_idx
    words = []
    for _ in range(max_out_len):
        raw = dec_model.predict([token, enc_outs, h, c], verbose=0)
        probs, h, c = raw[0], raw[1], raw[2]
        predicted = int(np.argmax(probs[0, -1, :]))
        if predicted in (0, end_idx):
            break
        word = i2w.get(predicted, "")
        if word and word not in ("\t", "\n"):
            words.append(word)
        token[0, 0] = predicted
    return " ".join(words)

# ── Beam search ───────────────────────────────────────────────────────────────
def beam_search(enc_model, dec_model, padded, enc_seq_len,
                start_idx, end_idx, i2w, max_out_len, latent_dim, beam_width=5):
    enc_outs, h, c = enc_model.predict(padded, verbose=0)
    beams = [(0.0, [start_idx], h, c)]
    completed = []

    for _ in range(max_out_len):
        candidates = []
        for score, ids, bh, bc in beams:
            if ids[-1] == end_idx:
                completed.append((score, ids, bh, bc))
                continue
            token = np.zeros((1, 1))
            token[0, 0] = ids[-1]
            raw = dec_model.predict([token, enc_outs, bh, bc], verbose=0)
            probs, nh, nc = raw[0], raw[1], raw[2]
            top_k = np.argsort(probs[0, -1, :])[-beam_width:][::-1]
            for tok in top_k:
                lp = np.log(probs[0, -1, tok] + 1e-10)
                candidates.append((score - lp, ids + [int(tok)], nh, nc))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        beams = candidates[:beam_width]
        if all(b[1][-1] == end_idx for b in beams):
            completed.extend(beams)
            break

    best = min(completed or beams, key=lambda x: x[0])
    words = []
    for idx in best[1][1:]:
        if idx in (0, end_idx):
            break
        word = i2w.get(idx, "")
        if word and word not in ("\t", "\n"):
            words.append(word)
    return " ".join(words)

# ── /translate endpoint ───────────────────────────────────────────────────────
@app.route("/translate", methods=["POST"])
def translate():
    data      = request.get_json(force=True)
    text      = data.get("text", "").strip()
    direction = data.get("direction", "en_hi")   # 'en_hi' | 'hi_en'
    mode      = data.get("mode", "attention")     # 'baseline' | 'attention' | 'beam'
    beam_k    = int(data.get("beam_k", 5))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        if direction == "en_hi":
            if not att_enc:
                return jsonify({"error": "EN→HI model not loaded. Check your .keras files."}), 500

            cleaned = clean_english(text)
            seq     = eng_tok.texts_to_sequences([cleaned])
            padded  = pad_sequences(seq, maxlen=MAX_ENG_ENC, padding="post")
            start   = hin_tok.word_index.get("\t", 1)
            end     = hin_tok.word_index.get("\n", 2)

            if mode in ("baseline", "attention"):
                translation = greedy_decode(att_enc, att_dec, padded, start, end, hin_i2w, MAX_HIN_DEC)
            else:
                translation = beam_search(att_enc, att_dec, padded, MAX_ENG_ENC, start, end, hin_i2w, MAX_HIN_DEC, LATENT_ENHI, beam_k)

        elif direction == "hi_en":
            if not rev_enc:
                return jsonify({"error": "HI→EN model not loaded. Run Section 15 of the notebook first."}), 500

            cleaned = clean_hindi(text)
            seq     = rev_hin_tok.texts_to_sequences([cleaned])
            padded  = pad_sequences(seq, maxlen=MAX_HIN_ENC, padding="post")
            start   = rev_eng_tok.word_index.get("\t", 1)
            end     = rev_eng_tok.word_index.get("\n", 2)

            if mode in ("baseline", "attention"):
                translation = greedy_decode(rev_enc, rev_dec, padded, start, end, rev_eng_i2w, MAX_ENG_DEC)
            else:
                translation = beam_search(rev_enc, rev_dec, padded, MAX_HIN_ENC, start, end, rev_eng_i2w, MAX_ENG_DEC, LATENT_HIEN, beam_k)

        else:
            return jsonify({"error": f"Unknown direction: {direction}"}), 400

        return jsonify({"translation": translation, "direction": direction, "mode": mode})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "en_hi_loaded": att_enc is not None,
        "hi_en_loaded": rev_enc is not None,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
