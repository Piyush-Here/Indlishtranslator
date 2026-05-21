# 🇬🇧 Indlish Translator — English ↔ Hindi Neural Translator 🇮🇳

A bidirectional English ↔ Hindi machine translation system built with a **Seq2Seq LSTM + Bahdanau Attention** architecture. Includes a Flask REST backend and a React + Vite frontend.

> **⚠️ Accuracy Note**
> This model was trained on Google Colab Free Tier with limited compute and a small dataset. Translation quality is functional but not production-grade — it works best on **short, simple sentences** (under 10 words). Think of it as a working proof-of-concept rather than a replacement for Google Translate. If you retrain on a larger dataset with a GPU (e.g. a T4 for several hours), quality improves significantly.

---

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Step 1 — Clone the Repository](#step-1--clone-the-repository)
- [Step 2 — Get the Trained Model Files](#step-2--get-the-trained-model-files)
- [Step 3 — Set Up the Backend](#step-3--set-up-the-backend)
- [Step 4 — Set Up the Frontend](#step-4--set-up-the-frontend)
- [Step 5 — Run the App](#step-5--run-the-app)
- [Optional — Retrain the Model](#optional--retrain-the-model)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Project Limitations](#project-limitations)

---

## Project Structure

```
Indlishtranslator/
│
├── backend/
│   └── app.py                        # Flask API server (runs on port 5000)
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js                # Includes /api proxy to avoid CORS issues
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── components/
│       │   └── TranslatorApp.jsx     # Main UI component
│       └── styles/
│           ├── index.css
│           └── TranslatorApp.module.css
│
├── models/                           # ← YOU NEED TO CREATE THIS (see Step 2)
│   ├── model_config_v3.json
│   ├── eng_to_hin/
│   │   ├── att_encoder_inference.keras
│   │   ├── att_decoder_inference.keras
│   │   ├── eng_tokenizer.pkl
│   │   └── hin_tokenizer.pkl
│   └── hin_to_eng/
│       ├── rev_encoder_inference.keras
│       ├── rev_decoder_inference.keras
│       ├── rev_eng_tokenizer.pkl
│       └── rev_eng_tokenizer.pkl
│
├── notebooks/
│   └── model_v3_clean.ipynb          # Full training notebook
│
├── assets/
│   └── training_history.png
│
├── requirements.txt
└── README.md
```

---

## How It Works

```
User types text
      │
      ▼
React Frontend (Vite, port 5173)
      │  POST /api/translate
      │  (proxied by Vite to port 5000)
      ▼
Flask Backend (app.py, port 5000)
      │
      ├─ Cleans & tokenizes input text
      ├─ Runs Encoder LSTM → gets context vectors
      ├─ Runs Decoder LSTM step-by-step (greedy or beam search)
      └─ Returns translated text as JSON
```

**Three translation modes:**
- **Baseline** — basic Seq2Seq with greedy decoding
- **Attention** — adds Bahdanau attention for better context handling
- **Beam Search** — explores multiple decode paths (configurable width k=3/5/8/10), best quality but slowest

---

## Prerequisites

Make sure the following are installed on your machine before starting.

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.9 – 3.11 | `python --version` |
| pip | any recent | `pip --version` |
| Node.js | 18 or above | `node --version` |
| npm | comes with Node | `npm --version` |
| Git | any | `git --version` |

> **Python 3.12+ may have issues with TensorFlow.** If you're on 3.12, consider using `pyenv` or `conda` to create a Python 3.10 environment.

---

## Step 1 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Piyush-Here/Indlishtranslator.git
cd Indlishtranslator
```

---

## Step 2 — Get the Trained Model Files

The `.keras` model files and `.pkl` tokenizer files are **not included in the repository** (they're too large for GitHub). You have two options:

### Option A — Download from Google Drive (Recommended)

If your team has uploaded the trained model files to Google Drive, download them and place them inside the project like this:

```
Indlishtranslator/
└── models/
    ├── model_config_v3.json
    ├── eng_to_hin/
    │   ├── att_encoder_inference.keras
    │   ├── att_decoder_inference.keras
    │   ├── eng_tokenizer.pkl
    │   └── hin_tokenizer.pkl
    └── hin_to_eng/
        ├── rev_encoder_inference.keras
        ├── rev_decoder_inference.keras
        ├── rev_hin_tokenizer.pkl
        └── rev_eng_tokenizer.pkl
```

Create the folders manually if they don't exist:

```bash
mkdir -p models/eng_to_hin
mkdir -p models/hin_to_eng
```

Then move your downloaded files into the correct subfolders.

### Option B — Train the Models Yourself (from scratch)

If you don't have the pre-trained files, you can run the notebook to generate them. See [Optional — Retrain the Model](#optional--retrain-the-model) below.

---

## Step 3 — Set Up the Backend

### 3.1 — Create a Python virtual environment (recommended)

A virtual environment keeps your project's packages isolated from the rest of your system.

```bash
# From the project root (Indlishtranslator/)

# Create the environment
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### 3.2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `flask` — the web server
- `flask-cors` — handles cross-origin requests
- `tensorflow` — runs the LSTM models
- `numpy` — numerical operations

> **Note:** TensorFlow is a large package (~500 MB). Installation may take a few minutes depending on your internet speed.

### 3.3 — Verify the models folder

Before starting the server, double-check that your `models/` folder looks correct:

```bash
# On macOS / Linux:
ls models/
ls models/eng_to_hin/
ls models/hin_to_eng/

# On Windows:
dir models\
dir models\eng_to_hin\
dir models\hin_to_eng\
```

You should see `.keras` and `.pkl` files in each subfolder. If not, revisit Step 2.

---

## Step 4 — Set Up the Frontend

Open a **new terminal window** (keep the backend terminal for later). Navigate to the frontend folder:

```bash
cd frontend
npm install
```

This reads `package.json` and downloads all React/Vite dependencies into a `node_modules/` folder. This takes a minute or two.

---

## Step 5 — Run the App

You need **two terminals running at the same time** — one for the backend, one for the frontend.

### Terminal 1 — Start the Backend

```bash
# From the project root (Indlishtranslator/)
# Make sure your venv is activated first!

python backend/app.py
```

Expected output:
```
Loading tokenizers...
✅ Reverse tokenizers loaded
Loading models (may take 20–40 s)...
✅ EN→HI attention models loaded
✅ HI→EN models loaded
✅ Ready!

 * Running on http://0.0.0.0:5000
```

> The first startup takes 20–40 seconds because TensorFlow loads the model weights into memory. This is normal.

### Terminal 2 — Start the Frontend

```bash
# From the frontend/ folder
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Open the App

Go to **http://localhost:5173** in your browser. You should see the translator UI with a **✅ Backend online** indicator in the header.

---

## Optional — Retrain the Model

If you want to retrain the model (to improve accuracy or use a different dataset):

### On Google Colab (recommended — free GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/model_v3_clean.ipynb`
3. In the top menu: **Runtime → Change runtime type → GPU (T4)**
4. Run all cells from top to bottom
5. The notebook saves model files to your Colab session storage
6. **Run Section 15** specifically to generate the HI→EN reverse model files
7. Download all `.keras` and `.pkl` files and place them in `models/` as described in Step 2

### To improve translation quality

The model's accuracy is limited by two things:

- **Dataset size** — we used a small parallel corpus. Using a larger dataset like [IIT Bombay English-Hindi Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/) (1.5M+ sentence pairs) significantly improves output
- **Training time** — the free Colab T4 gives ~12 hours/session. More training epochs = better results
- **Model size** — increasing `LATENT_DIM` (hidden units) in the notebook config also helps, at the cost of longer training

---

## API Reference

The backend exposes two endpoints:

### `POST /translate`

Translate a sentence.

**Request body (JSON):**
```json
{
  "text": "How are you?",
  "direction": "en_hi",
  "mode": "beam",
  "beam_k": 5
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `text` | any string | The sentence to translate |
| `direction` | `"en_hi"` or `"hi_en"` | Translation direction |
| `mode` | `"baseline"`, `"attention"`, `"beam"` | Decoding strategy |
| `beam_k` | `3`, `5`, `8`, `10` | Beam width (only used when `mode` is `"beam"`) |

**Response:**
```json
{
  "translation": "आप कैसे हैं?",
  "direction": "en_hi",
  "mode": "beam"
}
```

### `GET /health`

Check which models are loaded.

**Response:**
```json
{
  "status": "ok",
  "en_hi_loaded": true,
  "hi_en_loaded": true
}
```

---

## Troubleshooting

**UI is stuck on "Translating…" / ❌ Backend offline shown**
- Make sure `python backend/app.py` is running in a separate terminal
- Check that you activated your virtual environment before running it
- Wait 30–40 seconds after starting — TensorFlow model loading takes time
- Try visiting `http://localhost:5000/health` in your browser directly. If you get a JSON response, the backend is up

**`ModuleNotFoundError: No module named 'flask'` (or tensorflow, etc.)**
- You forgot to activate the virtual environment. Run `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows), then try again

**`FileNotFoundError: No model_config*.json found inside models/`**
- Your `models/` folder is missing or empty. Go back to Step 2 and make sure the model files are in place

**`npm: command not found`**
- Node.js is not installed. Download it from [nodejs.org](https://nodejs.org) (choose the LTS version)

**TensorFlow install fails on Python 3.12**
- Create a Python 3.10 environment: `conda create -n translator python=3.10` then `conda activate translator` and retry

**Translations look wrong / gibberish output**
- This is a known limitation of the model — see [Project Limitations](#project-limitations) below
- Try shorter, simpler sentences (5–8 words)
- Try switching from **Beam Search** to **Attention** mode for some inputs

---

## Project Limitations

We want to be upfront about what this project can and can't do.

- **Trained on limited compute** — Google Colab Free Tier with a T4 GPU, ~2–3 hours of training time
- **Small dataset** — the parallel corpus used is small compared to production translation systems
- **Best on short sentences** — the model handles 5–10 word sentences most reliably. Longer sentences tend to lose coherence
- **No subword tokenization** — we use word-level tokenization (Keras `Tokenizer`), so out-of-vocabulary words (rare words, names, slang) are dropped
- **No attention visualization** — the attention weights are computed but not yet displayed in the UI
- **This is a college project**, not a production system — the goal was to understand the Seq2Seq + Attention architecture end-to-end, not to beat Google Translate

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | TensorFlow / Keras — LSTM Seq2Seq + Bahdanau Attention |
| Backend | Python, Flask, Flask-CORS |
| Frontend | React 19, Vite, CSS Modules |
| Training | Google Colab (Free Tier, T4 GPU) |
| Dataset | Kaggle — Hindi-English Parallel Corpus |

---

---

*Built as a B.Tech project. Seq2Seq with Attention is now largely superseded by Transformer-based models (like the ones powering Google Translate), but building one from scratch is still one of the best ways to actually understand how neural machine translation works.*
