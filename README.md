# 🇬🇧 English ↔ Hindi Neural Translator 🇮🇳

Bidirectional Seq2Seq + Attention + Beam Search translation model with a React frontend and Flask backend.

## Project Structure

```
hindi-english-translator/
│
├── backend/
│   └── app.py                   # Flask API (port 5000)
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── components/
│       │   └── TranslatorApp.jsx
│       └── styles/
│           ├── index.css
│           └── TranslatorApp.module.css
│
├── models/
│   ├── model_config_v3.json     # Shared config
│   ├── eng_to_hin/
│   │   ├── att_encoder_inference.keras
│   │   ├── att_decoder_inference.keras
│   │   ├── eng_tokenizer.pkl
│   │   └── hin_tokenizer.pkl
│   └── hin_to_eng/
│       ├── rev_encoder_inference.keras
│       ├── rev_decoder_inference.keras
│       ├── rev_eng_tokenizer.pkl
│       └── rev_hin_tokenizer.pkl
│
├── data/
│   └── hindi_english_parallel.xls
│
├── notebooks/
│   └── model_v3_clean.ipynb
│
├── assets/
│   └── training_history.png
│
├── archive/
│   └── translator_model-*.zip
│
├── requirements.txt
└── README.md
```

## Quick Start

### 1 — Backend
```bash
pip install -r requirements.txt
python backend/app.py
# → http://localhost:5000
```

### 2 — Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## API

**POST** `/translate`
```json
{
  "text": "How are you?",
  "direction": "en_hi",      // "en_hi" | "hi_en"
  "mode": "beam",            // "baseline" | "attention" | "beam"
  "beam_k": 5
}
```

**GET** `/health` — check which models are loaded.

## Models

Trained in `notebooks/model_v3_clean.ipynb`.  
Run **Section 15** of the notebook to generate the HI→EN reverse models.
