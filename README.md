# TruthLens – AI-Powered Fake News Detection System

**TruthLens** is an enterprise-grade, multi-modal fake news detection and fact-verification platform built for citizens, journalists, and researchers. It combines a calibrated machine-learning stylometric classifier with real-time external evidence retrieval to evaluate claims with transparency and source attribution.

---

## 🎨 Visual Identity & Palette

TruthLens follows a warm, editorial visual identity inspired by classic investigative journalism:
- **Warm Cream/Beige**: `#F5F0E5`
- **Deep Navy/Black**: `#080F1B`
- **Golden Brown Accent**: `#B87927`
- **Card White**: `#FFFFFF`
- **Muted Slate Grey**: `#5C6470`

---

## 🚀 Key Features

### 1. Multi-Modal Verification Input System
- **📝 Text Input**: Type or paste complete news articles, viral social media posts, headlines, or atomic claims with live word count tracking and test sample presets.
- **🔗 URL Verification**: Paste any article hyperlink. TruthLens scrapes the headline, publisher, author, date, and body while gracefully handling paywalls and anti-scraping protections.
- **🖼️ Screenshot & Image OCR**: Drag-and-drop WhatsApp forwards, Instagram posts, or newspaper clippings. Client-side Tesseract OCR extracts text with an editable preview before submission.
- **🎙️ Voice Claim Studio**: Live Web Speech API integration with microphone permissions, real-time waveform animation, and editable transcript confirmation.
- **📄 Document Upload**: Native parsing of PDF, DOCX, and TXT files up to 15MB.

### 2. Forensic AI/ML & Evidence Pipeline
- **Hybrid Classifier**: TF-IDF word n-grams (1-2) + character n-grams (3-5) + custom stylometric features (sensationalism density, punctuation intensity, capitalization ratio, journalistic attribution).
- **Calibrated Classifier**: Logistic Regression with Sigmoid calibration, producing true probabilities $P(\text{Fake})$ and $P(\text{Real})$.
- **Claim Decomposition**: Identifies verifiable factual assertions, statistics, dates, and named entities.
- **Live Evidence Retrieval**: Queries Wikipedia Knowledge API, Google News RSS, and curated fact-checking registries without fabricating citations.
- **Three Deterministic Outcomes**: `REAL`, `FAKE`, or `UNABLE TO VERIFY`. Never forces binary verdicts when evidence is inconclusive.

### 3. "Explore More – Verify This News Yourself"
Every verification report provides clickable hyperlinks to primary documentation and independent fact-checkers, complete with publisher badges and relevance summaries.

### 4. Persistent SQLite Database & History Archive
- Backed by SQLAlchemy with an automated SQLite schema.
- Stores verification IDs, submitted text, input types, classifications, confidence metrics, claims, evidence, and timestamps.
- Interactive **History Archive** with real-time status filtering (REAL, FAKE, UNABLE TO VERIFY) and instant search.

---

## 🛠️ Project Structure

```
fake/
├── app/
│   ├── __init__.py
│   ├── config.py              # System configuration & limits
│   ├── database.py            # SQLite database engine & session
│   ├── models.py              # SQLAlchemy verification record model
│   ├── schemas.py             # Pydantic request/response models
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py        # Stylometric linguistic transformer
│   │   ├── trainer.py         # Calibrated NLP model trainer
│   │   ├── classifier.py      # Inference & probability engine
│   │   └── truthlens_model.joblib # Saved trained model
│   ├── services/
│   │   ├── __init__.py
│   │   ├── claim_extractor.py # Factual claim extractor
│   │   ├── evidence_search.py # Genuine source search & registry
│   │   ├── verifier.py        # Core triangulation pipeline
│   │   ├── document_parser.py # PDF/DOCX/TXT text extractor
│   │   └── url_scraper.py     # Resilient news article scraper
│   └── routers/
│       ├── __init__.py
│       ├── verify.py          # /api/verify, /api/extract-url, /api/extract-doc
│       ├── history.py         # /api/history, /api/history/{id}
│       └── stats.py           # /api/stats
├── static/
│   ├── css/
│   │   └── style.css          # Editorial cream/navy/gold stylesheet
│   ├── js/
│   │   ├── app.js             # Main SPA verification logic
│   │   ├── ocr.js             # Image drag-and-drop & OCR engine
│   │   └── voice.js           # Speech recognition & waveform visualizer
│   └── images/
│       └── logo.svg           # Magnifying glass vector brandmark
├── templates/
│   └── index.html             # Homepage, Hero, Workspace & Dashboard
├── truthlens.db               # Persistent SQLite database
├── test_truthlens.py          # Automated integration test suite
├── main.py                    # FastAPI server entry point
├── requirements.txt           # Python dependencies
└── run.bat                    # One-click Windows startup script
```

---

## 🏃 Quick Start

### 1. Launch the Server
Execute the automated batch launcher:
```cmd
run.bat
```
Or run directly with Python:
```powershell
python main.py
```

### 2. Access the Application
- **Web Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 3. Run Automated Tests
```powershell
python test_truthlens.py
```
