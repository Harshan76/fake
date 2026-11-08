import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_DIR = BASE_DIR / "app" / "ml"
MODEL_PATH = MODEL_DIR / "truthlens_model.joblib"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'truthlens.db'}")

# Application Settings
APP_NAME = "TruthLens"
APP_DESCRIPTION = "AI-Powered Fake News Detection & Multi-Modal Verification System"
VERSION = "2.0.0"

# Upload Limits
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Verification Credibility Thresholds
MIN_CLAIM_LENGTH = 10
MAX_TEXT_LENGTH = 20000

# Search & Verification Providers (Configurable via Environment Variables)
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID", "")
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Request User-Agent for ethical news scraping
USER_AGENT = "TruthLens-FactCheck-Bot/2.0 (+https://truthlens.ai/verification-bot; contact@truthlens.ai)"
