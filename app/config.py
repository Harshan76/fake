import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_DIR = BASE_DIR / "app" / "ml"
MODEL_PATH = MODEL_DIR / "truthlens_model.joblib"

# Environment & Server Settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"
DEBUG = False if IS_PRODUCTION else (os.getenv("DEBUG", "false").lower() == "true")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# Application Metadata
APP_NAME = "TruthLens"
APP_DESCRIPTION = "AI-Powered Fake News Detection & Multi-Modal Verification System"
VERSION = "2.0.0"

# Production Database Configuration
raw_database_url = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'truthlens.db'}")
# Normalize PostgreSQL URL for SQLAlchemy 1.4+ / 2.0 (Render & cloud providers often provide postgres://)
if raw_database_url.startswith("postgres://"):
    raw_database_url = raw_database_url.replace("postgres://", "postgresql://", 1)
DATABASE_URL = raw_database_url

# CORS Configuration (Configurable for Vercel deployment)
# Example: CORS_ORIGINS="https://truthlens.vercel.app,http://localhost:3000,http://localhost:5173"
raw_cors = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [orig.strip() for orig in raw_cors.split(",") if orig.strip()]

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
