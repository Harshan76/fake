import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables if .env file exists
load_dotenv(BASE_DIR / ".env")

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
VERSION = "2.3.0"

def normalize_database_url(url: str) -> str:
    """
    Normalizes PostgreSQL database URLs to ensure the explicit 'postgresql+psycopg://' scheme.
    Compatible with psycopg version 3 (psycopg[binary]>=3.2,<4).
    Handles:
      - postgres://... -> postgresql+psycopg://...
      - postgresql+psycopg2://... -> postgresql+psycopg://...
      - postgresql://... -> postgresql+psycopg://...
      - postgresql+psycopg://... (preserved)
      - sqlite://... (preserved)
    """
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

# Production Database Configuration
raw_database_url = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'truthlens.db'}")
DATABASE_URL = normalize_database_url(raw_database_url)

if IS_PRODUCTION and DATABASE_URL.startswith("sqlite"):
    import logging
    logging.getLogger("truthlens.config").warning(
        "CRITICAL WARNING: Running in PRODUCTION mode with an ephemeral SQLite database! "
        "Set the DATABASE_URL environment variable to a persistent PostgreSQL instance (e.g. on Render/Supabase/Neon)."
    )

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

# Dynamic Verification Providers (Configurable via Environment Variables)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", os.getenv("FACT_CHECK_API_KEY", "")).strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# Legacy search keys (kept for backward compatibility)
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "").strip()
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID", "").strip()
FACT_CHECK_API_KEY = GOOGLE_FACT_CHECK_API_KEY

def get_provider_capabilities() -> dict:
    """Returns availability status of dynamic verification and AI reasoning providers."""
    return {
        "tavily_search": bool(TAVILY_API_KEY),
        "google_fact_check": bool(GOOGLE_FACT_CHECK_API_KEY),
        "groq_reasoning": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL if GROQ_API_KEY else None,
        "ml_classifier": MODEL_PATH.exists()
    }

# Request User-Agent for ethical news scraping
USER_AGENT = "TruthLens-FactCheck-Bot/2.0 (+https://truthlens.ai/verification-bot; contact@truthlens.ai)"
