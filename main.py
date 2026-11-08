import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from app.config import (
    APP_NAME, APP_DESCRIPTION, VERSION, STATIC_DIR, TEMPLATES_DIR,
    HOST, PORT, CORS_ORIGINS, ENVIRONMENT, IS_PRODUCTION, DEBUG
)
from app.database import init_db, SessionLocal, check_db_connection
from app.models import VerificationRecord
from app.ml.classifier import get_model
from app.routers import verify, history, stats

# Configure structured application logging
logging.basicConfig(
    level=logging.INFO if IS_PRODUCTION else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("truthlens.app")

def seed_initial_demo_records():
    """Seeds a small set of verified records if database is empty for immediate demonstration."""
    db = SessionLocal()
    try:
        count = db.query(VerificationRecord).count()
        if count == 0:
            demo_items = [
                VerificationRecord(
                    input_type="text",
                    submitted_content="NASA's James Webb Space Telescope captured high-resolution infrared imagery of distant galaxies, revealing early cosmic structures formed 13 billion years ago.",
                    classification="REAL",
                    confidence_score=88.5,
                    ml_confidence=86.2,
                    evidence_score=92.0,
                    summary="NASA's James Webb Space Telescope observed galaxy structures dating back 13 billion years.",
                    explanation="This report was classified as REAL because the core factual claims correspond with established astrophysics records and official releases from NASA and international astronomical teams.",
                    identified_claims=[
                        "James Webb Space Telescope captured high-resolution infrared imagery",
                        "Observations reveal early cosmic structures formed 13 billion years ago"
                    ],
                    supporting_evidence=[
                        {"source": "NASA", "claim_addressed": "NASA's James Webb Space Telescope Discoveries", "detail": "Official scientific updates and peer-reviewed releases from NASA's astrophysics mission."}
                    ],
                    contradicting_evidence=[],
                    sources=[
                        {
                            "title": "NASA's James Webb Space Telescope Discoveries",
                            "publisher": "NASA",
                            "url": "https://www.nasa.gov/mission_pages/webb/main/index.html",
                            "description": "Official mission portal covering deep-field imagery and cosmic timeline findings."
                        }
                    ],
                    status="COMPLETED"
                ),
                VerificationRecord(
                    input_type="image",
                    submitted_content="URGENT ALERT: Boiled garlic water mixed with lemon 100% cures all viral respiratory infections overnight! Forward to all your family groups before doctors delete this!",
                    extracted_text="URGENT ALERT: Boiled garlic water mixed with lemon 100% cures all viral respiratory infections overnight! Forward to all your family groups before doctors delete this!",
                    classification="FAKE",
                    confidence_score=91.4,
                    ml_confidence=93.0,
                    evidence_score=90.0,
                    summary="Viral message claiming boiled garlic water cures respiratory viral infections overnight.",
                    explanation="Classified as FAKE based on documented medical investigations and official World Health Organization advisories refuting miraculous overnight cures. High sensationalism and chain-forward directives detected.",
                    identified_claims=[
                        "Boiled garlic water with lemon 100% cures viral infections overnight",
                        "Doctors are actively hiding this remedy from the public"
                    ],
                    supporting_evidence=[],
                    contradicting_evidence=[
                        {"source": "World Health Organization (WHO)", "claim_addressed": "COVID-19 Mythbusters: Garlic", "detail": "WHO confirms that garlic does not cure or protect against viral respiratory infections."}
                    ],
                    sources=[
                        {
                            "title": "Coronavirus disease (COVID-19) mythbusters: Garlic",
                            "publisher": "World Health Organization (WHO)",
                            "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
                            "description": "Official WHO mythbusters documentation evaluating viral misinformation claims."
                        }
                    ],
                    status="COMPLETED"
                )
            ]
            db.add_all(demo_items)
            db.commit()
            logger.info("Seeded initial benchmark verification records into database.")
    except Exception as e:
        logger.warning(f"Demo seeding skipped or non-fatal exception: {e}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info(f"Starting {APP_NAME} in '{ENVIRONMENT}' mode...")
    init_db()
    seed_initial_demo_records()
    logger.info("Pre-warming ML model into memory...")
    get_model()
    logger.info(f"{APP_NAME} AI-Powered Verification Platform ready on port {PORT}.")
    yield
    logger.info(f"Shutting down {APP_NAME} services cleanly.")

# Initialize FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=DEBUG
)

# Configure CORS Middleware (Supports Vercel production domains & localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for standalone/local hosting
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include API Routers
app.include_router(verify.router)
app.include_router(history.router)
app.include_router(stats.router)

# Production Health Check Endpoint (For Render, Kubernetes, or load balancers)
@app.get("/health", tags=["System"])
def health_check():
    """
    Production health-check endpoint.
    Verifies that the API service, database, and ML model are operating correctly.
    """
    db_ok = check_db_connection()
    model_ok = get_model() is not None
    from app.config import DATABASE_URL
    db_type = "postgresql" if DATABASE_URL.startswith("postgresql") else "sqlite"

    status_str = "healthy" if (db_ok and model_ok) else "degraded"
    status_code = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_str,
            "service": APP_NAME,
            "version": VERSION,
            "environment": ENVIRONMENT,
            "database": "connected" if db_ok else "disconnected",
            "database_type": db_type,
            "model_loaded": model_ok
        }
    )

# Production-safe Global Exception Handlers (Security: No stack traces leaked to clients)
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Request validation error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the verification request." if IS_PRODUCTION else str(exc),
            "status_code": 500
        }
    )

# Root route serves local UI when running in self-contained / developer mode
@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the primary TruthLens frontend interface for local or monolithic deployments."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>TruthLens Backend API is operational.</h1><p>Visit <a href='/docs'>/docs</a> for API specifications or <a href='/health'>/health</a> for system status.</p>")

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
