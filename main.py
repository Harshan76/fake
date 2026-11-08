import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import APP_NAME, APP_DESCRIPTION, VERSION, STATIC_DIR, TEMPLATES_DIR
from app.database import init_db, SessionLocal
from app.models import VerificationRecord
from app.ml.classifier import get_model
from app.routers import verify, history, stats

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
            print("Seeded demo verification records into database.")
    except Exception as e:
        print(f"Seed records exception: {e}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    print("Initializing TruthLens database...")
    init_db()
    seed_initial_demo_records()
    print("Loading TruthLens ML engine...")
    get_model()
    print("TruthLens AI-Powered Verification Platform ready.")
    yield

# Initialize FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for maximum flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include API Routers
app.include_router(verify.router)
app.include_router(history.router)
app.include_router(stats.router)

@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the primary TruthLens frontend user interface."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>TruthLens is starting up. Please refresh shortly.</h1>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
