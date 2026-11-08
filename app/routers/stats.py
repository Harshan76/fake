from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models import VerificationRecord
from app.schemas import PlatformStatsResponse

router = APIRouter(prefix="/api/stats", tags=["Platform Statistics"])

@router.get("", response_model=PlatformStatsResponse)
def get_platform_statistics(db: Session = Depends(get_db)):
    """Computes real-time platform statistics from the persistent database."""
    total = db.query(VerificationRecord).count()
    real_count = db.query(VerificationRecord).filter(VerificationRecord.classification == "REAL").count()
    fake_count = db.query(VerificationRecord).filter(VerificationRecord.classification == "FAKE").count()
    unverified_count = db.query(VerificationRecord).filter(VerificationRecord.classification == "UNABLE TO VERIFY").count()

    avg_conf_row = db.query(func.avg(VerificationRecord.confidence_score)).scalar()
    avg_conf = float(avg_conf_row) if avg_conf_row is not None else 0.0

    # Group by input_type
    type_counts = db.query(
        VerificationRecord.input_type, func.count(VerificationRecord.id)
    ).group_by(VerificationRecord.input_type).all()

    breakdown = {row[0]: row[1] for row in type_counts}

    return PlatformStatsResponse(
        total_verifications=total,
        real_count=real_count,
        fake_count=fake_count,
        unverified_count=unverified_count,
        average_confidence=round(avg_conf, 1),
        input_type_breakdown=breakdown
    )
