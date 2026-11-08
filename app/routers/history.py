from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from typing import Optional, List
from app.database import get_db
from app.models import VerificationRecord
from app.schemas import VerificationResponse

router = APIRouter(prefix="/api/history", tags=["Verification History"])

@router.get("", response_model=List[VerificationResponse])
def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="filter"),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Retrieves previous verification analyses stored in SQLite database.
    Supports filtering by classification and search queries.
    """
    query = db.query(VerificationRecord)

    if isinstance(status_filter, str) and status_filter.strip() and status_filter.upper() != "ALL":
        query = query.filter(VerificationRecord.classification == status_filter.upper())

    if isinstance(search, str) and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                VerificationRecord.submitted_content.ilike(term),
                VerificationRecord.explanation.ilike(term),
                VerificationRecord.summary.ilike(term)
            )
        )

    # Coerce page and limit if passed as FastAPI Query defaults in unit tests
    p = page if isinstance(page, int) else 1
    lim = limit if isinstance(limit, int) else 20

    records = query.order_by(desc(VerificationRecord.timestamp))\
                   .offset((p - 1) * lim)\
                   .limit(lim)\
                   .all()

    return [r.to_dict() for r in records]

@router.get("/{record_id}", response_model=VerificationResponse)
def get_history_detail(record_id: str, db: Session = Depends(get_db)):
    """Fetches full verification report by unique record ID."""
    record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Verification report not found.")
    return record.to_dict()

@router.delete("/{record_id}")
def delete_history_record(record_id: str, db: Session = Depends(get_db)):
    """Deletes an individual verification record."""
    record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Verification report not found.")
    db.delete(record)
    db.commit()
    return {"success": True, "message": "Verification record removed successfully."}

@router.delete("")
def clear_all_history(db: Session = Depends(get_db)):
    """Clears all verification records."""
    count = db.query(VerificationRecord).delete()
    db.commit()
    return {"success": True, "deleted_count": count, "message": "History cleared successfully."}
