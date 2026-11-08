import os
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.models import VerificationRecord
from app.schemas import (
    VerificationRequest, VerificationResponse,
    UrlExtractionRequest, UrlExtractionResponse,
    DocumentExtractionResponse
)
from app.services.verifier import verify_news_content
from app.services.url_scraper import scrape_article_from_url
from app.services.document_parser import parse_document
from app.config import MAX_UPLOAD_SIZE_BYTES, ALLOWED_DOCUMENT_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS

logger = logging.getLogger("truthlens.verify")
router = APIRouter(prefix="/api", tags=["Verification"])

@router.post("/verify", response_model=VerificationResponse)
def verify_endpoint(payload: VerificationRequest, db: Session = Depends(get_db)):
    """
    Main verification pipeline endpoint.
    Accepts news text from any input method, executes claim extraction,
    ML stylometric analysis, and evidence-based cross-referencing,
    then stores the full record in the database.
    """
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="News content cannot be empty.")

    # Execute full verification pipeline
    result = verify_news_content(
        text=content,
        input_type=payload.input_type,
        original_url=payload.original_url
    )

    rec_id = str(uuid.uuid4())
    rec_time = datetime.utcnow()

    # Persist record in database with rollback guard
    record = VerificationRecord(
        id=rec_id,
        input_type=payload.input_type,
        submitted_content=payload.content,
        extracted_text=payload.extracted_text,
        original_url=payload.original_url,
        classification=result["classification"],
        confidence_score=result["confidence_score"],
        ml_confidence=result.get("ml_confidence"),
        evidence_score=result.get("evidence_score"),
        summary=result.get("summary"),
        explanation=result["explanation"],
        identified_claims=result.get("identified_claims", []),
        supporting_evidence=result.get("supporting_evidence", []),
        contradicting_evidence=result.get("contradicting_evidence", []),
        sources=result.get("sources", []),
        status="COMPLETED",
        timestamp=rec_time
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.to_dict()
    except Exception as db_err:
        db.rollback()
        logger.error(f"Database error while saving verification record: {db_err}")
        return {
            "id": rec_id,
            "input_type": payload.input_type,
            "submitted_content": payload.content,
            "extracted_text": payload.extracted_text,
            "original_url": payload.original_url,
            "classification": result["classification"],
            "confidence_score": round(result["confidence_score"], 1),
            "ml_confidence": round(result.get("ml_confidence"), 1) if result.get("ml_confidence") is not None else None,
            "evidence_score": round(result.get("evidence_score"), 1) if result.get("evidence_score") is not None else None,
            "summary": result.get("summary"),
            "explanation": result["explanation"],
            "identified_claims": result.get("identified_claims", []),
            "supporting_evidence": result.get("supporting_evidence", []),
            "contradicting_evidence": result.get("contradicting_evidence", []),
            "sources": result.get("sources", []),
            "status": "COMPLETED",
            "timestamp": rec_time.isoformat()
        }

@router.post("/extract-url", response_model=UrlExtractionResponse)
def extract_url_endpoint(payload: UrlExtractionRequest):
    """
    Scrapes and extracts full text, headline, author, and date from a news link.
    Returns extracted text to frontend for user review before verification.
    """
    scrape_result = scrape_article_from_url(payload.url)
    if not scrape_result.get("success"):
        return UrlExtractionResponse(
            success=False,
            url=payload.url,
            extracted_text="",
            word_count=0,
            error=scrape_result.get("error", "Failed to extract content from URL.")
        )

    return UrlExtractionResponse(
        success=True,
        url=scrape_result["url"],
        title=scrape_result.get("title"),
        author=scrape_result.get("author"),
        publisher=scrape_result.get("publisher"),
        publish_date=scrape_result.get("publish_date"),
        extracted_text=scrape_result["extracted_text"],
        word_count=scrape_result["word_count"]
    )

@router.post("/extract-doc", response_model=DocumentExtractionResponse)
async def extract_document_endpoint(file: UploadFile = File(...)):
    """
    Extracts text from uploaded PDF, DOCX, or TXT documents.
    Allows users to preview and edit extracted content prior to verification.
    """
    filename = file.filename or "unknown.txt"
    _, ext = os.path.splitext(filename.lower())

    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        return DocumentExtractionResponse(
            success=False,
            filename=filename,
            file_type=ext.lstrip('.').upper(),
            extracted_text="",
            word_count=0,
            error=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_DOCUMENT_EXTENSIONS)}"
        )

    content_bytes = await file.read()
    if len(content_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return DocumentExtractionResponse(
            success=False,
            filename=filename,
            file_type=ext.lstrip('.').upper(),
            extracted_text="",
            word_count=0,
            error=f"File exceeds size limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    result = parse_document(content_bytes, filename)
    if not result.get("success"):
        return DocumentExtractionResponse(
            success=False,
            filename=filename,
            file_type=ext.lstrip('.').upper(),
            extracted_text="",
            word_count=0,
            error=result.get("error", "Document extraction failed.")
        )

    return DocumentExtractionResponse(
        success=True,
        filename=filename,
        file_type=result["file_type"],
        extracted_text=result["extracted_text"],
        word_count=result["word_count"],
        page_count=result.get("page_count", 1)
    )

@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Backend image upload fallback endpoint.
    Validates file extension and size.
    """
    filename = file.filename or "screenshot.png"
    _, ext = os.path.splitext(filename.lower())

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported image format '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        }

    content_bytes = await file.read()
    if len(content_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return {
            "success": False,
            "error": f"Image exceeds size limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        }

    return {
        "success": True,
        "filename": filename,
        "size_bytes": len(content_bytes),
        "message": "Image received successfully. Ready for client/worker OCR analysis."
    }
