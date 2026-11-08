import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, JSON
from app.database import Base

class VerificationRecord(Base):
    __tablename__ = "verifications"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    input_type = Column(String(50), nullable=False, index=True)  # 'text', 'url', 'image', 'voice', 'document'
    submitted_content = Column(Text, nullable=False)
    extracted_text = Column(Text, nullable=True)
    original_url = Column(String(1024), nullable=True)
    
    # Classification & Scoring
    classification = Column(String(50), nullable=False, index=True)  # 'REAL', 'FAKE', 'UNABLE TO VERIFY'
    confidence_score = Column(Float, nullable=False)  # 0.0 to 100.0
    ml_confidence = Column(Float, nullable=True)       # Model linguistic confidence
    evidence_score = Column(Float, nullable=True)      # External evidence credibility
    
    # Detailed Analysis
    summary = Column(Text, nullable=True)
    explanation = Column(Text, nullable=False)
    identified_claims = Column(JSON, default=list)     # List of strings or claim dicts
    supporting_evidence = Column(JSON, default=list)   # List of evidence dicts
    contradicting_evidence = Column(JSON, default=list) # List of contradicting evidence dicts
    sources = Column(JSON, default=list)               # List of {title, publisher, url, description}
    
    # Metadata & Status
    status = Column(String(50), default="COMPLETED")    # 'COMPLETED', 'FAILED', 'PENDING'
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "input_type": self.input_type,
            "submitted_content": self.submitted_content,
            "extracted_text": self.extracted_text,
            "original_url": self.original_url,
            "classification": self.classification,
            "confidence_score": round(self.confidence_score, 1),
            "ml_confidence": round(self.ml_confidence, 1) if self.ml_confidence is not None else None,
            "evidence_score": round(self.evidence_score, 1) if self.evidence_score is not None else None,
            "summary": self.summary,
            "explanation": self.explanation,
            "identified_claims": self.identified_claims or [],
            "supporting_evidence": self.supporting_evidence or [],
            "contradicting_evidence": self.contradicting_evidence or [],
            "sources": self.sources or [],
            "status": self.status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
