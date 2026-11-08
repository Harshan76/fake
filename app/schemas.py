from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime

class SourceItem(BaseModel):
    title: str
    publisher: str
    url: str
    description: str
    reliability: Optional[str] = "High"

class VerificationRequest(BaseModel):
    content: str = Field(..., min_length=5, max_length=50000, description="The news text or claims to verify")
    input_type: str = Field(default="text", description="Type of input: text, url, image, voice, document")
    original_url: Optional[str] = Field(default=None, description="Original source URL if provided")
    extracted_text: Optional[str] = Field(default=None, description="Text extracted from OCR/voice/document if applicable")

class UrlExtractionRequest(BaseModel):
    url: str = Field(..., description="Full URL of the news article to scrape and verify")

class UrlExtractionResponse(BaseModel):
    success: bool
    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publish_date: Optional[str] = None
    extracted_text: str
    word_count: int
    error: Optional[str] = None

class DocumentExtractionResponse(BaseModel):
    success: bool
    filename: str
    file_type: str
    extracted_text: str
    word_count: int
    page_count: Optional[int] = None
    error: Optional[str] = None

class VerificationResponse(BaseModel):
    id: str
    input_type: str
    submitted_content: str
    extracted_text: Optional[str] = None
    original_url: Optional[str] = None
    classification: str  # 'REAL', 'FAKE', 'UNABLE TO VERIFY'
    confidence_score: float
    ml_confidence: Optional[float] = None
    evidence_score: Optional[float] = None
    summary: Optional[str] = None
    explanation: str
    identified_claims: List[str] = []
    supporting_evidence: List[Dict[str, Any]] = []
    contradicting_evidence: List[Dict[str, Any]] = []
    sources: List[SourceItem] = []
    status: str
    timestamp: str

class PlatformStatsResponse(BaseModel):
    total_verifications: int
    real_count: int
    fake_count: int
    unverified_count: int
    average_confidence: float
    input_type_breakdown: Dict[str, int]
