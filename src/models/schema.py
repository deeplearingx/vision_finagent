from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from .enums import ReportStatus, AuditAction, DegradeReason


class ReportMeta(BaseModel):
    report_id: str
    company_ticker: str
    fiscal_year: int
    form_type: str  # 10K / 10Q
    status: ReportStatus = ReportStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PageEntity(BaseModel):
    report_id: str
    page_num: int
    image_base64: str
    # colpali_embeddings written directly to Milvus; shape: (num_patches, 128)


class PageResult(BaseModel):
    report_id: str
    page_num: int
    image_base64: str
    maxsim_score: float
    page_text: str = ""


class AuditLog(BaseModel):
    request_id: str
    action: AuditAction
    report_id: Optional[str] = None
    status: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvidenceSummary(BaseModel):
    report_id: str
    page_num: int
    maxsim_score: float


class QueryResponse(BaseModel):
    session_id: str
    answer: Optional[str] = None
    degraded: bool = False
    degrade_reason: DegradeReason = DegradeReason.NONE
    evidence: list[EvidenceSummary] = []
    evidence_source: Optional[str] = None
    retrieved_pages: list[PageResult] = []


class SessionSummary(BaseModel):
    session_id: str
    updated_at: datetime
    last_question: str = ""
    turn_count: int = 0
    has_evidence: bool = False
