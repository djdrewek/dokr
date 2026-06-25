import json
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class PipelineEventOut(BaseModel):
    state: str
    agent: str | None
    detail: str | None
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Document responses ────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    """Full document record — returned by POST /documents/submit and GET /documents/{id}."""

    id: str = Field(example="doc_01J8K3PXMQR4T7N")
    status: str = Field(example="CLASSIFYING")
    document_class: str | None = Field(
        default=None,
        description="Document Class ID assigned by the Classification Agent. Null until classification completes.",
        example="dc_006",
    )
    document_class_name: str | None = Field(
        default=None,
        example="Supplier Invoice",
    )
    variant: str | None = Field(
        default=None,
        description="Document Variant ID. Null until a Variant Key is resolved.",
        example=None,
    )
    variant_key: str | None = Field(
        default=None,
        description="Derived sender identifier used to key this document to a Variant.",
        example=None,
    )
    file_name: str = Field(example="continental_invoice_1250015149.pdf")
    file_size_bytes: int = Field(example=84312)
    priority: str = Field(example="standard")
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str | None = None
    shipment_id: str | None = Field(default=None, description="ShipmentRecord ID this document belongs to.")
    skip_stages: list[str] | None = Field(default=None, description="Pipeline stages skipped for this document.")

    # ── Governance fields ─────────────────────────────────────────────────────
    classification_confidence: Optional[float] = Field(
        default=None,
        description=(
            "Normalised classification confidence (0.0–1.0). "
            "Values below 0.35 trigger GovernanceAgent review after extraction."
        ),
        example=0.78,
    )
    suggested_class_name: Optional[str] = Field(
        default=None,
        description=(
            "Populated when status=CANDIDATE_NEW_CLASS. "
            "GovernanceAgent's suggested name for the probable new document type."
        ),
    )
    candidate_reason: Optional[str] = Field(
        default=None,
        description="GovernanceAgent reasoning for flagging this as a new document type.",
    )
    ai_governance_result: Optional[dict] = Field(
        default=None,
        description=(
            "Full GovernanceAgent result JSON: verdict, reasoning, confidence, "
            "suggested_class, suggested_class_name, suggested_class_description, suggested_keywords."
        ),
    )

    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _parse_governance_json(cls, data: Any) -> Any:
        """
        ai_governance_result is stored as a JSON string in SQLite.
        Parse it to a dict before Pydantic validates the model.
        Works for both ORM objects and plain dicts.
        """
        # Handle ORM objects (SQLAlchemy model instances)
        if hasattr(data, '__dict__') and not isinstance(data, dict):
            raw = getattr(data, 'ai_governance_result', None)
            if isinstance(raw, str):
                try:
                    data.ai_governance_result = json.loads(raw)
                except Exception:
                    data.ai_governance_result = None
        # Handle plain dicts
        elif isinstance(data, dict):
            raw = data.get('ai_governance_result')
            if isinstance(raw, str):
                try:
                    data['ai_governance_result'] = json.loads(raw)
                except Exception:
                    data['ai_governance_result'] = None
        return data

    class Config:
        from_attributes = True


class DocumentListOut(BaseModel):
    """Paginated document list — returned by GET /documents/."""

    total: int = Field(description="Total number of documents matching the filters.")
    page: int
    page_size: int
    pages: int = Field(description="Total number of pages.")
    documents: list[DocumentOut]


class DocumentStatusOut(BaseModel):
    """Lightweight status response — returned by GET /documents/{id}/status."""

    id: str
    status: str
    document_class: str | None = None
    variant: str | None = None
    pipeline: list[PipelineEventOut] = Field(
        description="Ordered list of all pipeline state transitions for this document."
    )

    class Config:
        from_attributes = True


# ── Error response (shared shape) ─────────────────────────────────────────────

class ErrorOut(BaseModel):
    error: str
    message: str
    doc_url: str | None = None
