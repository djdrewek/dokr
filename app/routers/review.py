"""
Review queue router — manage documents in NEEDS_REVIEW state.

NEEDS_REVIEW documents are those that failed validation (NIGO conditions),
had a match failure, or triggered a manual-approval instruction rule.
Ops staff work this queue in the portal and either approve (re-queue to
a target pipeline stage) or reject (mark as FAILED with a reason).

GET  /review/                    — paginated NEEDS_REVIEW queue
GET  /review/{id}                — single document with NIGO reasons
POST /review/{id}/approve        — re-queue to a target stage
POST /review/{id}/reject         — mark FAILED with a reason
POST /documents/{id}/retry       — re-run from FAILED (wired here for co-location)
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.document import Document, PipelineEvent
from app.models.extracted_field import ExtractedField
from app.pipeline.states import PipelineState

router = APIRouter(tags=["Review Queue"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ReviewItemOut(BaseModel):
    id: str
    file_name: str
    document_class_id: Optional[str]
    document_class_name: Optional[str]
    variant_key: Optional[str]
    shipment_id: Optional[str]
    nigo_conditions: list[str]
    field_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ReviewQueueOut(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    items: list[ReviewItemOut]


class ApproveIn(BaseModel):
    target_stage: str = Field(
        description=(
            "Pipeline stage to re-queue the document to. "
            "Valid: EXTRACTING, VALIDATING, MATCHING, POSTING, COMPLETED."
        ),
        example="MATCHING",
    )
    approved_by: str = Field(description="Name or email of the approving operator.", example="ops@tata.co.uk")
    note: Optional[str] = Field(default=None, description="Optional note to record against the approval.")


class RejectIn(BaseModel):
    rejected_by: str = Field(description="Name or email of the rejecting operator.", example="ops@tata.co.uk")
    reason: str = Field(description="Reason for rejection. Stored in pipeline event trail.", example="Duplicate invoice — already paid under ERP ref INV-2025-4412.")


class ReviewActionOut(BaseModel):
    document_id: str
    previous_status: str
    new_status: str
    action: str        # "approved" | "rejected" | "retried"
    performed_by: str
    note: Optional[str]
    timestamp: datetime


# ── Allowed re-queue targets ──────────────────────────────────────────────────
VALID_APPROVE_TARGETS = {
    PipelineState.EXTRACTING,
    PipelineState.VALIDATING,
    PipelineState.MATCHING,
    PipelineState.POSTING,
    PipelineState.COMPLETED,
}


# ── GET /review/ ──────────────────────────────────────────────────────────────

@router.get(
    "/review/",
    response_model=ReviewQueueOut,
    summary="List the NEEDS_REVIEW queue",
    description=(
        "Returns all documents currently in NEEDS_REVIEW state, ordered by age (oldest first). "
        "Each item includes the NIGO conditions extracted from the most recent pipeline event "
        "so the operator knows exactly why the document was flagged."
    ),
)
def list_review_queue(
    document_class: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    q = db.query(Document).filter(Document.status == PipelineState.NEEDS_REVIEW)
    if document_class:
        q = q.filter(Document.document_class_id == document_class)

    total = q.count()
    docs = (
        q.order_by(Document.updated_at.asc())   # oldest first — most urgent
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for doc in docs:
        nigo = _extract_nigo_conditions(doc)
        fc = db.query(ExtractedField).filter(ExtractedField.document_id == doc.id).count()
        items.append(ReviewItemOut(
            id=doc.id,
            file_name=doc.file_name,
            document_class_id=doc.document_class_id,
            document_class_name=doc.document_class.name if doc.document_class else None,
            variant_key=doc.variant_key,
            shipment_id=doc.shipment_id,
            nigo_conditions=nigo,
            field_count=fc,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        ))

    return ReviewQueueOut(
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total else 0,
        items=items,
    )


# ── POST /review/{id}/approve ─────────────────────────────────────────────────

@router.post(
    "/review/{document_id}/approve",
    response_model=ReviewActionOut,
    summary="Approve a NEEDS_REVIEW document and re-queue it",
    description=(
        "Re-queues a document from NEEDS_REVIEW to the specified target stage. "
        "Use VALIDATING to re-run business rules after a human field correction. "
        "Use MATCHING to bypass a failed match. "
        "Use COMPLETED to accept the document as-is with a supervisor override."
    ),
)
def approve_document(
    document_id: str,
    body: ApproveIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_needs_review(db, document_id)

    # Validate target stage
    try:
        target = PipelineState(body.target_stage)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_target_stage",
                "message": (
                    f"'{body.target_stage}' is not a valid pipeline stage. "
                    f"Valid targets: {', '.join(s.value for s in VALID_APPROVE_TARGETS)}."
                ),
            },
        )
    if target not in VALID_APPROVE_TARGETS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_target_stage",
                "message": (
                    f"Cannot re-queue to {target.value} from review. "
                    f"Valid targets: {', '.join(s.value for s in VALID_APPROVE_TARGETS)}."
                ),
            },
        )

    previous = doc.status
    now = datetime.utcnow()
    doc.status = target
    doc.updated_at = now

    note_text = f" Note: {body.note}" if body.note else ""
    event = PipelineEvent(
        document_id=document_id,
        state=target,
        agent="HumanReviewAgent",
        detail=(
            f"APPROVED by {body.approved_by}. "
            f"Re-queued from NEEDS_REVIEW → {target.value}.{note_text}"
        ),
    )
    db.add(event)
    db.commit()

    # If re-queuing to a pipeline stage that needs the runner, kick it off.
    # For COMPLETED we just mark it done. For others we need to re-run the tail.
    if target != PipelineState.COMPLETED:
        from app.pipeline.runner import run_pipeline_from
        # Fetch original PDF from... in production: SharePoint. For scaffold: no-op re-run.
        # We pass empty bytes — the runner will skip PDF-dependent stages (dedup, classify)
        # and only run stages from target onwards.
        background_tasks.add_task(run_pipeline_from, document_id, target.value)

    return ReviewActionOut(
        document_id=document_id,
        previous_status=previous,
        new_status=target.value,
        action="approved",
        performed_by=body.approved_by,
        note=body.note,
        timestamp=now,
    )


# ── POST /review/{id}/reject ──────────────────────────────────────────────────

@router.post(
    "/review/{document_id}/reject",
    response_model=ReviewActionOut,
    summary="Reject a NEEDS_REVIEW document",
    description=(
        "Permanently rejects the document. Sets status to FAILED with the operator's reason. "
        "The document remains in the system for audit purposes but will not be re-processed "
        "unless manually retried via POST /documents/{id}/retry."
    ),
)
def reject_document(
    document_id: str,
    body: RejectIn,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_needs_review(db, document_id)
    previous = doc.status
    now = datetime.utcnow()
    doc.status = PipelineState.FAILED
    doc.updated_at = now

    event = PipelineEvent(
        document_id=document_id,
        state=PipelineState.FAILED,
        agent="HumanReviewAgent",
        detail=f"REJECTED by {body.rejected_by}. Reason: {body.reason}",
    )
    db.add(event)
    db.commit()

    return ReviewActionOut(
        document_id=document_id,
        previous_status=previous,
        new_status=PipelineState.FAILED,
        action="rejected",
        performed_by=body.rejected_by,
        note=body.reason,
        timestamp=now,
    )


# ── POST /documents/{id}/retry ────────────────────────────────────────────────

@router.post(
    "/documents/{document_id}/retry",
    response_model=ReviewActionOut,
    summary="Retry a FAILED document",
    description=(
        "Re-queues a FAILED document back to RECEIVED so the full pipeline runs again. "
        "Use after fixing an upstream issue (e.g. a malformed PDF was replaced, or "
        "a classification rule was updated). Only valid for documents in FAILED state."
    ),
)
def retry_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    retried_by: str = "system",
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=404,
            detail={"error": "document_not_found", "message": f"No document with ID '{document_id}'."},
        )
    if doc.status != PipelineState.FAILED:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_retryable",
                "message": (
                    f"Document '{document_id}' is in state {doc.status}, not FAILED. "
                    "Only FAILED documents can be retried. "
                    "For NEEDS_REVIEW documents, use POST /review/{id}/approve."
                ),
            },
        )

    previous = doc.status
    now = datetime.utcnow()
    doc.status = PipelineState.RECEIVED
    doc.updated_at = now

    event = PipelineEvent(
        document_id=document_id,
        state=PipelineState.RECEIVED,
        agent="RecoveryAgent",
        detail=f"Retry initiated by {retried_by}. Re-queuing from FAILED → RECEIVED.",
    )
    db.add(event)
    db.commit()

    # Re-run the full pipeline. In production: fetch PDF bytes from SharePoint.
    # For scaffold: re-run from CLASSIFYING (dedup is idempotent, SHA already stored).
    background_tasks.add_task(_retry_pipeline, document_id)

    return ReviewActionOut(
        document_id=document_id,
        previous_status=previous,
        new_status=PipelineState.RECEIVED,
        action="retried",
        performed_by=retried_by,
        note="Full pipeline re-queued from RECEIVED.",
        timestamp=now,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_needs_review(db: Session, document_id: str) -> Document:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=404,
            detail={"error": "document_not_found", "message": f"No document '{document_id}'."},
        )
    if doc.status != PipelineState.NEEDS_REVIEW:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_in_review",
                "message": (
                    f"Document '{document_id}' is in state {doc.status}, not NEEDS_REVIEW. "
                    "Only NEEDS_REVIEW documents can be approved or rejected."
                ),
            },
        )
    return doc


def _extract_nigo_conditions(doc: Document) -> list[str]:
    """Pull NIGO condition strings from the most recent NEEDS_REVIEW pipeline event."""
    review_events = [
        e for e in doc.pipeline_events
        if e.state == PipelineState.NEEDS_REVIEW and e.detail
    ]
    if not review_events:
        return []
    latest = review_events[-1].detail
    # Strip the "[AgentName] NIGO — N condition(s) failed. " prefix and split on " | "
    if "condition(s) failed." in latest:
        _, _, rest = latest.partition("condition(s) failed.")
        return [c.strip() for c in rest.split("|") if c.strip()]
    return [latest]


async def _retry_pipeline(document_id: str) -> None:
    """Re-run pipeline from scratch (without original PDF bytes — uses class override path)."""
    from app.pipeline.runner import run_pipeline
    # In production, fetch PDF bytes from SharePoint. For scaffold, pass empty bytes.
    # The pipeline will use document_class_override if set, or re-classify from scratch.
    run_pipeline(document_id, b"")
