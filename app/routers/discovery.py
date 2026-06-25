"""
Discovery router — surface CANDIDATE_NEW_CLASS documents to operators.

When the GovernanceAgent cannot match a document to any of the 20 known
Document Classes, it sets status=CANDIDATE_NEW_CLASS and populates
suggested_class_name + candidate_reason.  This queue presents those
documents to a human operator who can either:

  A. Promote → create / assign a new Document Class and re-queue extraction.
  B. Dismiss → route to NEEDS_REVIEW for manual field entry under the
               closest existing class.

Endpoints
─────────
GET  /v1/documents/discovery/
     List all CANDIDATE_NEW_CLASS documents with governance suggestion.

POST /v1/documents/{id}/promote-class
     Operator confirms a new class (or assigns an existing one).
     Optionally supply a new dc_xxx-style slug — if not supplied, one is
     generated.  Routes the document to EXTRACTING for a retry.

POST /v1/documents/{id}/dismiss-discovery
     Operator decides this is not worth a new class.
     Routes to NEEDS_REVIEW with a reason.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.document import Document, DocumentClass, PipelineEvent
from app.pipeline.states import PipelineState
from app.utils.ids import generate_document_id

router = APIRouter(tags=["Discovery"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class DiscoveryItemOut(BaseModel):
    id: str
    file_name: str
    file_size_bytes: int
    current_class_id: Optional[str]
    current_class_name: Optional[str]
    classification_confidence: Optional[float]
    suggested_class_name: Optional[str]
    candidate_reason: Optional[str]
    ai_governance_result: Optional[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DiscoveryQueueOut(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    items: list[DiscoveryItemOut]


class PromoteClassIn(BaseModel):
    confirmed_class_name: str = Field(
        description="Human-readable name for the new (or existing) document class.",
        example="CE Certificate / Declaration of Conformity",
    )
    confirmed_class_description: Optional[str] = Field(
        default=None,
        description="One-sentence description of the document type.",
        example="CE marking declarations and conformity certificates from EU suppliers.",
    )
    confirmed_class_id: Optional[str] = Field(
        default=None,
        description=(
            "If this document actually fits an existing class, supply its ID (e.g. dc_017). "
            "Leave null to auto-assign a new dc_xxx ID."
        ),
    )
    promoted_by: str = Field(
        description="Name or email of the operator making this decision.",
        example="ops@tata.co.uk",
    )
    note: Optional[str] = Field(
        default=None,
        description="Optional note to attach to the pipeline event.",
    )


class DismissDiscoveryIn(BaseModel):
    dismissed_by: str = Field(
        description="Name or email of the operator dismissing the discovery.",
        example="ops@tata.co.uk",
    )
    reason: str = Field(
        description="Why this is not a new class (routes document to NEEDS_REVIEW).",
        example="This is a one-off technical manual, not a recurring trade document type.",
    )
    nearest_class_id: Optional[str] = Field(
        default=None,
        description="If the document fits an existing class despite poor confidence, supply its ID.",
    )


class DiscoveryActionOut(BaseModel):
    document_id: str
    previous_status: str
    new_status: str
    action: str        # "promoted" | "dismissed"
    performed_by: str
    note: Optional[str]
    timestamp: datetime


# ── GET /documents/discovery/ ─────────────────────────────────────────────────

@router.get(
    "/documents/discovery/",
    response_model=DiscoveryQueueOut,
    summary="List the discovery queue (CANDIDATE_NEW_CLASS documents)",
    description=(
        "Returns all documents in CANDIDATE_NEW_CLASS state, ordered by age (oldest first). "
        "Each item includes the GovernanceAgent's suggested class name and reasoning "
        "so the operator can make an informed promote/dismiss decision."
    ),
)
def list_discovery_queue(
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    import json as _json

    q = (
        db.query(Document)
        .filter(Document.status == PipelineState.CANDIDATE_NEW_CLASS)
        .order_by(Document.updated_at.asc())
    )
    total = q.count()
    docs = q.offset((page - 1) * page_size).limit(page_size).all()

    items = []
    for doc in docs:
        gov_dict = None
        if doc.ai_governance_result:
            try:
                gov_dict = _json.loads(doc.ai_governance_result)
            except Exception:
                gov_dict = None

        items.append(DiscoveryItemOut(
            id=doc.id,
            file_name=doc.file_name,
            file_size_bytes=doc.file_size_bytes,
            current_class_id=doc.document_class_id,
            current_class_name=doc.document_class.name if doc.document_class else None,
            classification_confidence=doc.classification_confidence,
            suggested_class_name=doc.suggested_class_name,
            candidate_reason=doc.candidate_reason,
            ai_governance_result=gov_dict,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        ))

    return DiscoveryQueueOut(
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total else 0,
        items=items,
    )


# ── POST /documents/{id}/promote-class ───────────────────────────────────────

@router.post(
    "/documents/{document_id}/promote-class",
    response_model=DiscoveryActionOut,
    summary="Promote a discovery candidate to a Document Class and retry extraction",
    description=(
        "Operator confirms this is a recurring document type worth learning. "
        "Supply a class name (and optionally an existing class ID to re-assign to). "
        "If no confirmed_class_id is supplied, a new DocumentClass row is created "
        "with a generated ID. The document is then re-queued to EXTRACTING for a retry."
    ),
)
def promote_class(
    document_id: str,
    body: PromoteClassIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_candidate(db, document_id)
    previous = doc.status
    now = datetime.utcnow()

    if body.confirmed_class_id:
        # Assign to an existing known class
        existing = db.query(DocumentClass).filter(
            DocumentClass.id == body.confirmed_class_id
        ).first()
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "class_not_found",
                    "message": f"Document class '{body.confirmed_class_id}' does not exist.",
                },
            )
        new_class_id = body.confirmed_class_id
        new_class_name = existing.name
    else:
        # Create a new DocumentClass row
        new_class_id = _next_dc_id(db)
        slug = (
            body.confirmed_class_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")[:40]
        )
        new_class = DocumentClass(
            id=new_class_id,
            name=body.confirmed_class_name,
            slug=slug,
            treatment="STORE",   # default — ops can change via admin
            active=True,
        )
        db.add(new_class)
        new_class_name = body.confirmed_class_name

    # Update document
    doc.document_class_id = new_class_id
    doc.document_class_override = new_class_id   # pin so re-classify won't override
    doc.status = PipelineState.EXTRACTING
    doc.updated_at = now

    note_text = f" Note: {body.note}" if body.note else ""
    event = PipelineEvent(
        document_id=document_id,
        state=PipelineState.EXTRACTING,
        agent="HumanDiscoveryAgent",
        detail=(
            f"PROMOTED by {body.promoted_by}. "
            f"New class: {new_class_id} ('{new_class_name}'). "
            f"Re-queued from CANDIDATE_NEW_CLASS → EXTRACTING for retry.{note_text}"
        ),
    )
    db.add(event)
    db.commit()

    # Re-run extraction with the new class assignment
    background_tasks.add_task(_retry_extraction, document_id)

    return DiscoveryActionOut(
        document_id=document_id,
        previous_status=previous,
        new_status=PipelineState.EXTRACTING,
        action="promoted",
        performed_by=body.promoted_by,
        note=body.note or f"Assigned to new class {new_class_id}: '{new_class_name}'.",
        timestamp=now,
    )


# ── POST /documents/{id}/dismiss-discovery ────────────────────────────────────

@router.post(
    "/documents/{document_id}/dismiss-discovery",
    response_model=DiscoveryActionOut,
    summary="Dismiss a discovery candidate and route to NEEDS_REVIEW",
    description=(
        "Operator decides this document is not worth creating a new class for. "
        "The document routes to NEEDS_REVIEW for manual field entry. "
        "Optionally supply nearest_class_id to reassign to an existing class before review."
    ),
)
def dismiss_discovery(
    document_id: str,
    body: DismissDiscoveryIn,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_candidate(db, document_id)
    previous = doc.status
    now = datetime.utcnow()

    if body.nearest_class_id:
        existing = db.query(DocumentClass).filter(
            DocumentClass.id == body.nearest_class_id
        ).first()
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "class_not_found",
                    "message": f"Document class '{body.nearest_class_id}' does not exist.",
                },
            )
        doc.document_class_id = body.nearest_class_id
        doc.document_class_override = body.nearest_class_id

    doc.status = PipelineState.NEEDS_REVIEW
    doc.updated_at = now

    event = PipelineEvent(
        document_id=document_id,
        state=PipelineState.NEEDS_REVIEW,
        agent="HumanDiscoveryAgent",
        detail=(
            f"DISMISSED by {body.dismissed_by}. "
            f"Reason: {body.reason}. "
            f"Routed from CANDIDATE_NEW_CLASS → NEEDS_REVIEW for manual processing."
            + (f" Re-assigned to {body.nearest_class_id}." if body.nearest_class_id else "")
        ),
    )
    db.add(event)
    db.commit()

    return DiscoveryActionOut(
        document_id=document_id,
        previous_status=previous,
        new_status=PipelineState.NEEDS_REVIEW,
        action="dismissed",
        performed_by=body.dismissed_by,
        note=body.reason,
        timestamp=now,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_candidate(db: Session, document_id: str) -> Document:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=404,
            detail={"error": "document_not_found", "message": f"No document '{document_id}'."},
        )
    if doc.status != PipelineState.CANDIDATE_NEW_CLASS:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_in_discovery",
                "message": (
                    f"Document '{document_id}' is in state {doc.status}, "
                    "not CANDIDATE_NEW_CLASS. Only discovery candidates can be promoted or dismissed."
                ),
            },
        )
    return doc


def _next_dc_id(db: Session) -> str:
    """
    Generate the next dc_XXX ID by finding the highest existing numeric suffix.
    Starts at dc_021 if all 20 built-in classes are present.
    """
    rows = db.query(DocumentClass.id).all()
    existing = {r[0] for r in rows}
    for n in range(1, 999):
        candidate = f"dc_{n:03d}"
        if candidate not in existing:
            return candidate
    return f"dc_999"


async def _retry_extraction(document_id: str) -> None:
    """Re-run pipeline from EXTRACTING stage with empty bytes (uses class override)."""
    from app.pipeline.runner import run_pipeline
    # In production, fetch original PDF bytes from SharePoint.
    # For scaffold: pass empty bytes — the pipeline will use the class override.
    run_pipeline(document_id, b"")
