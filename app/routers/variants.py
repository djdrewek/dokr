"""
Variants router — surfaces Document Variant management data.

GET /variants             — list all variants, filterable by class
GET /variants/{id}        — single variant with learning metrics
GET /variants/{id}/documents — documents processed under this variant (paginated)
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.document import Document, DocumentVariant

router = APIRouter(prefix="/variants", tags=["Variants"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class VariantOut(BaseModel):
    id: str
    document_class_id: str
    variant_key: str
    learning_stage: str
    confirmed_instance_count: int
    avg_confidence: float | None
    touchless_rate: float | None
    template_frozen: bool
    active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VariantListOut(BaseModel):
    total: int
    variants: list[VariantOut]


class VariantDocumentSummary(BaseModel):
    id: str
    status: str
    file_name: str
    priority: str
    created_at: datetime


class VariantDocumentsOut(BaseModel):
    variant_id: str
    variant_key: str
    learning_stage: str
    total: int
    page: int
    page_size: int
    documents: list[VariantDocumentSummary]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=VariantListOut,
    summary="List Document Variants",
    description=(
        "Returns all Document Variants across all Document Classes. "
        "Filter by `document_class_id` to scope results. "
        "Variants progress through ZERO_SHOT → LEARNING → LEARNED → OPTIMISED "
        "as confirmed instances accumulate."
    ),
)
def list_variants(
    document_class_id: str | None = Query(
        default=None,
        description="Filter variants by Document Class ID (e.g. 'dc_006').",
    ),
    learning_stage: str | None = Query(
        default=None,
        description="Filter by learning stage: ZERO_SHOT, LEARNING, LEARNED, OPTIMISED.",
    ),
    active_only: bool = Query(
        default=True,
        description="When true (default), only returns active variants.",
    ),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    q = db.query(DocumentVariant)

    if document_class_id:
        q = q.filter(DocumentVariant.document_class_id == document_class_id)
    if learning_stage:
        valid_stages = {"ZERO_SHOT", "LEARNING", "LEARNED", "OPTIMISED"}
        if learning_stage not in valid_stages:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_learning_stage",
                    "message": f"learning_stage must be one of: {', '.join(sorted(valid_stages))}.",
                    "doc_url": "https://docs.dokr.io/errors#invalid_learning_stage",
                },
            )
        q = q.filter(DocumentVariant.learning_stage == learning_stage)
    if active_only:
        q = q.filter(DocumentVariant.active == True)

    variants = q.order_by(DocumentVariant.created_at.desc()).all()

    return VariantListOut(
        total=len(variants),
        variants=[_to_variant_out(v) for v in variants],
    )


@router.get(
    "/{variant_id}",
    response_model=VariantOut,
    summary="Get a Document Variant",
    description=(
        "Returns a single Document Variant record with full learning metrics. "
        "The `confirmed_instance_count` drives stage advancement. "
        "The `touchless_rate` and `avg_confidence` fields populate as documents "
        "accumulate — null until at least one document is processed."
    ),
)
def get_variant(
    variant_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    variant = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
    if not variant:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "variant_not_found",
                "message": f"No variant with ID '{variant_id}' exists.",
                "doc_url": "https://docs.dokr.io/errors#variant_not_found",
            },
        )
    return _to_variant_out(variant)


@router.get(
    "/{variant_id}/documents",
    response_model=VariantDocumentsOut,
    summary="List documents processed under a variant",
    description=(
        "Returns all document instances processed under a given variant, "
        "most recent first. Useful for auditing extraction quality as the "
        "variant progresses through learning stages."
    ),
)
def list_variant_documents(
    variant_id: str,
    page: int = Query(default=1, ge=1, description="Page number (1-based)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page (max 100)."),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    variant = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
    if not variant:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "variant_not_found",
                "message": f"No variant with ID '{variant_id}' exists.",
                "doc_url": "https://docs.dokr.io/errors#variant_not_found",
            },
        )

    total = (
        db.query(Document)
        .filter(Document.variant_id == variant_id)
        .count()
    )

    documents = (
        db.query(Document)
        .filter(Document.variant_id == variant_id)
        .order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return VariantDocumentsOut(
        variant_id=variant_id,
        variant_key=variant.variant_key,
        learning_stage=variant.learning_stage,
        total=total,
        page=page,
        page_size=page_size,
        documents=[
            VariantDocumentSummary(
                id=d.id,
                status=d.status,
                file_name=d.file_name,
                priority=d.priority,
                created_at=d.created_at,
            )
            for d in documents
        ],
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_variant_out(v: DocumentVariant) -> VariantOut:
    return VariantOut(
        id=v.id,
        document_class_id=v.document_class_id,
        variant_key=v.variant_key,
        learning_stage=v.learning_stage,
        confirmed_instance_count=v.confirmed_instance_count,
        avg_confidence=v.avg_confidence,
        touchless_rate=v.touchless_rate,
        template_frozen=v.template_frozen,
        active=v.active,
        created_at=v.created_at,
        updated_at=v.updated_at,
    )
