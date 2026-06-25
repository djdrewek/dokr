"""
Document Classes router — read-only access to the Document Class registry.

These records are seeded at startup from database.py and represent the
10 supported document types for Tata Limited's import/export flows.

GET /document_classes      — list all classes with variant counts
GET /document_classes/{id} — single class with full metadata and variants
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.document import DocumentClass, DocumentVariant

router = APIRouter(prefix="/document_classes", tags=["Document Classes"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocumentClassOut(BaseModel):
    id: str
    name: str
    slug: str
    treatment: str
    active: bool
    variant_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class VariantSummary(BaseModel):
    id: str
    variant_key: str
    learning_stage: str
    confirmed_instance_count: int
    avg_confidence: float | None
    active: bool


class DocumentClassDetailOut(BaseModel):
    id: str
    name: str
    slug: str
    treatment: str
    active: bool
    created_at: datetime
    variants: list[VariantSummary]


class DocumentClassListOut(BaseModel):
    total: int
    classes: list[DocumentClassOut]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=DocumentClassListOut,
    summary="List all Document Classes",
    description=(
        "Returns the full Document Class registry with variant counts per class. "
        "The `treatment` field controls the pipeline route: PROCESS (extract + match), "
        "STORE (archive only), STORE_AND_FORWARD (archive + transmit), or GENERATED "
        "(system-produced output)."
    ),
)
def list_document_classes(
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    classes = (
        db.query(DocumentClass)
        .order_by(DocumentClass.id)
        .all()
    )

    result = []
    for cls in classes:
        variant_count = (
            db.query(DocumentVariant)
            .filter(DocumentVariant.document_class_id == cls.id)
            .count()
        )
        result.append(DocumentClassOut(
            id=cls.id,
            name=cls.name,
            slug=cls.slug,
            treatment=cls.treatment,
            active=cls.active,
            variant_count=variant_count,
            created_at=cls.created_at,
        ))

    return DocumentClassListOut(total=len(result), classes=result)


@router.get(
    "/{class_id}",
    response_model=DocumentClassDetailOut,
    summary="Get a Document Class with its Variants",
    description=(
        "Returns a single Document Class with the full list of Variants learned "
        "for that class. Use this to audit which sender layouts have accumulated "
        "enough confirmed instances to advance through the learning stages."
    ),
)
def get_document_class(
    class_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    cls = db.query(DocumentClass).filter(DocumentClass.id == class_id).first()
    if not cls:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "document_class_not_found",
                "message": (
                    f"No Document Class with ID '{class_id}'. "
                    f"Call GET /document_classes to see available class IDs."
                ),
                "doc_url": "https://docs.dokr.io/errors#document_class_not_found",
            },
        )

    variants = (
        db.query(DocumentVariant)
        .filter(DocumentVariant.document_class_id == class_id)
        .order_by(DocumentVariant.confirmed_instance_count.desc())
        .all()
    )

    return DocumentClassDetailOut(
        id=cls.id,
        name=cls.name,
        slug=cls.slug,
        treatment=cls.treatment,
        active=cls.active,
        created_at=cls.created_at,
        variants=[
            VariantSummary(
                id=v.id,
                variant_key=v.variant_key,
                learning_stage=v.learning_stage,
                confirmed_instance_count=v.confirmed_instance_count,
                avg_confidence=v.avg_confidence,
                active=v.active,
            )
            for v in variants
        ],
    )
