"""
Shipments router — read/manage ShipmentRecords.

A ShipmentRecord groups all documents belonging to the same physical
shipment (keyed by shared PO number, AWB, or invoice reference).

GET /shipments/          — list shipments with filters
GET /shipments/{id}      — single shipment with full match breakdown
GET /shipments/{id}/documents — documents linked to a shipment
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.document import Document
from app.models.shipment import ShipmentRecord

router = APIRouter(prefix="/shipments", tags=["Shipments"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class MatchCheckOut(BaseModel):
    name: str
    status: str      # PASS | FAIL | SKIP
    detail: str


class ShipmentOut(BaseModel):
    id: str
    reference_key: str
    all_reference_keys: list[str]
    document_ids: list[str]
    document_count: int
    class_summary: dict
    match_result: Optional[str]
    erp_reference: Optional[str]
    erp_posted_at: Optional[datetime]
    sharepoint_path: Optional[str]
    sharepoint_filed_at: Optional[datetime]
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ShipmentDetailOut(ShipmentOut):
    match_checks: list[MatchCheckOut] = []
    match_summary: Optional[str] = None


class ShipmentListOut(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    shipments: list[ShipmentOut]


class ShipmentDocumentSummary(BaseModel):
    id: str
    file_name: str
    status: str
    document_class_id: Optional[str]
    document_class_name: Optional[str]
    field_count: int
    created_at: datetime


class ShipmentDocumentsOut(BaseModel):
    shipment_id: str
    reference_key: str
    total: int
    documents: list[ShipmentDocumentSummary]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=ShipmentListOut,
    summary="List shipment records",
    description=(
        "Returns all ShipmentRecords with optional filters. "
        "A ShipmentRecord groups every document that shares a PO number, AWB, or invoice reference. "
        "Use match_result to find shipments awaiting three-way match resolution."
    ),
)
def list_shipments(
    status: Optional[str] = Query(default=None, description="Filter by shipment status: OPEN, MATCHED, POSTED, FILED, COMPLETE."),
    match_result: Optional[str] = Query(default=None, description="Filter by match result: PASS, PASS_PARTIAL, FAIL, or 'null' for unmatched."),
    reference_key: Optional[str] = Query(default=None, description="Partial match on reference_key (e.g. TSL/58237)."),
    date_from: Optional[str] = Query(default=None, description="Earliest created_at (ISO 8601)."),
    date_to: Optional[str] = Query(default=None, description="Latest created_at (ISO 8601)."),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    from datetime import timedelta
    q = db.query(ShipmentRecord)

    if status:
        q = q.filter(ShipmentRecord.status == status)
    if match_result:
        if match_result.lower() == "null":
            q = q.filter(ShipmentRecord.match_result.is_(None))
        else:
            q = q.filter(ShipmentRecord.match_result == match_result)
    if reference_key:
        q = q.filter(ShipmentRecord.reference_key.contains(reference_key))
    if date_from:
        try:
            q = q.filter(ShipmentRecord.created_at >= datetime.fromisoformat(date_from))
        except ValueError:
            raise HTTPException(422, detail={"error": "invalid_date", "message": f"date_from '{date_from}' is not valid ISO 8601."})
    if date_to:
        try:
            q = q.filter(ShipmentRecord.created_at < datetime.fromisoformat(date_to) + timedelta(days=1))
        except ValueError:
            raise HTTPException(422, detail={"error": "invalid_date", "message": f"date_to '{date_to}' is not valid ISO 8601."})

    total = q.count()
    records = q.order_by(ShipmentRecord.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    return ShipmentListOut(
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total else 0,
        shipments=[_to_shipment_out(s) for s in records],
    )


@router.get(
    "/{shipment_id}",
    response_model=ShipmentDetailOut,
    summary="Get a shipment record with full match breakdown",
    description=(
        "Returns a single ShipmentRecord with the seven-check match detail breakdown, "
        "ERP reference, and SharePoint filing path. Use this to audit why a shipment "
        "is in PASS_PARTIAL or FAIL state."
    ),
)
def get_shipment(
    shipment_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    s = _get_or_404(db, shipment_id)

    checks = []
    summary = None
    if s.match_detail:
        summary = s.match_detail.get("summary")
        for c in s.match_detail.get("checks", []):
            checks.append(MatchCheckOut(
                name=c["name"],
                status=c["status"],
                detail=c["detail"],
            ))

    out = ShipmentDetailOut(
        **_to_shipment_out(s).model_dump(),
        match_checks=checks,
        match_summary=summary,
    )
    return out


@router.get(
    "/{shipment_id}/documents",
    response_model=ShipmentDocumentsOut,
    summary="List documents linked to a shipment",
    description=(
        "Returns all documents that have been linked to this ShipmentRecord, "
        "with their current pipeline status and field count. Use this to check "
        "which documents in the set have completed and which are still processing."
    ),
)
def get_shipment_documents(
    shipment_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    s = _get_or_404(db, shipment_id)

    from app.models.extracted_field import ExtractedField
    doc_ids = s.document_ids or []
    docs = db.query(Document).filter(Document.id.in_(doc_ids)).order_by(Document.created_at).all()

    summaries = []
    for doc in docs:
        fc = db.query(ExtractedField).filter(ExtractedField.document_id == doc.id).count()
        summaries.append(ShipmentDocumentSummary(
            id=doc.id,
            file_name=doc.file_name,
            status=doc.status,
            document_class_id=doc.document_class_id,
            document_class_name=doc.document_class.name if doc.document_class else None,
            field_count=fc,
            created_at=doc.created_at,
        ))

    return ShipmentDocumentsOut(
        shipment_id=s.id,
        reference_key=s.reference_key,
        total=len(summaries),
        documents=summaries,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_404(db: Session, shipment_id: str) -> ShipmentRecord:
    s = db.query(ShipmentRecord).filter(ShipmentRecord.id == shipment_id).first()
    if not s:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "shipment_not_found",
                "message": f"No ShipmentRecord with ID '{shipment_id}'.",
                "doc_url": "https://docs.dokr.io/errors#shipment_not_found",
            },
        )
    return s


def _to_shipment_out(s: ShipmentRecord) -> ShipmentOut:
    return ShipmentOut(
        id=s.id,
        reference_key=s.reference_key,
        all_reference_keys=s.all_reference_keys or [],
        document_ids=s.document_ids or [],
        document_count=len(s.document_ids or []),
        class_summary=s.class_summary or {},
        match_result=s.match_result,
        erp_reference=s.erp_reference,
        erp_posted_at=s.erp_posted_at,
        sharepoint_path=s.sharepoint_path,
        sharepoint_filed_at=s.sharepoint_filed_at,
        status=s.status,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )
