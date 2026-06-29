from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import get_db
from app.models.document import Document, PipelineEvent
from app.models.extracted_field import ExtractedField
from app.pipeline.states import PipelineState
from app.schemas.document import DocumentListOut, DocumentOut, DocumentStatusOut, PipelineEventOut
from app.schemas.fields import (
    DocumentFieldsOut,
    ExtractedFieldOut,
    FieldCorrectionIn,
    FieldCorrectionOut,
    TableOut,
)
from app.utils.files import read_and_validate_pdf
from app.utils.hashing import sha256_bytes
from app.utils.ids import generate_document_id

router = APIRouter(prefix="/documents", tags=["Documents"])


# ── GET /documents/ ──────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=DocumentListOut,
    summary="Search and list documents",
    description=(
        "List documents with optional filters. Combine any of: status, document_class, "
        "shipment_id, field_name+field_value (extracted field search with * wildcard), "
        "date_from/date_to (ISO 8601 date, filters on created_at). "
        "Results are paginated — use page and page_size to navigate."
    ),
)
def list_documents(
    status: Optional[str] = Query(default=None, description="Filter by pipeline status, e.g. COMPLETED, NEEDS_REVIEW."),
    document_class: Optional[str] = Query(default=None, description="Filter by Document Class ID, e.g. dc_006."),
    shipment_id: Optional[str] = Query(default=None, description="Filter by ShipmentRecord ID."),
    field_name: Optional[str] = Query(default=None, description="Field name to search within extracted fields."),
    field_value: Optional[str] = Query(default=None, description="Field value to match. Use * as a suffix wildcard, e.g. 'E/IC*'."),
    date_from: Optional[str] = Query(default=None, description="Earliest created_at date (inclusive), ISO 8601 e.g. 2026-06-01."),
    date_to: Optional[str] = Query(default=None, description="Latest created_at date (inclusive), ISO 8601 e.g. 2026-06-30."),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page (max 100)."),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    import fnmatch

    query = db.query(Document)

    # ── Field-level search (join ExtractedField) ──────────────────────────────
    if field_name and field_value:
        # Collect document IDs where the field matches
        ef_query = db.query(ExtractedField.document_id).filter(
            ExtractedField.field_name == field_name
        )
        if "*" in field_value:
            # Wildcard: convert to SQL LIKE
            like_pattern = field_value.replace("*", "%")
            ef_query = ef_query.filter(
                or_(
                    ExtractedField.field_value.like(like_pattern),
                    ExtractedField.corrected_value.like(like_pattern),
                )
            )
        else:
            ef_query = ef_query.filter(
                or_(
                    ExtractedField.field_value == field_value,
                    ExtractedField.corrected_value == field_value,
                )
            )
        matching_ids = [row[0] for row in ef_query.all()]
        query = query.filter(Document.id.in_(matching_ids))

    elif field_name and not field_value:
        # Just filter by field existence
        ef_ids = db.query(ExtractedField.document_id).filter(
            ExtractedField.field_name == field_name
        ).all()
        query = query.filter(Document.id.in_([r[0] for r in ef_ids]))

    # ── Standard filters ──────────────────────────────────────────────────────
    if status:
        query = query.filter(Document.status == status)
    if document_class:
        query = query.filter(Document.document_class_id == document_class)
    if shipment_id:
        query = query.filter(Document.shipment_id == shipment_id)
    if date_from:
        try:
            df = datetime.fromisoformat(date_from)
            query = query.filter(Document.created_at >= df)
        except ValueError:
            raise HTTPException(status_code=422, detail={"error": "invalid_date", "message": f"date_from '{date_from}' is not a valid ISO 8601 date."})
    if date_to:
        try:
            dt = datetime.fromisoformat(date_to)
            # Include the full end date day
            from datetime import timedelta
            query = query.filter(Document.created_at < dt + timedelta(days=1))
        except ValueError:
            raise HTTPException(status_code=422, detail={"error": "invalid_date", "message": f"date_to '{date_to}' is not a valid ISO 8601 date."})

    # ── Pagination ────────────────────────────────────────────────────────────
    total = query.count()
    docs = (
        query.order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return DocumentListOut(
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total > 0 else 0,
        documents=[_to_document_out(d) for d in docs],
    )


# ── POST /documents/submit ────────────────────────────────────────────────────

@router.post(
    "/submit",
    response_model=DocumentOut,
    status_code=200,
    summary="Submit a document for processing",
    description=(
        "Submit a PDF document to the Dokr pipeline. The document is validated, "
        "hashed for deduplication, assigned an ID, and queued for classification "
        "and extraction. Returns immediately with the document record in RECEIVED "
        "state. Processing is asynchronous — subscribe to the document.completed "
        "webhook event or poll GET /documents/{id}/status for pipeline progress."
    ),
)
async def submit_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file. Max 50 MB."),
    document_class: str | None = Form(
        default=None,
        description="Override automatic classification. Must match an existing Document Class ID.",
    ),
    variant_key: str | None = Form(
        default=None,
        description="Force assignment to a specific Document Variant key.",
    ),
    priority: str = Form(
        default="standard",
        description="standard (default) or express. Express targets 60-second extraction.",
    ),
    webhook_url: str | None = Form(
        default=None,
        description="Override the account-level webhook URL for this document only.",
    ),
    metadata: str | None = Form(
        default=None,
        description="JSON string of arbitrary key-value pairs to attach to the document record.",
    ),
    skip_stages: str | None = Form(
        default=None,
        description=(
            "Comma-separated list of pipeline stages to skip for this document. "
            "e.g. 'MATCHING,POSTING' to store the document without ERP posting. "
            "STORE-treatment document classes automatically skip MATCHING and POSTING. "
            "Valid values: MATCHING, POSTING."
        ),
    ),
    submitter_email: str | None = Form(
        default=None,
        description=(
            "Email address of the person submitting this document. "
            "Populated automatically by the Outlook add-in from the signed-in user's identity. "
            "Used to send failure notifications when the document lands in NEEDS_REVIEW."
        ),
    ),
    match_mode: str = Form(
        default="REQUIRED",
        description=(
            "Controls three-way match behaviour. "
            "REQUIRED (default): a match FAIL halts the pipeline and sends the document to NEEDS_REVIEW. "
            "ADVISORY: the match runs and the result is recorded, but the pipeline always continues to POSTING regardless of outcome. "
            "SKIP: matching is bypassed entirely (equivalent to adding MATCHING to skip_stages)."
        ),
    ),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    import json

    # Validate match_mode
    VALID_MATCH_MODES = {"REQUIRED", "ADVISORY", "SKIP"}
    match_mode = match_mode.strip().upper()
    if match_mode not in VALID_MATCH_MODES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_match_mode",
                "message": f"match_mode must be one of: {', '.join(sorted(VALID_MATCH_MODES))}. Got: '{match_mode}'.",
                "doc_url": "https://docs.dokr.io/errors#invalid_match_mode",
            },
        )

    # Parse skip_stages
    VALID_SKIP = {"MATCHING", "POSTING"}
    parsed_skip_stages: list[str] | None = None
    if skip_stages:
        requested = [s.strip().upper() for s in skip_stages.split(",") if s.strip()]
        invalid = set(requested) - VALID_SKIP
        if invalid:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_skip_stages",
                    "message": f"Unknown stage(s) in skip_stages: {', '.join(sorted(invalid))}. Valid: {', '.join(sorted(VALID_SKIP))}.",
                    "doc_url": "https://docs.dokr.io/errors#invalid_skip_stages",
                },
            )
        parsed_skip_stages = requested if requested else None

    # Validate priority
    if priority not in ("standard", "express"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_priority",
                "message": "priority must be 'standard' or 'express'.",
                "doc_url": "https://docs.dokr.io/errors#invalid_priority",
            },
        )

    # Parse metadata JSON if provided
    parsed_metadata: dict = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_metadata",
                    "message": "metadata must be a valid JSON object (key-value pairs).",
                    "doc_url": "https://docs.dokr.io/errors#invalid_metadata",
                },
            )

    # Validate and read the PDF
    pdf_bytes = await read_and_validate_pdf(file)

    # Hash for deduplication (SHA-256 of raw bytes)
    file_hash = sha256_bytes(pdf_bytes)

    # Check for exact duplicate (byte-level match)
    existing = db.query(Document).filter(Document.file_sha256 == file_hash).first()
    if existing:
        # Return the original document record with a clear duplicate signal
        # The pipeline will formally set status = EXACT_DUPLICATE, but for the
        # submit response we surface the cached document immediately.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "exact_duplicate",
                "message": (
                    "This document has already been processed. "
                    "The original document record is referenced below."
                ),
                "original_document_id": existing.id,
                "original_status": existing.status,
                "doc_url": "https://docs.dokr.io/errors#exact_duplicate",
            },
        )

    # Generate document ID
    doc_id = generate_document_id()

    # Create document record
    doc = Document(
        id=doc_id,
        status=PipelineState.RECEIVED,
        document_class_override=document_class,
        variant_key=variant_key,
        file_name=file.filename or "upload.pdf",
        file_size_bytes=len(pdf_bytes),
        file_sha256=file_hash,
        priority=priority,
        doc_metadata=parsed_metadata,
        webhook_url=webhook_url,
        skip_stages=parsed_skip_stages,
        submitter_email=submitter_email or None,
        match_mode=match_mode,
    )
    db.add(doc)

    # Record initial pipeline event
    event = PipelineEvent(
        document_id=doc_id,
        state=PipelineState.RECEIVED,
        agent="IngestionAgent",
        detail=f"Document received. {len(pdf_bytes):,} bytes. SHA-256: {file_hash[:16]}…",
    )
    db.add(event)
    db.commit()
    db.refresh(doc)

    # Kick off the pipeline as a background task — returns RECEIVED immediately,
    # client polls /status or waits for webhook to see progress.
    from app.pipeline.runner import run_pipeline
    background_tasks.add_task(run_pipeline, doc.id, pdf_bytes)

    return _to_document_out(doc)


# ── GET /documents/{id} ───────────────────────────────────────────────────────

@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Retrieve a document record",
    description="Returns the full document record including current pipeline status, "
                "assigned Document Class and Variant, and submitted metadata.",
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_or_404(db, document_id)
    return _to_document_out(doc)


# ── GET /documents/{id}/status ────────────────────────────────────────────────

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusOut,
    summary="Get pipeline status and event timeline",
    description="Returns the current pipeline state and the full ordered timeline of "
                "every state transition the document has passed through, including "
                "which agent triggered each transition.",
)
def get_document_status(
    document_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_or_404(db, document_id)

    pipeline_events = [
        PipelineEventOut(
            state=e.state,
            agent=e.agent,
            detail=e.detail,
            timestamp=e.created_at,
        )
        for e in doc.pipeline_events
    ]

    return DocumentStatusOut(
        id=doc.id,
        status=doc.status,
        document_class=doc.document_class_id,
        variant=doc.variant_id,
        pipeline=pipeline_events,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

# ── GET /documents/{id}/fields ────────────────────────────────────────────────

@router.get(
    "/{document_id}/fields",
    response_model=DocumentFieldsOut,
    summary="Get extracted fields with confidence and provenance",
    description="Returns all extracted fields for a document with per-field confidence "
                "scores, extraction model, method (pattern vs zero-shot), and any "
                "human corrections applied. Only available once status reaches EXTRACTING "
                "or later. Satisfies FR-027f field-level lineage.",
)
def get_document_fields(
    document_id: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    doc = _get_or_404(db, document_id)

    if doc.status in (
        PipelineState.RECEIVED,
        PipelineState.DEDUPLICATING,
        PipelineState.CLASSIFYING,
        PipelineState.UNCLASSIFIED,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "extraction_not_complete",
                "message": (
                    f"Extraction has not run yet. Current status: {doc.status}. "
                    "Poll /status and retry once the document reaches EXTRACTING or later."
                ),
                "doc_url": "https://docs.dokr.io/errors#extraction_not_complete",
            },
        )

    fields = (
        db.query(ExtractedField)
        .filter(ExtractedField.document_id == document_id)
        .order_by(ExtractedField.field_name)
        .all()
    )

    avg_confidence = (
        round(sum(f.confidence for f in fields) / len(fields), 4) if fields else None
    )

    import json as _json
    field_outs = []
    table_outs = []

    for f in fields:
        ftype = getattr(f, "field_type", "scalar") or "scalar"
        if ftype == "table":
            rows = []
            try:
                rows = _json.loads(f.field_value or "[]")
            except Exception:
                rows = []
            table_outs.append(TableOut(
                table_name=f.field_name,
                columns=list(rows[0].keys()) if rows else [],
                rows=rows,
                row_count=len(rows),
                confidence=f.confidence,
                extraction_method=f.extraction_method,
            ))
        else:
            field_outs.append(ExtractedFieldOut(
                field_name=f.field_name,
                field_value=f.field_value,
                field_type="scalar",
                confidence=f.confidence,
                extraction_model=f.extraction_model,
                extraction_method=f.extraction_method,
                human_corrected=f.human_corrected,
                corrected_value=f.corrected_value,
                used_in_match=f.used_in_match,
                match_result=f.match_result,
                extracted_at=f.created_at,
            ))

    # ── Page sampling stats ───────────────────────────────────────────────────
    pages_total   = getattr(doc, "pages_total",         None)
    pages_skipped = getattr(doc, "pages_skipped_count", None)
    pages_sampled_json = getattr(doc, "pages_sampled_json", None)
    pages_sampled = len(_json.loads(pages_sampled_json)) if pages_sampled_json else None

    # PageProfileAgent stage for this variant
    page_profile_stage: str | None = None
    if doc.variant_id:
        try:
            from app.agents.page_profile import PageProfileAgent as _PPA
            page_profile_stage = _PPA(db).get_profile_stage(doc.variant_id)
        except Exception:
            pass

    return DocumentFieldsOut(
        document_id=doc.id,
        document_class=doc.document_class_id,
        document_class_name=doc.document_class.name if doc.document_class else None,
        variant_key=doc.variant_key,
        learning_stage=doc.variant.learning_stage if doc.variant else None,
        field_count=len(fields),
        avg_confidence=avg_confidence,
        fields=field_outs,
        tables=table_outs,
        pages_total=pages_total,
        pages_sampled=pages_sampled,
        pages_skipped=pages_skipped,
        page_profile_stage=page_profile_stage,
    )


# ── PATCH /documents/{id}/fields/{field_name}/correct ─────────────────────────

@router.patch(
    "/{document_id}/fields/{field_name}/correct",
    response_model=FieldCorrectionOut,
    summary="Apply a human correction to an extracted field",
    description=(
        "Override the AI-extracted value for a single field with a human-verified value. "
        "Recording a correction closes the learning loop: the document's Variant receives "
        "a confirmed instance count increment, which may advance its learning stage from "
        "ZERO_SHOT → LEARNING → LEARNED → OPTIMISED (FR-UEE-008, FR-027f). "
        "Corrections are stored with full provenance: who corrected, when, and what changed."
    ),
)
def correct_field(
    document_id: str,
    field_name: str,
    body: FieldCorrectionIn,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    from datetime import datetime
    from app.agents.variant_discovery import VariantDiscoveryAgent
    from app.models.document import DocumentVariant, PipelineEvent

    doc = _get_or_404(db, document_id)

    # Must have reached at least EXTRACTING
    if doc.status in (
        PipelineState.RECEIVED,
        PipelineState.DEDUPLICATING,
        PipelineState.CLASSIFYING,
        PipelineState.UNCLASSIFIED,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "extraction_not_complete",
                "message": (
                    f"Cannot correct fields before extraction. "
                    f"Current status: {doc.status}."
                ),
                "doc_url": "https://docs.dokr.io/errors#extraction_not_complete",
            },
        )

    # Find the extracted field
    ef = (
        db.query(ExtractedField)
        .filter(
            ExtractedField.document_id == document_id,
            ExtractedField.field_name == field_name,
        )
        .first()
    )
    if not ef:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "field_not_found",
                "message": (
                    f"Field '{field_name}' not found on document '{document_id}'. "
                    "Call GET /fields to see available field names."
                ),
                "doc_url": "https://docs.dokr.io/errors#field_not_found",
            },
        )

    original_value = ef.field_value
    now = datetime.utcnow()

    # Apply correction
    ef.human_corrected = True
    ef.corrected_value = body.corrected_value
    ef.corrected_by = body.corrected_by
    ef.corrected_at = now

    # ── Learning loop ─────────────────────────────────────────────────────────
    # Only increment variant count on the FIRST correction per document
    # (correction is per-field but variant progress tracks per-document).
    # Check how many corrected fields already exist (before this commit).
    prior_corrections = (
        db.query(ExtractedField)
        .filter(
            ExtractedField.document_id == document_id,
            ExtractedField.human_corrected == True,
            ExtractedField.field_name != field_name,
        )
        .count()
    )

    stage_before: str | None = None
    stage_after: str | None = None
    variant_id = doc.variant_id
    confirmed_count: int | None = None

    if variant_id:
        variant = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
        if variant:
            stage_before = variant.learning_stage
            # Increment on first correction for this document
            if prior_corrections == 0:
                variant_agent = VariantDiscoveryAgent(db)
                variant_agent.increment_confirmed_instance(variant_id)
                db.refresh(variant)
            stage_after = variant.learning_stage
            confirmed_count = variant.confirmed_instance_count

    db.commit()

    # Log pipeline event
    detail = (
        f"Field '{field_name}' corrected by {body.corrected_by}. "
        f"Original: '{original_value}' → Corrected: '{body.corrected_value}'."
    )
    if stage_before != stage_after:
        detail += f" Variant advanced: {stage_before} → {stage_after}."

    event = PipelineEvent(
        document_id=document_id,
        state=doc.status,
        agent="HumanReviewAgent",
        detail=detail,
    )
    db.add(event)
    db.commit()

    return FieldCorrectionOut(
        document_id=document_id,
        field_name=field_name,
        original_value=original_value,
        corrected_value=body.corrected_value,
        corrected_by=body.corrected_by,
        corrected_at=now,
        variant_id=variant_id,
        learning_stage_before=stage_before,
        learning_stage_after=stage_after,
        confirmed_instance_count=confirmed_count,
    )


def _get_or_404(db: Session, document_id: str) -> Document:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "document_not_found",
                "message": f"No document with ID '{document_id}' exists.",
                "doc_url": "https://docs.dokr.io/errors#document_not_found",
            },
        )
    return doc


def _to_document_out(doc: Document) -> DocumentOut:
    import json as _json
    gov_result = None
    if doc.ai_governance_result:
        try:
            gov_result = _json.loads(doc.ai_governance_result) if isinstance(doc.ai_governance_result, str) else doc.ai_governance_result
        except Exception:
            gov_result = None

    return DocumentOut(
        id=doc.id,
        status=doc.status,
        document_class=doc.document_class_id,
        document_class_name=doc.document_class.name if doc.document_class else None,
        variant=doc.variant_id,
        variant_key=doc.variant_key,
        file_name=doc.file_name,
        file_size_bytes=doc.file_size_bytes,
        priority=doc.priority,
        metadata=doc.doc_metadata or {},
        webhook_url=doc.webhook_url,
        shipment_id=doc.shipment_id,
        skip_stages=doc.skip_stages,
        match_mode=getattr(doc, "match_mode", "REQUIRED"),
        classification_confidence=doc.classification_confidence,
        suggested_class_name=doc.suggested_class_name,
        candidate_reason=doc.candidate_reason,
        ai_governance_result=gov_result,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


