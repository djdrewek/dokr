"""
Local Dokr admin dashboard — http://localhost:8000/dashboard

No API-key gate (localhost only). Designed to be used during setup and operations.

Pages
─────
  GET  /dashboard              overview   — stats + recent activity
  GET  /dashboard/queue        queue      — all documents, live-polling
  GET  /dashboard/review       review     — NEEDS_REVIEW queue + approve/reject
  GET  /dashboard/discovery    discovery  — CANDIDATE_NEW_CLASS + promote/dismiss
  GET  /dashboard/ingest       ingest     — drag-and-drop upload + pipeline trace
  GET  /dashboard/system       system     — AI pipeline modes + config

HTMX partials (polled / swap targets)
──────────────────────────────────────
  GET  /dashboard/partials/stats
  GET  /dashboard/partials/queue-rows
  GET  /dashboard/partials/review-rows
  GET  /dashboard/partials/discovery-rows
  GET  /dashboard/partials/doc-status/{id}

Actions (return HTML fragments)
────────────────────────────────
  POST /dashboard/ingest/upload
  POST /dashboard/review/{id}/approve
  POST /dashboard/review/{id}/reject
  POST /dashboard/discovery/{id}/promote
  POST /dashboard/discovery/{id}/dismiss
  POST /dashboard/system/ai-mode
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, Request, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models.document import Document, DocumentClass, PipelineEvent
from app.models.extracted_field import ExtractedField
from app.pipeline.states import PipelineState
from app.config import settings

# ── Templates ──────────────────────────────────────────────────────────────────
_TEMPLATES_DIR = pathlib.Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# ── PDF temporary store (local viewer only) ────────────────────────────────────
# PDFs ingested via the dashboard are saved here so the viewer can render them.
# In production, files live in SharePoint — this store is for localhost dev only.
# Stored inside the project so PDFs survive server restarts (unlike /tmp).
_PDF_STORE = pathlib.Path(__file__).parent.parent / "data" / "pdfs"
_PDF_STORE.mkdir(parents=True, exist_ok=True)


def _pdf_path_for(doc_id: str) -> pathlib.Path:
    return _PDF_STORE / f"{doc_id}.pdf"

# ── In-memory AI mode config ───────────────────────────────────────────────────
# Resets on server restart — fine for a local dev/ops tool.
# FAST     = keyword-only, no AI call
# VALIDATE = keyword primary + shadow AI review (logs disagreements)
# AUDIT    = AI primary, keyword as sanity check
AI_MODE: dict[str, str] = {
    "classification": "FAST",
    "extraction":     "FAST",
    "governance":     "ALWAYS",   # ALWAYS | SELECTIVE | OFF
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    return {
        "COMPLETED":           "#00C97A",
        "NEEDS_REVIEW":        "#F5A623",
        "CANDIDATE_NEW_CLASS": "#7C84E8",
        "FAILED":              "#FF4D4D",
        "GOVERNING":           "#7C84E8",
    }.get(status, "#6B9E82")


def _status_bg(status: str) -> str:
    return {
        "COMPLETED":           "#0D2B1A",
        "NEEDS_REVIEW":        "#2B1D0A",
        "CANDIDATE_NEW_CLASS": "#161B3A",
        "FAILED":              "#2B0D0D",
        "GOVERNING":           "#161B3A",
    }.get(status, "#111816")


def _stats(db: Session) -> dict:
    total      = db.query(Document).count()
    completed  = db.query(Document).filter(Document.status == "COMPLETED").count()
    review     = db.query(Document).filter(Document.status == "NEEDS_REVIEW").count()
    discovery  = db.query(Document).filter(Document.status == "CANDIDATE_NEW_CLASS").count()
    failed     = db.query(Document).filter(Document.status == "FAILED").count()
    processing = total - completed - review - discovery - failed
    return dict(
        total=total,
        completed=completed,
        review=review,
        discovery=discovery,
        failed=failed,
        processing=processing,
    )


def _recent_docs(db: Session, limit: int = 12) -> list[dict]:
    docs = db.query(Document).order_by(Document.created_at.desc()).limit(limit).all()
    result = []
    for d in docs:
        result.append(dict(
            id=d.id,
            file_name=d.file_name,
            status=d.status,
            status_color=_status_color(d.status),
            status_bg=_status_bg(d.status),
            document_class=d.document_class_id or "—",
            class_name=d.document_class.name if d.document_class else "—",
            confidence=f"{d.classification_confidence:.0%}" if d.classification_confidence else "—",
            created_at=d.created_at.strftime("%d %b %H:%M") if d.created_at else "—",
        ))
    return result


def _queue_docs(db: Session, status_filter: Optional[str], page: int, page_size: int) -> tuple[list[dict], int]:
    q = db.query(Document)
    if status_filter and status_filter != "ALL":
        q = q.filter(Document.status == status_filter)
    total = q.count()
    docs = q.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    rows = []
    for d in docs:
        n_fields = db.query(ExtractedField).filter(ExtractedField.document_id == d.id).count()
        rows.append(dict(
            id=d.id,
            file_name=d.file_name,
            status=d.status,
            status_color=_status_color(d.status),
            status_bg=_status_bg(d.status),
            document_class=d.document_class_id or "—",
            class_name=d.document_class.name if d.document_class else "—",
            confidence=f"{d.classification_confidence:.0%}" if d.classification_confidence else "—",
            conf_pct=int((d.classification_confidence or 0) * 100),
            n_fields=n_fields,
            created_at=d.created_at.strftime("%d %b %H:%M") if d.created_at else "—",
        ))
    return rows, total


def _review_docs(db: Session) -> list[dict]:
    docs = (
        db.query(Document)
        .filter(Document.status == "NEEDS_REVIEW")
        .order_by(Document.updated_at.asc())
        .all()
    )
    rows = []
    for d in docs:
        # Extract NIGO conditions from latest review pipeline event
        review_events = [e for e in d.pipeline_events if e.state == "NEEDS_REVIEW" and e.detail]
        nigo = []
        if review_events:
            latest = review_events[-1].detail
            if "condition(s) failed." in latest:
                _, _, rest = latest.partition("condition(s) failed.")
                nigo = [c.strip() for c in rest.split("|") if c.strip()]
            elif latest:
                nigo = [latest[:120]]
        n_fields = db.query(ExtractedField).filter(ExtractedField.document_id == d.id).count()
        gov = {}
        if d.ai_governance_result:
            try:
                gov = json.loads(d.ai_governance_result)
            except Exception:
                pass
        rows.append(dict(
            id=d.id,
            file_name=d.file_name,
            document_class=d.document_class_id or "—",
            class_name=d.document_class.name if d.document_class else "—",
            confidence=f"{d.classification_confidence:.0%}" if d.classification_confidence else "—",
            nigo=nigo,
            n_fields=n_fields,
            gov_verdict=gov.get("verdict", ""),
            gov_reasoning=gov.get("reasoning", "")[:100],
            updated_at=d.updated_at.strftime("%d %b %H:%M") if d.updated_at else "—",
        ))
    return rows


def _discovery_docs(db: Session) -> list[dict]:
    docs = (
        db.query(Document)
        .filter(Document.status == "CANDIDATE_NEW_CLASS")
        .order_by(Document.updated_at.asc())
        .all()
    )
    rows = []
    for d in docs:
        gov = {}
        if d.ai_governance_result:
            try:
                gov = json.loads(d.ai_governance_result)
            except Exception:
                pass
        rows.append(dict(
            id=d.id,
            file_name=d.file_name,
            suggested_name=d.suggested_class_name or "Unknown type",
            candidate_reason=(d.candidate_reason or "")[:120],
            gov_verdict=gov.get("verdict", ""),
            gov_reasoning=gov.get("reasoning", "")[:140],
            gov_keywords=gov.get("suggested_keywords", [])[:6],
            confidence=f"{gov.get('confidence', 0):.0%}" if gov.get("confidence") else "—",
            created_at=d.created_at.strftime("%d %b %H:%M") if d.created_at else "—",
        ))
    return rows


_TIER_LABEL = {1: "Text layer", 2: "OCR", 3: "AI Vision"}
_TIER_COLOR = {1: "#00C97A", 2: "#F5A623", 3: "#7C84E8"}
_MATCH_COLOR = {"PASS": "#00C97A", "WITHIN_TOLERANCE": "#F5A623", "FAIL": "#FF4D4D"}


def _doc_detail(db: Session, doc_id: str) -> Optional[dict]:
    d = db.query(Document).filter(Document.id == doc_id).first()
    if not d:
        return None
    fields = (
        db.query(ExtractedField)
        .filter(ExtractedField.document_id == doc_id)
        .order_by(ExtractedField.field_name)
        .all()
    )
    events = d.pipeline_events
    gov = {}
    if d.ai_governance_result:
        try:
            gov = json.loads(d.ai_governance_result)
        except Exception:
            pass

    field_dicts = []
    table_dicts = []   # {name, rows, confidence, method, tier_label, tier_color}
    for f in fields:
        field_type = getattr(f, "field_type", "scalar") or "scalar"
        if field_type == "table":
            # Parse the JSON array stored in field_value
            rows = []
            try:
                rows = json.loads(f.field_value or "[]")
            except Exception:
                rows = []
            table_dicts.append(dict(
                name=f.field_name,
                rows=rows,
                row_count=len(rows),
                columns=list(rows[0].keys()) if rows else [],
                confidence=f"{f.confidence:.0%}" if f.confidence else "—",
                conf_pct=int((f.confidence or 0) * 100),
                tier=f.extraction_tier,
                tier_label=_TIER_LABEL.get(f.extraction_tier, f"Tier {f.extraction_tier}"),
                tier_color=_TIER_COLOR.get(f.extraction_tier, "#6B9E82"),
                method=f.extraction_method or "",
            ))
        else:
            display_value = f.corrected_value or f.field_value or "—"
            # Spatial provenance — stored at extraction time; enables instant page-jump
            _bbox = None
            if getattr(f, "extraction_bbox_json", None):
                try:
                    _bbox = json.loads(f.extraction_bbox_json)
                except Exception:
                    pass
            # Parse address_json if present
            _address = None
            if getattr(f, "address_json", None):
                try:
                    _address = json.loads(f.address_json)
                except Exception:
                    pass
            field_dicts.append(dict(
                name=f.field_name,
                value=display_value,
                raw_value=f.field_value or "—",
                corrected=f.human_corrected,
                corrected_value=f.corrected_value,
                confidence=f"{f.confidence:.0%}" if f.confidence else "—",
                conf_pct=int((f.confidence or 0) * 100),
                tier=f.extraction_tier,
                tier_label=_TIER_LABEL.get(f.extraction_tier, f"Tier {f.extraction_tier}"),
                tier_color=_TIER_COLOR.get(f.extraction_tier, "#6B9E82"),
                method=f.extraction_method or "",
                model=f.extraction_model or "",
                match_result=f.match_result or "",
                match_color=_MATCH_COLOR.get(f.match_result or "", "#6B9E82"),
                used_in_match=f.used_in_match,
                # Spatial provenance
                extraction_page=getattr(f, "extraction_page", None),
                extraction_bbox=_bbox,
                # AddressAgent structured parse
                address=_address,
            ))

    # ── Variant info ──────────────────────────────────────────────────────────
    from app.models.document import DocumentVariant as DV
    from app.models.client import DocumentTypeProfile

    variant = db.query(DV).filter(DV.id == d.variant_id).first() if d.variant_id else None

    # Learning stage: variant-level takes priority over class-level
    dtp = db.query(DocumentTypeProfile).filter_by(
        client_id="cp_001", document_class_id=d.document_class_id
    ).first() if d.document_class_id else None

    learning_stage = (
        variant.learning_stage if variant
        else (dtp.learning_stage if dtp else "ZERO_SHOT")
    )
    stage_labels = {
        "ZERO_SHOT": "Zero-shot", "LEARNING": "Learning",
        "LEARNED_PROPOSED": "Schema proposed", "LEARNED": "Learned",
        "OPTIMISED": "Optimised",
    }
    stage_colors = {
        "ZERO_SHOT": "#6B9E82", "LEARNING": "#F5A623",
        "LEARNED_PROPOSED": "#7C84E8", "LEARNED": "#00C97A",
        "OPTIMISED": "#00C97A",
    }

    # Variant display info
    variant_label  = (variant.variant_label  if variant else None) or "—"
    variant_issuer = (variant.issuer_slug    if variant else None) or "unknown"
    variant_key_display = (variant.variant_key if variant else d.variant_key) or "—"
    variant_doc_count   = variant.doc_count if variant else 0

    # Primary extraction method (from first field persisted)
    first_field = fields[0] if fields else None
    extraction_method = first_field.extraction_method if first_field else "—"
    extraction_tier   = first_field.extraction_tier   if first_field else 0
    tier_label = _TIER_LABEL.get(extraction_tier, "—")

    # Confidence colour
    conf_pct = int((d.classification_confidence or 0) * 100)
    conf_color = "#00C97A" if conf_pct >= 80 else ("#F5A623" if conf_pct >= 60 else "#E55C5C")
    conf_bg    = ("rgba(0,201,122,0.12)" if conf_pct >= 80
                  else ("rgba(245,166,35,0.12)" if conf_pct >= 60
                        else "rgba(229,92,92,0.12)"))

    # Discovery mode: show the checkbox UI whenever stage is ZERO_SHOT and we have
    # fields — regardless of the extraction method that produced them.
    is_discovery = (
        learning_stage == "ZERO_SHOT"
        and bool(field_dicts)
    )

    has_tables = bool(table_dicts)

    # ── Page profile stats ────────────────────────────────────────────────────
    pages_total    = getattr(d, "pages_total",         None)
    pages_skipped  = getattr(d, "pages_skipped_count", None)
    pages_sampled_json = getattr(d, "pages_sampled_json", None)
    pages_sampled  = len(json.loads(pages_sampled_json)) if pages_sampled_json else None

    page_profile_stage = "none"
    page_profile_skip_count = 0
    page_profile_cost_saved = 0.0
    page_profile_instances  = 0
    if variant and getattr(variant, "page_profile_json", None):
        try:
            pp = json.loads(variant.page_profile_json)
            page_profile_instances  = pp.get("instances_seen", 0)
            page_profile_skip_count = len(pp.get("confident_skip", []))
            page_profile_cost_saved = pp.get("cost_saved_usd", 0.0)
            from app.agents.page_profile import PageProfileAgent as _PPA, MIN_INSTANCES as _MIN
            page_profile_stage = (
                "confident" if page_profile_skip_count > 0
                else ("learning" if page_profile_instances >= 1 else "none")
            )
        except Exception:
            pass

    return dict(
        id=d.id,
        file_name=d.file_name,
        file_size_kb=round(d.file_size_bytes / 1024, 1),
        status=d.status,
        status_color=_status_color(d.status),
        status_bg=_status_bg(d.status),
        document_class=d.document_class_id or "—",
        class_name=d.document_class.name if d.document_class else "Unknown",
        confidence=f"{d.classification_confidence:.0%}" if d.classification_confidence else "—",
        conf_pct=conf_pct,
        conf_color=conf_color,
        conf_bg=conf_bg,
        suggested_class=d.suggested_class_name or "",
        candidate_reason=d.candidate_reason or "",
        has_pdf=_pdf_path_for(doc_id).exists(),
        fields=field_dicts,
        tables=table_dicts,
        has_tables=has_tables,
        events=[{
            "state": e.state,
            "agent": e.agent or "—",
            "detail": (e.detail or "")[:120],
            "ts": e.created_at.strftime("%H:%M:%S") if e.created_at else "",
        } for e in events],
        gov=gov,
        created_at=d.created_at.strftime("%d %b %Y %H:%M") if d.created_at else "",
        sha256=d.file_sha256[:16] + "…" if d.file_sha256 else "—",
        # Learning / variant
        learning_stage=learning_stage,
        stage_label=stage_labels.get(learning_stage, learning_stage),
        stage_color=stage_colors.get(learning_stage, "#6B9E82"),
        variant_label=variant_label,
        variant_issuer=variant_issuer,
        variant_key=variant_key_display,
        variant_doc_count=variant_doc_count,
        has_variant=variant is not None,
        extraction_method=extraction_method,
        extraction_tier=extraction_tier,
        tier_label=tier_label,
        is_discovery_mode=is_discovery,
        # Page sampling
        pages_total=pages_total,
        pages_sampled=pages_sampled,
        pages_skipped=pages_skipped,
        # PageProfileAgent
        page_profile_stage=page_profile_stage,
        page_profile_skip_count=page_profile_skip_count,
        page_profile_cost_saved=page_profile_cost_saved,
        page_profile_instances=page_profile_instances,
        # Signature detection
        is_signed=d.is_signed,
        signature_confidence=round(d.signature_confidence or 0.0, 2),
        signature_evidence=json.loads(d.signature_evidence_json) if d.signature_evidence_json else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Full pages
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def dashboard_overview(request: Request, db: Session = Depends(get_db)):
    stats = _stats(db)
    recent = _recent_docs(db)
    classes = db.query(DocumentClass).filter(DocumentClass.active == True).order_by(DocumentClass.id).all()
    return templates.TemplateResponse("dashboard/overview.html", {
        "request": request,
        "page": "overview",
        "stats": stats,
        "recent": recent,
        "classes": classes,
        "ai_mode": AI_MODE,
        "api_key_set": bool(settings.anthropic_api_key),
        "sp_connected": bool(settings.sp_site_url and settings.sp_access_token),
        "bc_connected": bool(settings.bc_api_url and settings.bc_api_key),
    })


@router.get("/queue", response_class=HTMLResponse)
def dashboard_queue(
    request: Request,
    status: str = "ALL",
    page: int = 1,
    db: Session = Depends(get_db),
):
    page_size = 25
    rows, total = _queue_docs(db, status, page, page_size)
    pages = max(1, (total + page_size - 1) // page_size)
    return templates.TemplateResponse("dashboard/queue.html", {
        "request": request,
        "page": "queue",
        "rows": rows,
        "total": total,
        "current_page": page,
        "pages": pages,
        "page_size": page_size,
        "status_filter": status,
        "statuses": ["ALL", "COMPLETED", "NEEDS_REVIEW", "CANDIDATE_NEW_CLASS",
                     "FAILED", "CLASSIFYING", "EXTRACTING", "GOVERNING"],
    })


@router.get("/review", response_class=HTMLResponse)
def dashboard_review(request: Request, db: Session = Depends(get_db)):
    rows = _review_docs(db)
    classes = db.query(DocumentClass).filter(DocumentClass.active == True).order_by(DocumentClass.id).all()
    return templates.TemplateResponse("dashboard/review.html", {
        "request": request,
        "page": "review",
        "rows": rows,
        "classes": classes,
        "total": len(rows),
    })


@router.get("/discovery", response_class=HTMLResponse)
def dashboard_discovery(request: Request, db: Session = Depends(get_db)):
    rows = _discovery_docs(db)
    classes = db.query(DocumentClass).filter(DocumentClass.active == True).order_by(DocumentClass.id).all()
    return templates.TemplateResponse("dashboard/discovery.html", {
        "request": request,
        "page": "discovery",
        "rows": rows,
        "classes": classes,
        "total": len(rows),
    })


# ── Documents section ──────────────────────────────────────────────────────────

def _review_count(db: Session) -> int:
    return db.query(Document).filter(Document.status == "NEEDS_REVIEW").count()


def _documents_context(db: Session) -> list[dict]:
    """
    Build the left-panel list for the Documents page.
    Returns one dict per DocumentClass, each with a list of variant dicts.
    Parent class has no schema — schema lives at variant level only.
    """
    from app.models.document import DocumentVariant
    from app.models.client import DocumentTypeProfile

    classes = (
        db.query(DocumentClass)
        .filter(DocumentClass.active == True)
        .order_by(DocumentClass.name)
        .all()
    )

    result = []
    for dc in classes:
        variants = (
            db.query(DocumentVariant)
            .filter(DocumentVariant.document_class_id == dc.id)
            .order_by(DocumentVariant.doc_count.desc())
            .all()
        )
        total_docs = sum(v.doc_count or 0 for v in variants)
        # Overall stage: best variant stage wins for display
        stage_order = {"OPTIMISED": 5, "LEARNED": 4, "LEARNED_PROPOSED": 3,
                       "LEARNING": 2, "ZERO_SHOT": 1}
        best_stage = max(
            (v.learning_stage for v in variants),
            key=lambda s: stage_order.get(s, 0),
            default="ZERO_SHOT",
        ) if variants else "ZERO_SHOT"

        variant_dicts = []
        for v in variants:
            schema = json.loads(v.field_schema_json) if v.field_schema_json else {}
            variant_dicts.append(dict(
                id=v.id,
                label=v.variant_label or v.issuer_slug or v.variant_key or f"Variant {v.id[:6]}",
                issuer_slug=v.issuer_slug or "—",
                learning_stage=v.learning_stage,
                doc_count=v.doc_count or 0,
                field_count=len(schema),
                variant_key=v.variant_key or "",
            ))

        result.append(dict(
            id=dc.id,
            name=dc.name,
            best_stage=best_stage,
            total_docs=total_docs,
            variant_count=len(variants),
            variants=variant_dicts,
        ))
    return result


@router.get("/documents", response_class=HTMLResponse)
def dashboard_documents(request: Request, db: Session = Depends(get_db)):
    doc_types = _documents_context(db)
    return templates.TemplateResponse("dashboard/documents.html", {
        "request":      request,
        "page":         "documents",
        "doc_types":    doc_types,
        "review_count": _review_count(db),
    })


@router.get("/documents/class/{dc_id}/panel", response_class=HTMLResponse)
def documents_class_panel(dc_id: str, request: Request, db: Session = Depends(get_db)):
    """HTMX — right panel for a selected document class (category view, no schema)."""
    from app.models.document import DocumentVariant

    dc = db.query(DocumentClass).filter(DocumentClass.id == dc_id).first()
    if not dc:
        return HTMLResponse("<div style='padding:24px;color:var(--danger)'>Class not found</div>")

    variants = (
        db.query(DocumentVariant)
        .filter(DocumentVariant.document_class_id == dc_id)
        .order_by(DocumentVariant.doc_count.desc())
        .all()
    )
    total_docs = sum(v.doc_count or 0 for v in variants)
    recent = (
        db.query(Document)
        .filter(Document.document_class_id == dc_id)
        .order_by(Document.created_at.desc())
        .limit(8)
        .all()
    )
    recent_dicts = [dict(
        id=d.id,
        file_name=d.file_name,
        status=d.status,
        status_color=_status_color(d.status),
        created_at=d.created_at.strftime("%d %b %H:%M") if d.created_at else "—",
        is_signed=d.is_signed,
        variant_label=(
            db.query(DocumentVariant).filter(DocumentVariant.id == d.variant_id).first().variant_label
            if d.variant_id else None
        ),
    ) for d in recent]

    stage_order = {"OPTIMISED": 5, "LEARNED": 4, "LEARNED_PROPOSED": 3,
                   "LEARNING": 2, "ZERO_SHOT": 1}
    best_stage = max(
        (v.learning_stage for v in variants),
        key=lambda s: stage_order.get(s, 0),
        default="ZERO_SHOT",
    ) if variants else "ZERO_SHOT"

    variant_dicts = []
    total_confirmed_fields = 0
    for v in variants:
        schema = json.loads(v.field_schema_json) if v.field_schema_json else {}
        field_count = len(schema)
        total_confirmed_fields += field_count
        # Last doc uploaded for this variant
        last_doc = (
            db.query(Document)
            .filter(Document.variant_id == v.id)
            .order_by(Document.created_at.desc())
            .first()
        )
        variant_dicts.append(dict(
            id=v.id,
            label=v.variant_label or v.issuer_slug or v.variant_key or f"Variant {v.id[:6]}",
            learning_stage=v.learning_stage,
            doc_count=v.doc_count or 0,
            field_count=field_count,
            last_activity=(
                last_doc.created_at.strftime("%d %b %H:%M")
                if last_doc and last_doc.created_at else "—"
            ),
        ))

    # ── Classifier profile ──────────────────────────────────────────────────────
    classifier = {"keywords": [], "negative_keywords": [], "priority": 3,
                  "notes": "", "ai_observations": ""}
    if dc.classifier_profile_json:
        try:
            classifier = json.loads(dc.classifier_profile_json)
        except Exception:
            pass

    return templates.TemplateResponse("dashboard/partials/doc_class_panel.html", {
        "request":                request,
        "dc":                     dc,
        "variants":               variant_dicts,
        "total_docs":             total_docs,
        "total_confirmed_fields": total_confirmed_fields,
        "best_stage":             best_stage,
        "recent":                 recent_dicts,
        "classifier":             classifier,
    })


@router.post("/documents/class/{dc_id}/classifier", response_class=HTMLResponse)
async def save_class_classifier(
    dc_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Save operator edits to the classifier profile (keywords, negative_keywords, priority, notes)."""
    dc = db.query(DocumentClass).filter(DocumentClass.id == dc_id).first()
    if not dc:
        return HTMLResponse(_toast("error", "Document class not found"), status_code=404)

    form = await request.form()

    # Keywords — comma-separated from hidden input
    raw_kw = (form.get("keywords") or "").strip()
    keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]

    raw_neg = (form.get("negative_keywords") or "").strip()
    negative_keywords = [k.strip().lower() for k in raw_neg.split(",") if k.strip()]

    try:
        priority = int(form.get("priority") or 3)
    except ValueError:
        priority = 3

    notes = (form.get("notes") or "").strip()

    # Preserve existing ai_observations — operator can't edit those directly
    existing = {}
    if dc.classifier_profile_json:
        try:
            existing = json.loads(dc.classifier_profile_json)
        except Exception:
            pass

    profile = {
        "keywords":          keywords,
        "negative_keywords": negative_keywords,
        "priority":          priority,
        "notes":             notes,
        "ai_observations":   existing.get("ai_observations", ""),
    }
    dc.classifier_profile_json = json.dumps(profile)
    db.commit()

    return HTMLResponse(_toast("success", f"Classifier profile saved — {len(keywords)} keywords, priority {priority}"))


@router.get("/documents/variant/{variant_id}/panel", response_class=HTMLResponse)
def documents_variant_panel(variant_id: str, request: Request, db: Session = Depends(get_db)):
    """HTMX — right panel for a selected variant: fields (with aliases), coverage, recent."""
    from app.models.document import DocumentVariant
    from sqlalchemy import func

    v = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
    if not v:
        return HTMLResponse("<div style='padding:24px;color:var(--danger)'>Variant not found</div>")

    dc = db.query(DocumentClass).filter(DocumentClass.id == v.document_class_id).first()
    schema = json.loads(v.field_schema_json) if v.field_schema_json else {}

    # ── Build alias map from all extracted fields across this variant's docs ──
    # For each canonical name in schema, collect all distinct raw field_name values
    # that have been extracted from this variant's documents.
    raw_field_counts = (
        db.query(ExtractedField.field_name, func.count(ExtractedField.id).label("cnt"))
        .join(Document, Document.id == ExtractedField.document_id)
        .filter(Document.variant_id == variant_id)
        .group_by(ExtractedField.field_name)
        .all()
    )
    raw_names_seen = {row.field_name for row in raw_field_counts}

    # ── If no confirmed schema yet, synthesize one from extracted fields ───────
    # This makes the variant panel useful immediately after the first document is
    # processed — the user can see what the AI found and save it in one click,
    # rather than facing an empty "no schema confirmed" state.
    is_proposed = False
    if not schema and raw_names_seen:
        is_proposed = True
        # Grab example values from the variant's most-recent document
        example_doc = (
            db.query(Document)
            .filter(Document.variant_id == variant_id)
            .order_by(Document.created_at.desc())
            .first()
        )
        example_values: dict[str, str] = {}
        if example_doc:
            for ef in (
                db.query(ExtractedField)
                .filter(ExtractedField.document_id == example_doc.id)
                .all()
            ):
                if ef.field_name and ef.field_name not in example_values:
                    example_values[ef.field_name] = ef.field_value or ""
        # Build proposed schema: field_name → canonical entry
        # Sort by frequency descending so most-consistent fields appear first
        for row in sorted(raw_field_counts, key=lambda r: (-r.cnt, r.field_name)):
            fname = row.field_name
            if fname:
                schema[fname] = {
                    "aliases":     [fname],
                    "example":     example_values.get(fname, ""),
                    "format_hint": "",
                    "format_type": "",
                }

    # Build field list with canonical name, aliases, example value, confidence
    fields = []
    for canonical, meta in schema.items():
        if isinstance(meta, dict):
            stored_aliases = meta.get("aliases", [canonical])
            example        = meta.get("example", "")
        else:
            stored_aliases = [canonical]
            example        = str(meta) if meta else ""

        # Merge stored aliases with any raw names seen that fuzzy-match this canonical
        all_aliases = list(dict.fromkeys(stored_aliases))  # preserve order, dedup
        for raw in raw_names_seen:
            if raw not in all_aliases and (
                raw == canonical or raw.replace("_", "") == canonical.replace("_", "")
            ):
                all_aliases.append(raw)

        # Average confidence for this field across the variant's docs
        conf_row = (
            db.query(func.avg(ExtractedField.confidence))
            .join(Document, Document.id == ExtractedField.document_id)
            .filter(Document.variant_id == variant_id, ExtractedField.field_name.in_(all_aliases))
            .scalar()
        )
        format_hint = meta.get("format_hint", "") if isinstance(meta, dict) else ""
        format_type = meta.get("format_type", "") if isinstance(meta, dict) else ""
        fields.append(dict(
            canonical=canonical,
            aliases=all_aliases,
            example=example,
            confidence=int((conf_row or 0) * 100),
            format_hint=format_hint,
            format_type=format_type,
        ))

    # ── Recent documents ───────────────────────────────────────────────────────
    recent_docs = (
        db.query(Document)
        .filter(Document.variant_id == variant_id)
        .order_by(Document.created_at.desc())
        .limit(10)
        .all()
    )

    # ── Coverage heatmap ───────────────────────────────────────────────────────
    canonical_names = list(schema.keys())
    coverage_rows = []
    for d in recent_docs[:8]:
        doc_fields = {
            ef.field_name: ef.confidence
            for ef in db.query(ExtractedField)
            .filter(ExtractedField.document_id == d.id)
            .all()
        }
        row_cells = []
        for canonical, meta in schema.items():
            aliases = meta.get("aliases", [canonical]) if isinstance(meta, dict) else [canonical]
            # Find best match in extracted fields
            best_conf = None
            for alias in aliases:
                if alias in doc_fields:
                    c = doc_fields[alias]
                    if best_conf is None or c > best_conf:
                        best_conf = c
            if best_conf is None:
                row_cells.append("missing")
            elif best_conf >= 0.70:
                row_cells.append("found")
            else:
                row_cells.append("low")
        coverage_rows.append(dict(
            file_name=d.file_name,
            doc_id=d.id,
            cells=row_cells,
        ))

    recent_dicts = [dict(
        id=d.id,
        file_name=d.file_name,
        status=d.status,
        status_color=_status_color(d.status),
        n_fields=db.query(ExtractedField).filter(ExtractedField.document_id == d.id).count(),
        created_at=d.created_at.strftime("%d %b %H:%M") if d.created_at else "—",
        is_signed=d.is_signed,
    ) for d in recent_docs]

    # Format agent metadata
    format_agent_meta = schema.get("__format_agent__", {}) if schema else {}
    format_ran_at = format_agent_meta.get("ran_at", "")
    format_ran_at_display = ""
    if format_ran_at:
        try:
            from datetime import datetime as _dt
            _d = _dt.fromisoformat(format_ran_at)
            format_ran_at_display = _d.strftime("%-d %b %H:%M")
        except Exception:
            format_ran_at_display = format_ran_at[:16]

    return templates.TemplateResponse("dashboard/partials/doc_variant_panel.html", {
        "request":              request,
        "variant":              v,
        "dc":                   dc,
        "fields":               fields,
        "canonical_names":      canonical_names,
        "coverage_rows":        coverage_rows,
        "recent":               recent_dicts,
        "format_ran_at":        format_ran_at_display,
        "is_proposed":          is_proposed,
    })


@router.post("/documents/variant/{variant_id}/schema", response_class=HTMLResponse)
async def documents_save_variant_schema(
    variant_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Save canonical field names + alias mappings for a variant.
    Form fields (per schema entry keyed by old_name):
      canonical_{old_name}  = new_canonical_name
      checked_{old_name}    = "on"  (checkbox — include this field)
      aliases_{old_name}    = comma-joined alias list (e.g. "order_no,po_no,po_number")
    """
    import datetime as _dt
    from app.models.document import DocumentVariant

    v = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
    if not v:
        return _toast_raw("danger", "Variant not found")

    form = await request.form()
    existing_schema = json.loads(v.field_schema_json) if v.field_schema_json else {}

    new_schema: dict = {}
    for old_name, meta in existing_schema.items():
        checked_key   = f"checked_{old_name}"
        canonical_key = f"canonical_{old_name}"

        if form.get(checked_key) != "on":
            continue  # field was unchecked — remove from schema

        new_canonical = (form.get(canonical_key) or old_name).strip().lower().replace(" ", "_")
        if not new_canonical:
            new_canonical = old_name

        # Aliases: use whatever the user submitted (comma-joined hidden input).
        # Always guarantee old_name and new_canonical are in the list so the
        # normaliser can still remap docs extracted before the rename.
        aliases_raw = (form.get(f"aliases_{old_name}") or "").strip()
        submitted_aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
        if not submitted_aliases:
            # Fallback: inherit existing aliases
            submitted_aliases = meta.get("aliases", [old_name]) if isinstance(meta, dict) else [old_name]
        # Guarantee both old and new canonical are present as aliases
        all_aliases = list(dict.fromkeys(submitted_aliases + [old_name, new_canonical]))

        new_schema[new_canonical] = {
            "aliases":  all_aliases,
            "required": True,
            "example":  meta.get("example", "") if isinstance(meta, dict) else "",
        }

    v.field_schema_json = json.dumps(new_schema)
    v.updated_at        = _dt.datetime.utcnow()
    if v.learning_stage == "ZERO_SHOT":
        v.learning_stage = "LEARNING"
    db.commit()

    return _toast_raw("success",
        f"✓ {len(new_schema)} fields saved for "
        f"<strong>{v.variant_label or v.issuer_slug or variant_id[:8]}</strong>. "
        "Canonical names will be used in all API responses."
    )


@router.post("/documents/ingest/{dc_id}", response_class=HTMLResponse)
async def documents_ingest_for_class(
    dc_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
):
    """
    Upload a PDF with a document-class hint. The classifier still runs
    and may reassign the document if it looks different enough.
    """
    from app.pipeline.runner import run_pipeline

    dc = db.query(DocumentClass).filter(DocumentClass.id == dc_id).first()
    if not dc:
        return _toast_raw("danger", "Document class not found")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        return _toast_raw("warn", "Empty file — nothing uploaded")

    doc = Document(
        file_name=file.filename or "upload.pdf",
        status="CLASSIFYING",
        document_class_id=dc_id,          # hint — classifier may change this
        classification_confidence=None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Save PDF for viewer
    pdf_path = _PDF_STORE / f"{doc.id}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    background_tasks.add_task(run_pipeline, doc.id, pdf_bytes)

    return _toast_raw("success",
        f"✓ <strong>{file.filename}</strong> submitted as a "
        f"<em>{dc.name}</em> example. Pipeline running — check Manual Review if it needs attention."
    )


@router.post("/settings/nuclear-reset", response_class=HTMLResponse)
async def settings_nuclear_reset(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Wipe EVERYTHING — all document classes, variants, documents, extracted fields,
    pipeline events, DTPs, and shipment records.  Client profile is kept.
    Sets a DB flag so classes are NOT re-seeded on the next restart.
    """
    from app.models.extracted_field import ExtractedField
    from app.models.document import DocumentVariant, DocumentClass, PipelineEvent
    from app.models.shipment import ShipmentRecord
    from app.models.client import DocumentTypeProfile

    # Counts for the toast
    n_docs     = db.query(Document).count()
    n_classes  = db.query(DocumentClass).count()

    # Delete in FK-safe order (children before parents)
    db.query(ExtractedField).delete(synchronize_session=False)
    db.query(PipelineEvent).delete(synchronize_session=False)
    db.query(ShipmentRecord).delete(synchronize_session=False)
    db.query(Document).delete(synchronize_session=False)
    db.query(DocumentVariant).delete(synchronize_session=False)
    db.query(DocumentTypeProfile).delete(synchronize_session=False)
    db.query(DocumentClass).delete(synchronize_session=False)

    # Write a flag row so _seed_document_classes() skips re-seeding on restart
    import sqlalchemy as _sa
    try:
        with db.bind.connect() as conn:
            conn.execute(_sa.text(
                "CREATE TABLE IF NOT EXISTS system_flags "
                "(key TEXT PRIMARY KEY, value TEXT)"
            ))
            conn.execute(_sa.text(
                "INSERT OR REPLACE INTO system_flags (key, value) VALUES ('seed_classes', 'false')"
            ))
            conn.commit()
    except Exception:
        pass  # flag table creation is best-effort

    db.commit()

    return _toast_raw("success",
        f"✓ Full reset complete — deleted {n_docs} document{'s' if n_docs != 1 else ''} "
        f"and {n_classes} document type{'s' if n_classes != 1 else ''}. "
        "Restart the server and upload your first document to begin learning from scratch."
    )


@router.post("/documents/class/{dc_id}/reset", response_class=HTMLResponse)
async def documents_reset_class(
    dc_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Wipe all learned data for a document class so it can re-learn from scratch.
    Deletes: all documents (+ their extracted fields, pipeline events, shipment records)
             all variants, and the class-level DTP schema/learning data.
    The DocumentClass row itself is kept so the class still appears in the list.
    """
    from app.models.extracted_field import ExtractedField
    from app.models.document import DocumentVariant, PipelineEvent
    from app.models.shipment import ShipmentRecord
    from app.models.client import DocumentTypeProfile

    dc = db.query(DocumentClass).filter(DocumentClass.id == dc_id).first()
    if not dc:
        return _toast_raw("danger", "Document class not found")

    # Collect all document IDs for this class
    doc_ids = [r[0] for r in db.query(Document.id).filter(Document.document_class_id == dc_id).all()]

    if doc_ids:
        db.query(ExtractedField).filter(ExtractedField.document_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(PipelineEvent).filter(PipelineEvent.document_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(ShipmentRecord).filter(ShipmentRecord.document_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(Document).filter(Document.id.in_(doc_ids)).delete(synchronize_session=False)

    # Wipe variants
    db.query(DocumentVariant).filter(DocumentVariant.document_class_id == dc_id).delete(synchronize_session=False)

    # Reset DTP learning data (keep the row, just clear learned content)
    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.document_class_id == dc_id).first()
    if dtp:
        dtp.field_schema_json      = None
        dtp.generated_patterns_json = None
        dtp.learning_stage         = "ZERO_SHOT"
        dtp.doc_count              = 0

    db.commit()

    n_docs = len(doc_ids)
    return _toast_raw("success",
        f"✓ Reset <strong>{dc.name}</strong> — deleted {n_docs} document{'s' if n_docs != 1 else ''}, "
        "all variants, and cleared learned schema. "
        "Refresh the page and upload fresh examples to start learning."
    )


@router.post("/documents/variant/{variant_id}/reset", response_class=HTMLResponse)
async def documents_reset_variant(
    variant_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Wipe a single variant's learned schema + its documents so it re-discovers fields.
    """
    from app.models.extracted_field import ExtractedField
    from app.models.document import DocumentVariant, PipelineEvent
    from app.models.shipment import ShipmentRecord

    v = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
    if not v:
        return _toast_raw("danger", "Variant not found")

    doc_ids = [r[0] for r in db.query(Document.id).filter(Document.variant_id == variant_id).all()]

    if doc_ids:
        db.query(ExtractedField).filter(ExtractedField.document_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(PipelineEvent).filter(PipelineEvent.document_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(ShipmentRecord).filter(ShipmentRecord.document_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(Document).filter(Document.id.in_(doc_ids)).delete(synchronize_session=False)

    db.delete(v)
    db.commit()

    return _toast_raw("success",
        f"✓ Variant reset — deleted {len(doc_ids)} document{'s' if len(doc_ids) != 1 else ''} "
        "and cleared the variant schema. Upload fresh examples to re-learn."
    )


@router.get("/ingest", response_class=HTMLResponse)
def dashboard_ingest(request: Request, db: Session = Depends(get_db)):
    recent = _recent_docs(db, limit=5)
    return templates.TemplateResponse("dashboard/ingest.html", {
        "request": request,
        "page": "ingest",
        "recent": recent,
        "api_key": settings.dokr_api_key,
    })


@router.get("/system", response_class=HTMLResponse)
def dashboard_system(request: Request):
    return templates.TemplateResponse("dashboard/system.html", {
        "request": request,
        "page": "system",
        "ai_mode": AI_MODE,
        "cfg": settings,
        "api_key_set": bool(settings.anthropic_api_key),
        "sp_connected": bool(settings.sp_site_url and settings.sp_access_token),
        "bc_connected": bool(settings.bc_api_url and settings.bc_api_key),
        "api_key_preview": (settings.dokr_api_key[:8] + "••••" + settings.dokr_api_key[-4:])
                           if len(settings.dokr_api_key) > 12 else "••••",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  HTMX partials
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/partials/stats", response_class=HTMLResponse)
def partial_stats(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("dashboard/partials/stats.html", {
        "request": request,
        "stats": _stats(db),
    })


@router.get("/partials/queue-rows", response_class=HTMLResponse)
def partial_queue_rows(
    request: Request,
    status: str = "ALL",
    page: int = 1,
    db: Session = Depends(get_db),
):
    rows, total = _queue_docs(db, status, page, 25)
    return templates.TemplateResponse("dashboard/partials/queue_rows.html", {
        "request": request,
        "rows": rows,
        "total": total,
    })


@router.get("/partials/review-rows", response_class=HTMLResponse)
def partial_review_rows(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("dashboard/partials/review_rows.html", {
        "request": request,
        "rows": _review_docs(db),
    })


@router.get("/partials/discovery-rows", response_class=HTMLResponse)
def partial_discovery_rows(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("dashboard/partials/discovery_rows.html", {
        "request": request,
        "rows": _discovery_docs(db),
    })


@router.get("/partials/doc-status/{doc_id}", response_class=HTMLResponse)
def partial_doc_status(doc_id: str, request: Request, db: Session = Depends(get_db)):
    doc = _doc_detail(db, doc_id)
    if not doc:
        return HTMLResponse('<div style="color:#FF4D4D;font-family:Inter,sans-serif;font-size:13px;">Document not found</div>')
    is_terminal = doc["status"] in {"COMPLETED", "NEEDS_REVIEW", "CANDIDATE_NEW_CLASS", "FAILED", "EXACT_DUPLICATE"}
    return templates.TemplateResponse("dashboard/partials/doc_status.html", {
        "request": request,
        "doc": doc,
        "is_terminal": is_terminal,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Actions
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ingest/upload", response_class=HTMLResponse)
async def ingest_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Accept a PDF, create the Document record, kick off the pipeline, return a status fragment."""
    import io
    from app.utils.files import read_and_validate_pdf
    from app.utils.hashing import sha256_bytes
    from app.utils.ids import generate_document_id
    from app.pipeline.runner import run_pipeline

    error_msg = None
    doc_id = None

    try:
        pdf_bytes = await file.read()

        # Basic validation
        if not pdf_bytes[:4] == b"%PDF":
            error_msg = "File does not appear to be a valid PDF."
        elif len(pdf_bytes) > settings.max_file_size_mb * 1024 * 1024:
            error_msg = f"File exceeds {settings.max_file_size_mb} MB limit."
        else:
            sha = sha256_bytes(pdf_bytes)
            # Check for exact duplicate
            existing = db.query(Document).filter(Document.file_sha256 == sha).first()
            if existing:
                doc_id = existing.id
                return templates.TemplateResponse("dashboard/partials/doc_status.html", {
                    "request": request,
                    "doc": _doc_detail(db, existing.id),
                    "is_terminal": True,
                    "duplicate": True,
                })

            doc_id = generate_document_id()
            now = datetime.utcnow()
            doc = Document(
                id=doc_id,
                status=PipelineState.RECEIVED,
                file_name=file.filename or "upload.pdf",
                file_size_bytes=len(pdf_bytes),
                file_sha256=sha,
                created_at=now,
                updated_at=now,
            )
            db.add(doc)
            event = PipelineEvent(
                document_id=doc_id,
                state=PipelineState.RECEIVED,
                agent="DashboardIngest",
                detail=f"Uploaded via dashboard: {file.filename}",
            )
            db.add(event)
            db.commit()

            background_tasks.add_task(run_pipeline, doc_id, pdf_bytes)

            # Save PDF for the dashboard viewer (best-effort)
            try:
                _pdf_path_for(doc_id).write_bytes(pdf_bytes)
            except Exception:
                pass

    except Exception as exc:
        error_msg = f"Upload failed: {exc}"

    if error_msg:
        return HTMLResponse(f"""
        <div style="border:1px solid #FF4D4D;padding:16px;color:#FF4D4D;font-family:'Inter',sans-serif;font-size:13px;">
          {error_msg}
        </div>""")

    doc = _doc_detail(db, doc_id)
    return templates.TemplateResponse("dashboard/partials/doc_status.html", {
        "request": request,
        "doc": doc,
        "is_terminal": False,
        "duplicate": False,
    })


@router.post("/review/{doc_id}/approve", response_class=HTMLResponse)
def review_approve(
    doc_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    target_stage: str = Form(...),
    approved_by: str = Form(default="dashboard-user"),
    note: str = Form(default=""),
    db: Session = Depends(get_db),
):
    from app.pipeline.runner import run_pipeline_from

    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.status != "NEEDS_REVIEW":
        return _toast("danger", "Not in NEEDS_REVIEW state")

    valid = {"EXTRACTING", "VALIDATING", "MATCHING", "POSTING", "COMPLETED"}
    if target_stage not in valid:
        return _toast("danger", f"Invalid target stage: {target_stage}")

    prev = doc.status
    doc.status = target_stage
    doc.updated_at = datetime.utcnow()
    ev = PipelineEvent(
        document_id=doc_id,
        state=target_stage,
        agent="HumanReviewAgent",
        detail=f"APPROVED by {approved_by}. Re-queued → {target_stage}." + (f" {note}" if note else ""),
    )
    db.add(ev)
    db.commit()

    if target_stage != "COMPLETED":
        background_tasks.add_task(run_pipeline_from, doc_id, target_stage)

    return _toast("success", f"Approved → {target_stage}")


@router.post("/review/{doc_id}/reject", response_class=HTMLResponse)
def review_reject(
    doc_id: str,
    request: Request,
    reason: str = Form(...),
    rejected_by: str = Form(default="dashboard-user"),
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.status != "NEEDS_REVIEW":
        return _toast("danger", "Not in NEEDS_REVIEW state")

    doc.status = "FAILED"
    doc.updated_at = datetime.utcnow()
    ev = PipelineEvent(
        document_id=doc_id,
        state="FAILED",
        agent="HumanReviewAgent",
        detail=f"REJECTED by {rejected_by}. Reason: {reason}",
    )
    db.add(ev)
    db.commit()
    return _toast("success", "Rejected → FAILED")


@router.post("/discovery/{doc_id}/promote", response_class=HTMLResponse)
def discovery_promote(
    doc_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    class_name: str = Form(...),
    promoted_by: str = Form(default="dashboard-user"),
    db: Session = Depends(get_db),
):
    from app.routers.discovery import _next_dc_id

    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.status != "CANDIDATE_NEW_CLASS":
        return _toast("danger", "Not in CANDIDATE_NEW_CLASS state")

    new_id = _next_dc_id(db)
    slug = class_name.lower().replace(" ", "_").replace("/", "_")[:40]
    new_class = DocumentClass(id=new_id, name=class_name, slug=slug, treatment="STORE", active=True)
    db.add(new_class)

    doc.document_class_id = new_id
    doc.document_class_override = new_id
    doc.status = "EXTRACTING"
    doc.updated_at = datetime.utcnow()
    ev = PipelineEvent(
        document_id=doc_id,
        state="EXTRACTING",
        agent="HumanDiscoveryAgent",
        detail=f"PROMOTED by {promoted_by}. New class: {new_id} ('{class_name}'). Re-queued → EXTRACTING.",
    )
    db.add(ev)
    db.commit()

    from app.routers.discovery import _retry_extraction
    background_tasks.add_task(_retry_extraction, doc_id)
    return _toast("success", f"Promoted → {new_id}: {class_name}")


@router.post("/discovery/{doc_id}/dismiss", response_class=HTMLResponse)
def discovery_dismiss(
    doc_id: str,
    request: Request,
    reason: str = Form(...),
    dismissed_by: str = Form(default="dashboard-user"),
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.status != "CANDIDATE_NEW_CLASS":
        return _toast("danger", "Not in CANDIDATE_NEW_CLASS state")

    doc.status = "NEEDS_REVIEW"
    doc.updated_at = datetime.utcnow()
    ev = PipelineEvent(
        document_id=doc_id,
        state="NEEDS_REVIEW",
        agent="HumanDiscoveryAgent",
        detail=f"DISMISSED by {dismissed_by}. Reason: {reason}. Routed → NEEDS_REVIEW.",
    )
    db.add(ev)
    db.commit()
    return _toast("success", "Dismissed → NEEDS_REVIEW")


@router.post("/system/ai-mode", response_class=HTMLResponse)
def system_ai_mode(
    request: Request,
    classification: str = Form(default="FAST"),
    extraction: str = Form(default="FAST"),
    governance: str = Form(default="ALWAYS"),
):
    AI_MODE["classification"] = classification
    AI_MODE["extraction"] = extraction
    AI_MODE["governance"] = governance
    return _toast("success", "AI pipeline mode updated")


# ─────────────────────────────────────────────────────────────────────────────
#  Document detail — PDF viewer + extracted fields
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/docs/{doc_id}", response_class=HTMLResponse)
def doc_detail(
    doc_id: str,
    request: Request,
    pdf_page_num: int = Query(default=0, alias="page"),
    db: Session = Depends(get_db),
):
    """Full document detail: PDF viewer with canvas highlight overlay + fields panel."""
    doc = _doc_detail(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    page_count = 0
    if doc["has_pdf"]:
        try:
            import fitz
            fitz_doc = fitz.open(str(_pdf_path_for(doc_id)))
            page_count = len(fitz_doc)
            fitz_doc.close()
        except Exception:
            pass

    classes = db.query(DocumentClass).filter(DocumentClass.active == True).order_by(DocumentClass.id).all()

    return templates.TemplateResponse("dashboard/doc_detail.html", {
        "request": request,
        "page": "queue",          # nav highlight
        "doc": doc,
        "pdf_page_num": min(pdf_page_num, max(0, page_count - 1)),
        "page_count": page_count,
        "classes": classes,
        "is_review": doc["status"] == "NEEDS_REVIEW",
        "is_discovery": doc["status"] == "CANDIDATE_NEW_CLASS",
        "tables":     doc.get("tables", []),
        "has_tables": doc.get("has_tables", False),
    })


# ── Inline reclassify (HTMX — returns updated classification banner) ──────────

@router.post("/docs/{doc_id}/reclassify", response_class=HTMLResponse)
def doc_reclassify(
    doc_id: str,
    request: Request,
    class_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Override classification for a document and return the refreshed banner fragment."""
    d = db.query(Document).filter(Document.id == doc_id).first()
    if not d:
        return _toast_raw("danger", "Document not found")
    dc = db.query(DocumentClass).filter(DocumentClass.id == class_id).first()
    if not dc:
        return _toast_raw("danger", f"Unknown class: {class_id}")

    old = d.document_class_id
    d.document_class_id       = class_id
    d.document_class_override = class_id   # record that it was overridden
    d.classification_confidence = 1.0      # operator is authoritative
    db.commit()

    db.add(PipelineEvent(
        document_id=doc_id,
        state="RECLASSIFIED",
        agent="operator",
        detail=f"Class changed from {old} → {class_id} ({dc.name}) by operator",
    ))
    db.commit()

    # Return refreshed banner fragment
    doc = _doc_detail(db, doc_id)
    all_classes = db.query(DocumentClass).filter(DocumentClass.active == True).order_by(DocumentClass.id).all()
    return templates.TemplateResponse("dashboard/partials/class_banner.html", {
        "request": request,
        "doc": doc,
        "all_classes": all_classes,
    })


# ── Re-extract (HTMX — re-runs extraction on stored PDF, returns fields tab) ──

@router.post("/docs/{doc_id}/re-extract", response_class=HTMLResponse)
def doc_re_extract(
    doc_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Re-run AI free-form extraction on the stored PDF for a document that previously
    got 0 fields. Updates fields in-place and returns refreshed fields panel HTML.
    """
    import datetime as _dt
    from app.pipeline.states import PipelineState
    from app.agents.extraction import ExtractionAgent
    from app.agents.variant_discovery import VariantDiscoveryAgent

    d = db.query(Document).filter(Document.id == doc_id).first()
    if not d:
        return _toast_raw("danger", "Document not found")

    pdf_path = _pdf_path_for(doc_id)
    if not pdf_path.exists():
        return _toast_raw("danger", "PDF not on disk — please re-upload this document")

    pdf_bytes = pdf_path.read_bytes()

    # Reset to allow clean re-run
    d.status      = PipelineState.EXTRACTING
    d.variant_id  = None
    d.variant_key = None
    d.updated_at  = _dt.datetime.utcnow()
    db.add(PipelineEvent(
        document_id=doc_id,
        state=PipelineState.EXTRACTING,
        agent="operator",
        detail="Operator triggered re-extraction (AI free-form discovery).",
    ))
    db.commit()

    # Run extraction synchronously so fields are ready before we respond
    extraction_agent = ExtractionAgent(db)
    fields = extraction_agent.extract(d, pdf_bytes)

    if fields:
        db.refresh(d)
        variant_agent = VariantDiscoveryAgent(db)
        d.variant_id  = variant_agent.discover(d)
        db.commit()

    d.status     = PipelineState.NEEDS_REVIEW
    d.updated_at = _dt.datetime.utcnow()
    db.commit()

    # Return the refreshed fields panel for HTMX to swap in
    doc = _doc_detail(db, doc_id)
    return templates.TemplateResponse("dashboard/partials/fields_panel.html", {
        "request": request,
        "doc": doc,
        "tables":     (doc or {}).get("tables", []),
        "has_tables": (doc or {}).get("has_tables", False),
    })


# ── Confirm field schema from ZERO_SHOT discovery (HTMX) ─────────────────────

@router.post("/docs/{doc_id}/suggest-renames")
async def doc_suggest_renames(doc_id: str, request: Request, db: Session = Depends(get_db)):
    """
    AI-assisted field rename suggestions.
    Accepts JSON body: {"field_names": ["Payment Terms", "P.O. No.", ...]}
    Returns JSON: {"Payment Terms": "payment_terms", "P.O. No.": "po_number", ...}
    """
    from fastapi.responses import JSONResponse

    if not settings.anthropic_api_key:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set"}, status_code=400)

    try:
        body = await request.json()
        field_names: list[str] = body.get("field_names", [])
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if not field_names:
        return JSONResponse({})

    # Build the prompt
    names_list = "\n".join(f"  - {n}" for n in field_names[:50])
    prompt = (
        "You are a data field naming expert for a logistics and procurement ERP system.\n\n"
        "Below are field labels extracted verbatim from a business document (invoice, PO, RFQ, etc.).\n"
        "Suggest a clean, concise snake_case canonical name for each one that would suit an ERP API.\n\n"
        "Rules:\n"
        "- lowercase, underscores only — no spaces, hyphens, or special chars\n"
        "- Short but descriptive: 1-4 words max\n"
        "- Standard ERP terminology: po_number, invoice_date, payment_terms, delivery_address, etc.\n"
        "- If the name is already good snake_case, return it unchanged\n\n"
        f"Field labels:\n{names_list}\n\n"
        'Return ONLY a JSON object mapping original label → suggested canonical name:\n'
        '{"original label": "suggested_name", ...}\n'
        "No explanation, no markdown fences."
    )

    try:
        import anthropic
        client_ai = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client_ai.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        import re as _re
        raw = _re.sub(r"^```[a-z]*\n?", "", raw)
        raw = _re.sub(r"\n?```$", "", raw)
        suggestions: dict = json.loads(raw)
        # Validate: keys must be strings, values must be snake_case strings
        clean = {}
        for orig, suggested in suggestions.items():
            if isinstance(orig, str) and isinstance(suggested, str):
                clean[orig] = _re.sub(r"[^a-z0-9_]+", "_", suggested.lower()).strip("_")
        return JSONResponse(clean)
    except Exception as exc:
        logger.warning("suggest-renames: Haiku failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/docs/{doc_id}/confirm-fields", response_class=HTMLResponse)
async def doc_confirm_fields(
    doc_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Operator confirms which discovered fields to keep.

    Writes the confirmed schema to:
      1. DocumentVariant (if this doc has a variant_id) — most specific, takes priority
      2. DocumentTypeProfile (class-level fallback for docs without a variant yet)

    Form data:  confirmed_fields = ["material_number","rfq_date", ...]   (checkboxes)
                rename_{old_name} = "new_name"                           (optional rename)
    """
    import datetime as _dt
    from app.models.document import DocumentVariant
    from app.models.client import DocumentTypeProfile
    from app.models.extracted_field import ExtractedField

    form      = await request.form()
    confirmed = list(form.getlist("confirmed_fields"))
    if not confirmed:
        return _toast_raw("warn", "No fields selected — schema unchanged.")

    d = db.query(Document).filter(Document.id == doc_id).first()
    if not d or not d.document_class_id:
        return _toast_raw("danger", "Document not found")

    # Apply optional renames: form key "rename_{old}" → new snake_case name
    renamed: dict[str, str] = {}
    for key, val in form.items():
        if key.startswith("rename_") and val.strip():
            old_name = key[len("rename_"):]
            renamed[old_name] = val.strip().lower().replace(" ", "_")

    # Pull example values from already-extracted scalar fields (exclude tables)
    existing = {
        ef.field_name: ef.field_value
        for ef in db.query(ExtractedField).filter(ExtractedField.document_id == doc_id).all()
        if ef.field_value and (getattr(ef, "field_type", "scalar") or "scalar") == "scalar"
    }

    schema: dict = {}
    for fname in confirmed:
        final_name = renamed.get(fname, fname)
        # Build aliases: always include the raw extracted name AND the canonical name.
        # Later extractions may add more aliases via the Documents → variant panel.
        aliases = list(dict.fromkeys([fname, final_name]))
        schema[final_name] = {
            "aliases":     aliases,
            "required":    True,
            "example":     existing.get(fname, ""),
        }

    schema_json = json.dumps(schema)
    now         = _dt.datetime.utcnow()

    # ── Write to variant (primary target) ─────────────────────────────────────
    variant     = None
    target_desc = ""
    if d.variant_id:
        variant = db.query(DocumentVariant).filter(DocumentVariant.id == d.variant_id).first()

    if variant:
        variant.field_schema_json = schema_json
        variant.learning_stage    = "LEARNED"
        variant.updated_at        = now
        db.commit()
        target_desc = f"variant '{variant.variant_label or variant.variant_key}'"

    # ── Also update class-level DTP so it knows the canonical fields ──────────
    dtp = db.query(DocumentTypeProfile).filter_by(
        client_id="cp_001", document_class_id=d.document_class_id
    ).first()
    if dtp:
        # Only overwrite class schema if it's still blank (don't stomp a prior confirmation)
        if not dtp.field_schema_json:
            dtp.field_schema_json  = schema_json
            dtp.schema_proposed_at = now
            dtp.learning_stage     = "LEARNED"
            dtp.confirmed          = True
            dtp.updated_at         = now
        db.commit()
        if not target_desc:
            target_desc = f"class '{d.document_class.name if d.document_class else d.document_class_id}'"

    db.add(PipelineEvent(
        document_id=doc_id,
        state="SCHEMA_CONFIRMED",
        agent="operator",
        detail=(
            f"Operator confirmed {len(schema)} fields for {target_desc}. "
            f"Stage → LEARNED. Fields: {', '.join(schema.keys())}"
        ),
    ))
    db.commit()

    # ── Opportunistically run FormatAgent (non-fatal) ─────────────────────────
    if variant and variant.id:
        try:
            from app.agents.format_agent import FormatAgent as _FormatAgent
            _fa = _FormatAgent(db)
            _fa.run_for_variant(variant.id)
        except Exception as _fa_exc:
            logger.warning("confirm-fields: FormatAgent non-fatal: %s", _fa_exc)

    class_name = d.document_class.name if d.document_class else d.document_class_id
    return _toast_raw("success",
        f"✓ {len(schema)} fields locked for {target_desc}. "
        f"All future <strong>{class_name}</strong> docs from this issuer will use this schema."
    )


@router.post("/setup/variants/{variant_id}/run-format-agent", response_class=HTMLResponse)
async def setup_run_format_agent(variant_id: str, db: Session = Depends(get_db)):
    """
    Manually trigger the FormatAgent for a variant.
    Returns an HTML fragment (HTMX-friendly) with a summary toast + updated hints.
    """
    from app.models.document import DocumentVariant
    from app.agents.format_agent import FormatAgent as _FormatAgent

    variant = db.query(DocumentVariant).filter(DocumentVariant.id == variant_id).first()
    if not variant:
        return _toast_raw("danger", "Variant not found.")

    agent = _FormatAgent(db)
    result = agent.run_for_variant(variant_id)

    if "error" in result:
        return _toast_raw("danger", f"FormatAgent error: {result['error']}")

    if result.get("note") == "no_schema":
        return _toast_raw("warn", "No confirmed schema yet — confirm fields first.")

    if result.get("note") == "no_values_yet":
        return _toast_raw("warn", "Not enough extracted documents yet (need ≥ 3 values per field).")

    updated   = result.get("fields_updated", 0)
    haiku_cnt = result.get("haiku_audited", 0)
    field_types = result.get("field_types", {})

    # Build a compact type-badge list for inline display
    badge_html = ""
    type_colors = {
        "integer": "#4f8ef7", "decimal": "#4f8ef7",
        "currency_amt": "#2eb885", "currency_code": "#2eb885",
        "date": "#e6a817", "datetime": "#e6a817",
        "code": "#9b59b6", "incoterm": "#e74c3c",
        "country_code": "#1abc9c", "percentage": "#f39c12",
        "boolean": "#95a5a6", "enum": "#7f8c8d",
        "name": "#3498db", "text": "#bdc3c7",
    }
    badge_parts = []
    for fname, ftype in sorted(field_types.items()):
        color = type_colors.get(ftype, "#bdc3c7")
        badge_parts.append(
            f'<span style="background:{color};color:#fff;padding:1px 6px;border-radius:3px;'
            f'font-size:10px;font-weight:600;margin:1px;">{fname}:{ftype}</span>'
        )
    if badge_parts:
        badge_html = "<div style='margin-top:6px;line-height:1.8;'>" + " ".join(badge_parts) + "</div>"

    haiku_note = f" ({haiku_cnt} AI-audited)" if haiku_cnt else ""
    msg = (
        f"✓ Format analysis complete. <strong>{updated}</strong> hints updated{haiku_note}."
        + badge_html
    )
    return _toast_raw("success", msg)


@router.get("/docs/{doc_id}/pdf-page/{page_num}")
def doc_pdf_page(doc_id: str, page_num: int = 0, dpi: int = 130):
    """Render a PDF page as JPEG. Uses PyMuPDF (fitz) with pdf2image fallback."""
    p = _pdf_path_for(doc_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="PDF not stored for this document")
    try:
        import io as _io
        jpeg_bytes: Optional[bytes] = None

        # ── Try PyMuPDF first (no system dependencies) ─────────────────
        try:
            import fitz
            fitz_doc = fitz.open(str(p))
            page_num = max(0, min(page_num, len(fitz_doc) - 1))
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = fitz_doc[page_num].get_pixmap(matrix=mat, alpha=False)
            jpeg_bytes = pix.tobytes("jpeg")
            fitz_doc.close()
        except ImportError:
            pass

        # ── Fallback: pdf2image (requires poppler) ──────────────────────
        if jpeg_bytes is None:
            from pdf2image import convert_from_path
            imgs = convert_from_path(
                str(p), dpi=dpi,
                first_page=page_num + 1, last_page=page_num + 1,
            )
            if not imgs:
                raise HTTPException(status_code=404, detail=f"Page {page_num} not found")
            buf = _io.BytesIO()
            imgs[0].save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()

        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=300"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/docs/{doc_id}/field-rects")
def doc_field_rects(
    doc_id: str,
    value: str = Query(..., description="Field value to search for in the PDF"),
    page: int = Query(default=0, description="Page number (0-indexed)"),
    dpi: int = Query(default=130),
):
    """
    Search for a text string in a PDF page and return pixel-space bounding rects.
    Used by the viewer's click-to-highlight feature.
    Returns {rects: [{x,y,w,h}, ...], found: bool}
    """
    p = _pdf_path_for(doc_id)
    if not p.exists():
        return {"rects": [], "found": False, "reason": "pdf_not_stored"}

    val = value.strip()
    if len(val) < 2:
        return {"rects": [], "found": False, "reason": "value_too_short"}

    def _candidate_searches(v: str) -> list[str]:
        """
        Conservative search candidates — full value first, then safe normalisation
        variants only. No sliding windows or short tokens: those cause false positives
        (e.g. "30" matching "30%" or "item 30" on the wrong line).
        """
        import re as _re
        candidates: list[str] = [v]

        # Strip AI-added surrounding quotes / brackets
        stripped = v.strip('"\'()[]').strip()
        if stripped and stripped != v:
            candidates.append(stripped)

        # Punctuation-swap variants (date separators, dash/slash)
        for a, b in [("-", "/"), ("/", "-"), (".", "/"), (",", "")]:
            alt = v.replace(a, b).strip()
            if alt and alt != v:
                candidates.append(alt)

        # Split on common AI-inserted join separators (" — ", " | ", " / ").
        # Add the RIGHT segment first — when extraction accidentally captures surrounding
        # text (e.g. a stamp code), the right part is usually the actual field value.
        # Then add the left part as a fallback.  Skip parts that themselves contain the
        # separator (e.g. "TML/MUM/FOO" in "TML/MUM/FOO / 30 Days") to avoid false
        # positives caused by path-like codes matching elsewhere in the document.
        for sep in [" — ", " | ", " / "]:
            if sep in v:
                parts = v.split(sep, 1)
                left  = parts[0].strip()
                right = parts[1].strip() if len(parts) > 1 else ""
                # Right first: more likely to be the discriminating human-readable value
                if right and len(right) >= 6:
                    candidates.append(right)
                # Left only if it doesn't look like a code/path (no embedded slashes)
                if left and len(left) >= 6 and "/" not in left:
                    candidates.append(left)

        # Dedupe preserving order, minimum length 4 to avoid matching noise
        seen: set[str] = set()
        out: list[str] = []
        for c in candidates:
            c = c.strip()
            if c and len(c) >= 4 and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    try:
        # ── PyMuPDF: full coordinate search ────────────────────────────
        try:
            import fitz
            fitz_doc = fitz.open(str(p))
            n_pages_fitz = len(fitz_doc)
            scale = dpi / 72
            # Search order: current page first, then all others in doc order.
            # Current page is first so the user gets the result on the page they're
            # looking at if the text appears there.
            pages_to_search = [page] + [pg for pg in range(n_pages_fitz) if pg != page]
            found_page = None
            all_rects: list[dict] = []
            matched_term: str = val

            def _body_rect(r, pg: "fitz.Page") -> bool:
                """Return True if rect y-centre is in the body region (12%–88%)."""
                ph = pg.rect.height
                return ph > 0 and 0.12 <= (r.y0 / ph) <= 0.88

            for search_term in _candidate_searches(val):
                if found_page is not None:
                    break

                # Collect matches across all pages for this search term
                candidate_pages: list[tuple] = []  # (pg_num, rects)
                for pg_num in pages_to_search:
                    rects = fitz_doc[pg_num].search_for(search_term, quads=False)
                    if rects:
                        candidate_pages.append((pg_num, rects))

                if not candidate_pages:
                    continue

                # If only one page has matches, use it directly.
                # If multiple pages have matches, prefer a body-region match over
                # a header/footer match — this handles repeated header text like
                # "PO-1234" stamped on every page as a watermark or header.
                chosen_pg, chosen_rects = candidate_pages[0]
                if len(candidate_pages) > 1:
                    for pg_num, rects in candidate_pages:
                        pg = fitz_doc[pg_num]
                        if _body_rect(rects[0], pg):
                            chosen_pg, chosen_rects = pg_num, rects
                            break

                found_page    = chosen_pg
                matched_term  = search_term
                all_rects = [
                    {
                        "x": int(r.x0 * scale),
                        "y": int(r.y0 * scale),
                        "w": max(4, int((r.x1 - r.x0) * scale)),
                        "h": max(4, int((r.y1 - r.y0) * scale)),
                        "page": chosen_pg,
                    }
                    for r in chosen_rects[:8]
                ]

            fitz_doc.close()
            partial = matched_term != val and bool(all_rects)
            return {
                "rects": all_rects,
                "found": bool(all_rects),
                "found_page": found_page,
                "partial_match": partial,
                "matched_term": matched_term if partial else None,
            }

        except ImportError:
            pass

        # ── Fallback: pypdf text search (no coordinates — returns found=True only) ──
        import pypdf
        reader = pypdf.PdfReader(str(p))
        # Try several normalised forms of the value to handle PDF encoding quirks
        candidates = {val.lower()}
        candidates.add(val.lower().replace("-", "/"))
        candidates.add(val.lower().replace("/", "-"))
        candidates.add(val.lower().replace(" ", ""))
        candidates.add(val.lower().replace(",", ""))
        for pg_num, pg in enumerate(reader.pages):
            text = (pg.extract_text() or "").lower().replace("\n", " ")
            if any(c in text for c in candidates):
                return {
                    "rects": [],
                    "found": True,
                    "found_page": pg_num,
                    "no_coords": True,
                    "reason": "fitz_not_installed",
                }
        return {"rects": [], "found": False, "reason": "fitz_not_installed"}

    except Exception as exc:
        return {"rects": [], "found": False, "reason": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
#  SSE stream (optional — for live doc-arrival notifications)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sse")
async def dashboard_sse(request: Request):
    """Server-sent events — fires a 'refresh' event when document count changes."""

    async def stream():
        last_count = -1
        while True:
            if await request.is_disconnected():
                break
            db = SessionLocal()
            try:
                count = db.query(Document).count()
                if count != last_count:
                    last_count = count
                    payload = json.dumps({"count": count})
                    yield f"event: doc_update\ndata: {payload}\n\n"
            except Exception:
                pass
            finally:
                db.close()
            yield ": heartbeat\n\n"
            await asyncio.sleep(4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Setup  — client profile + document type configuration + AI schema discovery
# ─────────────────────────────────────────────────────────────────────────────

# Store uploaded sample PDFs in a separate temp dir
_SAMPLE_STORE = pathlib.Path("/tmp/dokr_samples")
_SAMPLE_STORE.mkdir(parents=True, exist_ok=True)


def _sample_path(sample_id: str) -> pathlib.Path:
    return _SAMPLE_STORE / f"{sample_id}.pdf"


def _get_or_create_default_client(db: Session):
    """Return the first active client, or None."""
    from app.models.client import ClientProfile
    return db.query(ClientProfile).filter(ClientProfile.active.is_(True)).first()


def _setup_context(db: Session) -> dict:
    """Assemble all data for the Setup page."""
    from app.models.client import ClientProfile, DocumentTypeProfile
    from app.models.document import DocumentClass
    import json

    client = _get_or_create_default_client(db)

    # All document classes with their type profile (if any) for this client
    classes = db.query(DocumentClass).filter(DocumentClass.active.is_(True)).order_by(DocumentClass.id).all()
    doc_types = []
    for dc in classes:
        dtp = None
        if client:
            dtp = (
                db.query(DocumentTypeProfile)
                .filter(
                    DocumentTypeProfile.client_id == client.id,
                    DocumentTypeProfile.document_class_id == dc.id,
                    DocumentTypeProfile.active.is_(True),
                )
                .first()
            )

        schema = {}
        if dtp and dtp.field_schema_json:
            try:
                schema = json.loads(dtp.field_schema_json)
            except Exception:
                pass

        # Target fields from extraction agent (hardcoded defaults)
        from app.agents.extraction import FIELDS_BY_CLASS
        default_fields = FIELDS_BY_CLASS.get(dc.id, [])

        doc_types.append(dict(
            id=dc.id,
            name=dc.name,
            slug=dc.slug,
            treatment=dc.treatment,
            dtp_id=dtp.id if dtp else None,
            sample_count=dtp.sample_count if dtp else 0,
            confirmed=dtp.confirmed if dtp else False,
            ai_description=dtp.ai_description if dtp else None,
            field_schema=schema,
            default_fields=default_fields,
            samples=_sample_list(db, dtp.id) if dtp else [],
            # ── Learning progress ────────────────────────────────────────────
            learning_stage=dtp.learning_stage if dtp else "ZERO_SHOT",
            stage_label=dtp.stage_label if dtp else "Learning (0–2 docs)",
            stage_pct=dtp.stage_pct if dtp else 0,
            doc_count=dtp.doc_count if dtp else 0,
            needs_attention=dtp.needs_attention if dtp else False,
            field_stats=dtp.computed_field_stats() if dtp else [],
            parsability=dtp.parsability if dtp else "UNKNOWN",
            parsability_reason=dtp.parsability_reason if dtp else "",
            has_fast_patterns=bool(dtp.generated_patterns_json) if dtp else False,
            check_signature=dtp.check_signature if dtp else False,
        ))

    settings_data = {}
    if client and client.settings_json:
        try:
            settings_data = json.loads(client.settings_json)
        except Exception:
            pass

    return dict(client=client, doc_types=doc_types, settings_data=settings_data)


def _sample_list(db: Session, dtp_id: str) -> list[dict]:
    from app.models.client import SampleDocument
    import json
    rows = (
        db.query(SampleDocument)
        .filter(SampleDocument.document_type_profile_id == dtp_id)
        .order_by(SampleDocument.created_at.desc())
        .all()
    )
    result = []
    for s in rows:
        analysis = {}
        if s.ai_analysis_json:
            try:
                analysis = json.loads(s.ai_analysis_json)
            except Exception:
                pass
        result.append(dict(
            id=s.id,
            file_name=s.file_name,
            file_size_kb=round(s.file_size_bytes / 1024, 1),
            created_at=s.created_at.strftime("%d %b %H:%M") if s.created_at else "—",
            has_analysis=bool(analysis),
            notes=analysis.get("notes", ""),
        ))
    return result


@router.get("/api-reference", response_class=HTMLResponse)
def api_reference_page(request: Request):
    """Static API reference page — no DB needed."""
    return templates.TemplateResponse(
        "dashboard/api_docs.html",
        {"request": request, "page": "api-reference"},
    )


@router.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request, db: Session = Depends(get_db)):
    ctx = _setup_context(db)
    from app.models.client import DocumentTypeProfile
    attention_count = (
        db.query(DocumentTypeProfile)
        .filter(DocumentTypeProfile.learning_stage == "LEARNED_PROPOSED")
        .count()
    )
    return templates.TemplateResponse(
        "dashboard/setup.html",
        {
            "request": request,
            "page": "setup",
            "review_count": db.query(Document).filter(Document.status == "NEEDS_REVIEW").count(),
            "setup_attention": attention_count,
            "api_key_set": bool(settings.anthropic_api_key),
            **ctx,
        },
    )


@router.post("/setup/client", response_class=HTMLResponse)
async def setup_update_client(
    request: Request,
    name: str = Form(...),
    display_name: str = Form(...),
    domain: str = Form(""),
    industry: str = Form(""),
    erp_system: str = Form(""),
    db: Session = Depends(get_db),
):
    from app.models.client import ClientProfile
    client = _get_or_create_default_client(db)
    if not client:
        client = ClientProfile(id="cp_001")
        db.add(client)

    client.name = name.strip()
    client.display_name = display_name.strip()
    client.domain = domain.strip() or None
    client.industry = industry.strip() or None
    client.erp_system = erp_system.strip() or None
    db.commit()
    return _toast("success", "Client profile saved.")


@router.post("/setup/doctype/{dtp_id}/sample", response_class=HTMLResponse)
async def setup_upload_sample(
    request: Request,
    dtp_id: str,
    sample_pdf: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a sample PDF for a document type profile."""
    import hashlib
    from app.models.client import DocumentTypeProfile, SampleDocument

    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.id == dtp_id).first()
    if not dtp:
        raise HTTPException(status_code=404, detail="DocumentTypeProfile not found")

    content = await sample_pdf.read()
    sha = hashlib.sha256(content).hexdigest()

    # Deduplicate by hash
    existing = (
        db.query(SampleDocument)
        .filter(
            SampleDocument.document_type_profile_id == dtp_id,
            SampleDocument.file_sha256 == sha,
        )
        .first()
    )
    if existing:
        return _toast("warn", f"{sample_pdf.filename} already uploaded.")

    # Extract text from PDF for storage (so AI doesn't need raw PDF later)
    extracted_text = ""
    try:
        import pypdf, io as _io
        reader = pypdf.PdfReader(_io.BytesIO(content))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
        extracted_text = " ".join(parts)[:20000]
    except Exception:
        pass

    sample = SampleDocument(
        document_type_profile_id=dtp_id,
        file_name=sample_pdf.filename or "sample.pdf",
        file_sha256=sha,
        file_size_bytes=len(content),
        extracted_text=extracted_text,
    )
    db.add(sample)
    dtp.sample_count = (
        db.query(SampleDocument)
        .filter(SampleDocument.document_type_profile_id == dtp_id)
        .count()
    ) + 1
    db.commit()

    # Save PDF to temp store
    _sample_path(sample.id).write_bytes(content)

    return HTMLResponse(
        content=(
            f'<div class="sample-row" id="sr-{sample.id}">'
            f'<span class="mono fs-12 text-muted">{sample.file_name}</span>'
            f'<span class="fs-11 text-dim">{round(len(content)/1024,1)} KB · just now</span>'
            f'</div>'
            + _toast_raw("success", f"Uploaded {sample.file_name}")
        )
    )


@router.post("/setup/doctype/{dtp_id}/discover", response_class=HTMLResponse)
async def setup_discover_schema(
    request: Request,
    dtp_id: str,
    db: Session = Depends(get_db),
):
    """
    Run AI schema discovery across all uploaded samples for a document type.
    Analyses texts → Claude suggests field schema → stored as field_schema_json.
    """
    import json as _json
    from app.models.client import DocumentTypeProfile, SampleDocument
    from app.models.document import DocumentClass

    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.id == dtp_id).first()
    if not dtp:
        raise HTTPException(status_code=404, detail="DocumentTypeProfile not found")

    if not settings.anthropic_api_key:
        return _toast("warn", "ANTHROPIC_API_KEY not set — cannot run AI discovery.")

    samples = (
        db.query(SampleDocument)
        .filter(SampleDocument.document_type_profile_id == dtp_id)
        .all()
    )
    if not samples:
        return _toast("warn", "Upload at least one sample PDF first.")

    dc = db.query(DocumentClass).filter(DocumentClass.id == dtp.document_class_id).first()
    dc_name = dc.name if dc else dtp.document_class_id

    # Collect text from samples (up to 4000 chars each)
    sample_texts = []
    for i, s in enumerate(samples[:5], 1):
        text = (s.extracted_text or "")[:4000]
        if text.strip():
            sample_texts.append(f"=== Sample {i}: {s.file_name} ===\n{text}")

    if not sample_texts:
        return _toast("warn", "No extracted text found in samples — PDFs may be image-only.")

    combined = "\n\n".join(sample_texts)

    # Get default fields from extraction agent as a baseline suggestion
    from app.agents.extraction import FIELDS_BY_CLASS, FIELD_DESCRIPTIONS
    default_fields = FIELDS_BY_CLASS.get(dtp.document_class_id, [])
    default_field_hint = ", ".join(default_fields) if default_fields else "unknown"

    prompt = (
        f"You are an expert document analyst for a logistics and procurement company.\n\n"
        f"You have been given {len(samples)} sample(s) of a '{dc_name}' document.\n\n"
        f"SAMPLE DOCUMENT TEXTS:\n{combined}\n\n"
        f"TASK: Analyse the samples and identify all fields that appear consistently.\n"
        f"The system currently extracts these fields: [{default_field_hint}]\n\n"
        f"Return a JSON object with this structure:\n"
        "{\n"
        '  "fields": {\n'
        '    "field_name": {\n'
        '      "description": "what this field means in the context of this document type",\n'
        '      "required": true/false,\n'
        '      "format_hint": "e.g. date DD/MM/YYYY, reference like ABC/12345, amount in GBP",\n'
        '      "example_values": ["val1", "val2"]\n'
        "    }\n"
        "  },\n"
        '  "notes": "brief AI observations about this document type",\n'
        '  "ai_description": "one paragraph describing what this document type is and how it looks"\n'
        "}\n\n"
        "Use snake_case for field names. Include ALL fields that seem important for processing.\n"
        "Return ONLY valid JSON, no markdown fences."
    )

    try:
        import anthropic
        client_ai = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client_ai.messages.create(
            model=settings.anthropic_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        discovered = _json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception as exc:
        return _toast("danger", f"AI discovery failed: {exc}")

    # Save the suggested schema
    dtp.field_schema_json = _json.dumps(discovered.get("fields", {}))
    dtp.ai_description = discovered.get("ai_description", "")
    db.commit()

    # Build HTML snippet to swap into the schema panel
    fields = discovered.get("fields", {})
    notes = discovered.get("notes", "")
    ai_desc = discovered.get("ai_description", "")

    field_rows = ""
    for fname, fdata in fields.items():
        req_badge = '<span class="chip" style="color:var(--accent);background:var(--accent-dim);font-size:9px;">required</span>' if fdata.get("required") else ""
        field_rows += (
            f'<div class="flex gap-12" style="border-bottom:1px solid var(--border);padding:8px 0;">'
            f'<span class="mono fs-11 text-muted" style="flex:1;text-transform:uppercase;">{fname}</span>'
            f'<span class="fs-12 text-text" style="flex:3;">{fdata.get("description","")}</span>'
            f'<span class="mono fs-11 text-dim" style="flex:1.5;">{fdata.get("format_hint","")}</span>'
            f'<span style="flex:none;">{req_badge}</span>'
            f'</div>'
        )

    notes_html = f'<p class="fs-12 text-dim" style="margin-top:10px;">{notes}</p>' if notes else ""
    n_fields = len(fields)
    html = (
        f'<div id="schema-panel-{dtp_id}" style="margin-top:16px;">'
        f'<div class="panel-title">AI-Suggested Schema <span class="chip" style="color:var(--purple);background:#161B3A;margin-left:8px;">draft</span></div>'
        f'<p class="fs-12 text-muted" style="margin:8px 0 12px;">{ai_desc}</p>'
        f'{field_rows}'
        f'{notes_html}'
        f'<div style="margin-top:16px;display:flex;gap:8px;">'
        f'<form hx-post="/dashboard/setup/doctype/{dtp_id}/confirm" hx-swap="outerHTML" hx-target="#schema-panel-{dtp_id}">'
        f'<button class="btn btn-primary btn-sm" type="submit">✓ Confirm schema</button>'
        f'</form>'
        f'<button class="btn btn-secondary btn-sm" '
        f'onclick="this.closest(\'[id^=schema-panel]\').remove()">Dismiss</button>'
        f'</div>'
        f'</div>'
        + _toast_raw("success", f"Schema discovered — {n_fields} fields suggested.")
    )
    return HTMLResponse(content=html)


@router.post("/setup/doctype/{dtp_id}/assess-parsability", response_class=HTMLResponse)
async def setup_assess_parsability(
    request: Request,
    dtp_id: str,
    db: Session = Depends(get_db),
):
    """Run AI parsability assessment for a document type profile."""
    from app.models.client import DocumentTypeProfile
    from app.agents.schema_learner import SchemaLearnerAgent

    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.id == dtp_id).first()
    if not dtp:
        raise HTTPException(status_code=404)
    if not settings.anthropic_api_key:
        return _toast("warn", "ANTHROPIC_API_KEY not set.")

    learner = SchemaLearnerAgent(db)
    learner.run_parsability_assessment(dtp.document_class_id)
    db.refresh(dtp)

    color_map = {
        "OPTIMISABLE": ("var(--accent)", "var(--accent-dim)"),
        "CAN_LEARN":   ("var(--warn)", "#2B1D0A"),
        "ALWAYS_AI":   ("var(--purple)", "#161B3A"),
        "UNKNOWN":     ("var(--dim)", "#111"),
    }
    color, bg = color_map.get(dtp.parsability, color_map["UNKNOWN"])

    html = (
        f'<div id="parsability-{dtp_id}" style="margin-top:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span class="badge" style="color:{color};background:{bg};">{dtp.parsability}</span>'
        f'<span class="fs-12 text-muted">{dtp.parsability_reason}</span>'
        f'</div>'
        + (
            f'<form hx-post="/dashboard/setup/doctype/{dtp_id}/generate-patterns" '
            f'hx-target="#parsability-{dtp_id}" hx-swap="outerHTML" style="margin-top:10px;">'
            f'<button class="btn btn-primary btn-sm" type="submit">⚡ Generate fast patterns</button>'
            f'</form>'
            if dtp.parsability == "OPTIMISABLE" else ""
        )
        + f'</div>'
        + _toast_raw("success", f"Parsability assessed: {dtp.parsability}")
    )
    return HTMLResponse(content=html)


@router.post("/setup/doctype/{dtp_id}/generate-patterns", response_class=HTMLResponse)
async def setup_generate_patterns(
    request: Request,
    dtp_id: str,
    db: Session = Depends(get_db),
):
    """Generate AI fast-path patterns for an OPTIMISABLE document type."""
    from app.models.client import DocumentTypeProfile
    from app.agents.schema_learner import SchemaLearnerAgent

    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.id == dtp_id).first()
    if not dtp:
        raise HTTPException(status_code=404)

    learner = SchemaLearnerAgent(db)
    learner.generate_fast_patterns(dtp.document_class_id)
    db.refresh(dtp)

    if dtp.generated_patterns_json:
        import json as _j
        patterns = _j.loads(dtp.generated_patterns_json)
        n = len(patterns)
        # Advance to OPTIMISED
        dtp.learning_stage = "OPTIMISED"
        db.commit()
        return HTMLResponse(content=(
            f'<div id="parsability-{dtp_id}">'
            f'<span class="badge" style="color:var(--accent);background:var(--accent-dim);">OPTIMISED</span>'
            f'<span class="fs-12 text-muted" style="margin-left:8px;">{n} fast-path patterns generated and active.</span>'
            f'</div>'
            + _toast_raw("success", f"Fast patterns generated — {n} fields on fast path.")
        ))

    return _toast("warn", "Pattern generation returned nothing — try again after more documents.")


@router.post("/setup/doctype/{dtp_id}/confirm", response_class=HTMLResponse)
async def setup_confirm_schema(
    request: Request,
    dtp_id: str,
    db: Session = Depends(get_db),
):
    from app.models.client import DocumentTypeProfile
    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.id == dtp_id).first()
    if not dtp:
        raise HTTPException(status_code=404, detail="DocumentTypeProfile not found")

    dtp.confirmed      = True
    dtp.learning_stage = "LEARNED"
    db.commit()
    return HTMLResponse(content=(
        '<div style="display:flex;align-items:center;gap:8px;padding:12px;">'
        '<span style="color:var(--accent);font-size:16px;">✓</span>'
        '<span class="fs-13 text-text">Schema confirmed — AI now extracts against this schema precisely.</span>'
        '</div>'
        + _toast_raw("success", "Schema confirmed. Stage advanced to LEARNED.")
    ))


@router.post("/setup/doctype/{dtp_id}/toggle-signature", response_class=HTMLResponse)
async def setup_toggle_signature(
    request: Request,
    dtp_id: str,
    db: Session = Depends(get_db),
):
    """Toggle the check_signature flag for a DocumentTypeProfile. Returns updated toggle HTML."""
    from app.models.client import DocumentTypeProfile
    dtp = db.query(DocumentTypeProfile).filter(DocumentTypeProfile.id == dtp_id).first()
    if not dtp:
        return _toast_raw("danger", "DocumentTypeProfile not found")

    dtp.check_signature = not dtp.check_signature
    db.commit()

    new_val = dtp.check_signature
    color = "var(--accent)" if new_val else "var(--muted)"
    bg    = "var(--accent-dim)" if new_val else "var(--surface-2)"
    label = "✍ Signature detection ON" if new_val else "○ Signature detection OFF"
    note  = ("Every new document of this type will be scanned for signatures."
             if new_val else
             "Signature scanning is disabled for this document type.")
    return HTMLResponse(content=(
        f'<div id="sig-toggle-{dtp_id}">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<button class="btn btn-sm" style="color:{color};background:{bg};border:1px solid {color};"'
        f'        hx-post="/dashboard/setup/doctype/{dtp_id}/toggle-signature"'
        f'        hx-target="#sig-toggle-{dtp_id}" hx-swap="outerHTML">'
        f'  {label}'
        f'</button>'
        f'<span class="fs-11 text-dim">{note}</span>'
        f'</div>'
        f'</div>'
    ))


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline stage toggles
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/setup/pipeline", response_class=HTMLResponse)
async def setup_pipeline_panel(request: Request, db: Session = Depends(get_db)):
    """Return the pipeline stage toggle panel (full page fragment for HTMX)."""
    from app.agents.registry import PIPELINE_STAGES
    from app.models.client import ClientAgentConfig
    cac = db.query(ClientAgentConfig).filter(ClientAgentConfig.client_id == "cp_001").first()
    disabled = set(json.loads(cac.disabled_stages_json or "[]")) if cac else set()
    stages = [
        {
            "key": s.key,
            "label": s.label,
            "description": s.description,
            "skippable": s.skippable,
            "enabled": s.key.upper() not in disabled,
        }
        for s in PIPELINE_STAGES
    ]
    return templates.TemplateResponse(
        "dashboard/partials/pipeline_stages.html",
        {"request": request, "stages": stages},
    )


@router.post("/setup/pipeline/toggle", response_class=HTMLResponse)
async def setup_toggle_pipeline_stage(
    request: Request,
    db: Session = Depends(get_db),
):
    """Toggle a pipeline stage on or off. Expects form fields: stage, enabled (0/1)."""
    from datetime import datetime
    from app.agents.registry import PIPELINE_STAGE_MAP
    from app.models.client import ClientAgentConfig
    from app.utils.ids import generate_id

    form = await request.form()
    stage_key = (form.get("stage") or "").strip().lower()
    enabled   = str(form.get("enabled", "1")) == "1"

    if not stage_key or stage_key not in PIPELINE_STAGE_MAP:
        return _toast_raw("danger", f"Unknown pipeline stage: {stage_key!r}")

    stage_info = PIPELINE_STAGE_MAP[stage_key]
    if not stage_info.skippable:
        return _toast_raw("danger", f"Stage '{stage_info.label}' cannot be disabled.")

    cac = db.query(ClientAgentConfig).filter(ClientAgentConfig.client_id == "cp_001").first()
    if not cac:
        cac = ClientAgentConfig(
            id=generate_id("cac"),
            client_id="cp_001",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(cac)

    disabled = set(json.loads(cac.disabled_stages_json or "[]"))
    stage_upper = stage_key.upper()
    if enabled:
        disabled.discard(stage_upper)
    else:
        disabled.add(stage_upper)

    cac.disabled_stages_json = json.dumps(sorted(disabled))
    cac.updated_at = datetime.utcnow()
    db.commit()

    # Return a refreshed toggle button for this specific stage
    color  = "var(--accent)"   if enabled else "var(--muted)"
    bg     = "var(--accent-dim)" if enabled else "var(--surface-2)"
    border = "var(--accent)"   if enabled else "var(--border)"
    label  = "ON"              if enabled else "OFF"
    next_v = "0"               if enabled else "1"
    return HTMLResponse(content=(
        f'<div id="stage-toggle-{stage_key}" style="display:flex;align-items:center;gap:10px;">'
        f'<button class="btn btn-sm" style="min-width:52px;color:{color};background:{bg};'
        f'border:1px solid {border};font-weight:600;"'
        f' hx-post="/dashboard/setup/pipeline/toggle"'
        f' hx-vals=\'{{"stage":"{stage_key}","enabled":"{next_v}"}}\''
        f' hx-target="#stage-toggle-{stage_key}" hx-swap="outerHTML">'
        f'  {label}'
        f'</button>'
        f'<span class="fs-11 text-dim">'
        + (f'{stage_info.label} is active.' if enabled else f'{stage_info.label} is disabled — pipeline will skip this stage.')
        + f'</span></div>'
    ))


# ─────────────────────────────────────────────────────────────────────────────
#  Agents dashboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request, db: Session = Depends(get_db)):
    """Agents dashboard — catalog, run controls, history."""
    from app.agents.audit import AGENT_CATALOG
    from app.models.agent_run import AgentRun
    from app.models.document import DocumentVariant

    # Load last run per agent for the card sub-line
    last_runs: dict[str, dict] = {}
    for entry in AGENT_CATALOG:
        run = (
            db.query(AgentRun)
            .filter(AgentRun.agent_name == entry["name"])
            .order_by(AgentRun.created_at.desc())
            .first()
        )
        if run:
            last_runs[entry["name"]] = {
                "id":         run.id,
                "status":     run.status,
                "summary":    run.summary or "",
                "created_at": run.created_at.strftime("%d %b %H:%M") if run.created_at else "—",
                "duration_ms": run.duration_ms,
            }

    # Recent run history
    recent_runs = (
        db.query(AgentRun)
        .order_by(AgentRun.created_at.desc())
        .limit(20)
        .all()
    )
    run_rows = []
    for r in recent_runs:
        catalog_entry = next((e for e in AGENT_CATALOG if e["name"] == r.agent_name), None)
        run_rows.append({
            "id":           r.id,
            "agent_name":   r.agent_name,
            "agent_label":  catalog_entry["label"] if catalog_entry else r.agent_name,
            "status":       r.status,
            "summary":      r.summary or "",
            "error":        r.error or "",
            "triggered_by": r.triggered_by,
            "duration_ms":  r.duration_ms,
            "created_at":   r.created_at.strftime("%d %b %H:%M") if r.created_at else "—",
        })

    # Stats banner
    total_runs  = db.query(AgentRun).count()
    runs_today  = db.query(AgentRun).filter(
        AgentRun.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    ).count()
    last_run    = db.query(AgentRun).order_by(AgentRun.created_at.desc()).first()

    # Variants for scope selectors (used in run forms)
    variants = db.query(DocumentVariant).order_by(DocumentVariant.variant_label).all()
    variant_opts = [
        {"id": v.id, "label": v.variant_label or v.issuer_slug or v.variant_key or v.id[:8]}
        for v in variants
    ]

    return templates.TemplateResponse("dashboard/agents.html", {
        "request":      request,
        "page":         "agents",
        "catalog":      AGENT_CATALOG,
        "last_runs":    last_runs,
        "run_rows":     run_rows,
        "total_runs":   total_runs,
        "runs_today":   runs_today,
        "last_run_ts":  last_run.created_at.strftime("%d %b %H:%M") if last_run and last_run.created_at else None,
        "variant_opts": variant_opts,
    })


@router.post("/agents/{agent_name}/run", response_class=HTMLResponse)
async def run_agent_dashboard(
    agent_name:       str,
    request:          Request,
    background_tasks: BackgroundTasks,
    db:               Session = Depends(get_db),
):
    """HTMX — trigger an agent run, return polling partial."""
    import uuid as _uuid
    from app.agents.audit import CATALOG_BY_NAME
    from app.models.agent_run import AgentRun

    if agent_name not in CATALOG_BY_NAME:
        return HTMLResponse(
            f'<div style="color:var(--danger);font-size:13px;">Unknown agent: {agent_name}</div>'
        )

    form = await request.form()
    # Collect any numeric / text params from the form
    params: dict = {}
    entry = CATALOG_BY_NAME[agent_name]
    for p in entry.get("params", []):
        raw = form.get(p["key"])
        if raw is not None and str(raw).strip():
            params[p["key"]] = int(raw) if p["type"] == "number" else str(raw).strip()
        elif "default" in p:
            params[p["key"]] = p["default"]

    run_id = f"ar_{_uuid.uuid4().hex[:14]}"
    now    = datetime.utcnow()
    run    = AgentRun(
        id=run_id,
        agent_name=agent_name,
        status="pending",
        triggered_by="manual",
        params_json=json.dumps(params),
        created_at=now,
    )
    db.add(run)
    db.commit()

    # Fire background task
    from app.routers.agents import _execute_run
    background_tasks.add_task(_execute_run, run_id, agent_name, params)

    return templates.TemplateResponse("dashboard/partials/agent_result.html", {
        "request": request,
        "run": {
            "id":         run_id,
            "agent_name": agent_name,
            "status":     "pending",
            "summary":    None,
            "error":      None,
            "result":     None,
            "created_at": now.strftime("%d %b %H:%M"),
            "duration_ms": None,
        },
        "agent_label": entry["label"],
        "polling":     True,
    })


@router.get("/agents/runs/{run_id}", response_class=HTMLResponse)
def get_agent_run_dashboard(run_id: str, request: Request, db: Session = Depends(get_db)):
    """HTMX polling target — returns run status partial."""
    from app.agents.audit import CATALOG_BY_NAME
    from app.models.agent_run import AgentRun

    run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not run:
        return HTMLResponse(
            '<div style="color:var(--danger);font-size:13px;">Run not found</div>'
        )

    entry = CATALOG_BY_NAME.get(run.agent_name, {})
    result = None
    try:
        if run.result_json:
            result = json.loads(run.result_json)
    except Exception:
        pass

    is_terminal = run.status in {"completed", "failed"}

    return templates.TemplateResponse("dashboard/partials/agent_result.html", {
        "request":     request,
        "run": {
            "id":         run.id,
            "agent_name": run.agent_name,
            "status":     run.status,
            "summary":    run.summary,
            "error":      run.error,
            "result":     result,
            "created_at": run.created_at.strftime("%d %b %H:%M") if run.created_at else "—",
            "duration_ms": run.duration_ms,
        },
        "agent_label": entry.get("label", run.agent_name),
        "polling":     not is_terminal,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Toast helper
# ─────────────────────────────────────────────────────────────────────────────

_TOAST_COLORS = {
    "success": ("#00C97A", "#0D2B1A"),
    "warn":    ("#F5A623", "#2B1D0A"),
    "danger":  ("#FF4D4D", "#2B0D0D"),
    "info":    ("#7C84E8", "#161B3A"),
}


def _toast_raw(kind: str = "success", message: str = "") -> str:
    """Return a toast HTML string suitable for embedding in a larger response."""
    color, bg = _TOAST_COLORS.get(kind, _TOAST_COLORS["success"])
    return (
        f'<div id="toast-{kind}" style="'
        f'position:fixed;bottom:24px;right:24px;z-index:999;'
        f'background:{bg};border:1px solid {color};'
        f'color:{color};padding:12px 20px;'
        f'font-family:Inter,sans-serif;font-size:13px;font-weight:500;'
        f'opacity:1;transition:opacity 0.4s ease;"'
        f' hx-on:load="setTimeout(()=>{{this.style.opacity=0;setTimeout(()=>this.remove(),400)}},2500)">'
        f'{message}'
        f'</div>'
    )


def _toast(kind: str = "success", message: str = "") -> HTMLResponse:
    """Return an HTMLResponse wrapping a toast notification."""
    return HTMLResponse(_toast_raw(kind, message))
