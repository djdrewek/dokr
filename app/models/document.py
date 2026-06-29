from datetime import datetime

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DocumentClass(Base):
    """
    Tier 1 — a logical grouping of documents sharing pipeline treatment,
    integration targets, and compliance rules (e.g. 'Supplier Invoice').
    """
    __tablename__ = "document_classes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    treatment: Mapped[str] = mapped_column(String, default="PROCESS")
    # PROCESS | STORE | STORE_AND_FORWARD | GENERATED
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # ── Classifier profile ────────────────────────────────────────────────────
    classifier_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: {
    #   "keywords":          ["purchase order no", "tml/", ...],   ← trigger words
    #   "negative_keywords": ["quotation date", "rfq"],            ← exclusion words
    #   "priority":          6,                                    ← tiebreaker (higher wins)
    #   "notes":             "Operator notes about this type...",
    #   "ai_observations":   "Auto-generated summary from seen docs"
    # }
    # Populated from CLASSIFICATION_RULES on first seed; thereafter stored here
    # so operators can edit, add, or delete keywords without touching code.

    variants: Mapped[list["DocumentVariant"]] = relationship(
        "DocumentVariant", back_populates="document_class"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="document_class"
    )


class DocumentVariant(Base):
    """
    Tier 2 — a learned extraction template scoped to a specific sender/issuer
    AND layout sub-type within a Document Class.

    Variant identity = issuer_slug + field_fingerprint.
    Two documents from the same issuer with ≥75% field overlap → same variant.
    Same issuer, different field set → different variant (different template).

    This gives a three-level hierarchy:
        DocumentClass  →  Issuer (issuer_slug)  →  Format variant (field_fingerprint)
    """
    __tablename__ = "document_variants"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_class_id: Mapped[str] = mapped_column(
        String, ForeignKey("document_classes.id"), nullable=False
    )

    # Composite identity ──────────────────────────────────────────────────────
    variant_key: Mapped[str] = mapped_column(String, nullable=False)
    # Derived: "{issuer_slug}__{fingerprint_short}" e.g. "tata-steel__rfq_date,rfq_number"
    # Legacy: sender email domain for variants created before issuer-aware keying.

    issuer_slug: Mapped[str | None] = mapped_column(String, nullable=True)
    # Normalised issuer/supplier name: "tata-steel", "ssab-emea", "unknown"

    field_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Sorted CSV of discovered field names: "delivery_date,material_number,rfq_date,rfq_number"
    # Used for format-divergence detection within the same issuer.

    variant_label: Mapped[str | None] = mapped_column(String, nullable=True)
    # Human-readable: "Tata Steel RFQ — Format A", auto-generated on creation

    # Per-variant confirmed schema ─────────────────────────────────────────────
    field_schema_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Operator-confirmed field list for this specific variant.
    # JSON: { "field_name": { "required": bool, "format_hint": str, "example": str } }
    # Takes precedence over class-level DocumentTypeProfile.field_schema_json.

    # SignatureAgent — learned signature location for this variant ────────────
    signature_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: {
    #   "instances_seen":  7,
    #   "confident":       true,        ← True once instances_seen >= CONFIDENT_THRESHOLD (3)
    #   "page_anchor":     "end",       ← "start" or "end" (whichever side of the doc is nearer)
    #   "page_offset":     0,           ← pages from that anchor (0 = first or last page)
    #   "bbox_hint":       [x0,y0,x1,y1],  ← fractional, rolling average of past bboxes
    #   "label_nearby":    "signed by",    ← most-seen signature label in vicinity
    #   "last_updated":    "ISO datetime"
    # }
    # Anchor system: page 0 of 10 → anchor="start", offset=0
    #                page 9 of 10 → anchor="end",   offset=0
    #                page 1 of 10 → anchor="start", offset=1
    # Stable across different page counts for the same document variant.
    # Fast path: when confident=True, resolve page from anchor+offset, check for
    # image+label at bbox_hint, crop evidence — zero API calls.

    # StructuralProfileAgent — learned structural fingerprint ────────────────
    structural_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: {
    #   "instances_seen":   12,
    #   "page_count":       { "mode": 1, "min": 1, "max": 2, "counts": {"1":10,"2":2} },
    #   "filename_patterns": ["PO-\\d+-\\d+"],
    #   "page1_headings":   { "PURCHASE ORDER": 10, "OFFICIAL PURCHASE ORDER": 2 },
    #   "header_lines":     { "CONFIDENTIAL": 3 },
    #   "footer_lines":     { "Tata Steel UK Limited": 11 },
    #   "last_updated":     "2026-06-23T10:00:00"
    # }
    # Populated by StructuralProfileAgent after each processed document.
    # Used by VariantDiscoveryAgent to validate new documents match the expected
    # structural fingerprint (page count, headings, header/footer text).

    # PageProfileAgent — learned page skip map ────────────────────────────────
    page_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: {
    #   "instances_seen":  5,
    #   "page_data": {
    #     "0":  { "present_in": 5, "contributed_in": 5, "fields_seen": ["po_number", "entity"] },
    #     "62": { "present_in": 5, "contributed_in": 5, "fields_seen": ["line_items"] },
    #     "3":  { "present_in": 5, "contributed_in": 0, "fields_seen": [] }
    #   },
    #   "confident_skip":      [3, 4, 5, 6, 7],   ← pages always skipped safely
    #   "last_updated":        "2026-06-22T10:30:00",
    #   "tokens_saved_estimate": 12500,
    #   "cost_saved_usd":      0.038
    # }
    # Populated by PageProfileAgent after each confirmed extraction.
    # Once confident_skip is non-empty, _smart_page_sample excludes those pages entirely.

    # Learning lifecycle ───────────────────────────────────────────────────────
    learning_stage: Mapped[str] = mapped_column(String, default="ZERO_SHOT")
    # ZERO_SHOT | LEARNING | LEARNED_PROPOSED | LEARNED | OPTIMISED

    confirmed_instance_count: Mapped[int] = mapped_column(Integer, default=0)
    # Incremented each time an operator confirms the extraction is correct.

    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    # Total documents seen (including unconfirmed) — drives auto stage advancement.

    avg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    touchless_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    template_frozen: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    document_class: Mapped[DocumentClass] = relationship(
        "DocumentClass", back_populates="variants"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="variant"
    )


class Document(Base):
    """
    Tier 3 — an individual document instance received and processed by Dokr.
    Each instance is tagged to its Document Class, Variant, extraction result,
    and every pipeline event and human correction applied to it.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # Format: doc_<ULID>  e.g. doc_01J8K3PXMQR4T7N

    # Pipeline state
    status: Mapped[str] = mapped_column(String, default="RECEIVED")

    # Classification
    document_class_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("document_classes.id"), nullable=True
    )
    document_class_override: Mapped[str | None] = mapped_column(String, nullable=True)
    variant_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("document_variants.id"), nullable=True
    )
    variant_key: Mapped[str | None] = mapped_column(String, nullable=True)

    # Classification confidence (0.0–1.0): winner_score / class_keyword_count
    # Low values (< 0.35) trigger GovernanceAgent review
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # GovernanceAgent results — populated when GOVERNING state is entered
    # JSON: {verdict, reasoning, suggested_class, suggested_class_name,
    #        suggested_class_description, suggested_keywords, is_new_type}
    ai_governance_result: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Human-readable suggestion for new document type (populated when CANDIDATE_NEW_CLASS)
    suggested_class_name: Mapped[str | None] = mapped_column(String, nullable=True)
    candidate_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Signature detection ───────────────────────────────────────────────────
    # Populated by ExtractionAgent when DocumentTypeProfile.check_signature=True.
    # None = not checked; True/False = checked and result.
    is_signed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    signature_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # JSON: {page: int, bbox: [x0,y0,x1,y1], screenshot_b64: str, method: str}
    # method: "digital" | "visual_text" | "ai_vision"
    signature_evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # File metadata (content never stored here — goes to SharePoint in production)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String, nullable=False)
    # SHA-256 used for exact duplicate detection

    # Content fingerprint for near-duplicate detection (Section 7)
    content_simhash: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # 64-bit SimHash of normalised PDF text. Stored as signed integer.

    # If this document is a near-duplicate or content-duplicate, links to the original
    original_document_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id"), nullable=True
    )

    # If this document was produced by SplittingAgent from a multi-doc PDF,
    # links back to the bundle parent. NULL for all normal (non-split) documents.
    parent_document_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id"), nullable=True
    )
    # Hamming distance to original (for near-duplicate audit trail)
    duplicate_hamming_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Processing options
    priority: Mapped[str] = mapped_column(String, default="standard")
    # standard | express

    # Page sampling metadata — populated by ExtractionAgent ──────────────────
    pages_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Total pages in the PDF (0 if extraction never ran or doc was non-PDF)

    pages_sampled_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON list of 0-indexed page numbers actually passed to the AI.
    # e.g. [0, 1, 2, 60, 61, 62, 63, 64, 65, 66, 67, 68]
    # Used by PageProfileAgent to build per-page contribution maps.

    pages_skipped_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # pages_total - len(pages_sampled_json). Exposed in API response so callers
    # know when information might be missing because pages were skipped.

    # Caller-supplied metadata (arbitrary key-value)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    # Shipment linkage (populated by LinkingAgent)
    shipment_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # References a ShipmentRecord.id — not a FK so shipments can be deleted independently

    # Per-document pipeline gate controls (list of PipelineState names to skip)
    # e.g. ["MATCHING", "POSTING"] for STORE-only docs or caller-overridden routing
    skip_stages: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Integration
    webhook_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Submitter identity (set by Outlook add-in at upload time)
    submitter_email: Mapped[str | None] = mapped_column(String, nullable=True)

    # Failure tracking — set whenever the document transitions to NEEDS_REVIEW
    error_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document_class: Mapped[DocumentClass | None] = relationship(
        "DocumentClass", back_populates="documents"
    )
    variant: Mapped[DocumentVariant | None] = relationship(
        "DocumentVariant", back_populates="documents"
    )
    pipeline_events: Mapped[list["PipelineEvent"]] = relationship(
        "PipelineEvent", back_populates="document", order_by="PipelineEvent.created_at"
    )
    extracted_fields: Mapped[list["ExtractedField"]] = relationship(  # noqa: F821
        "ExtractedField", back_populates="document", order_by="ExtractedField.field_name"
    )


class PipelineEvent(Base):
    """
    Immutable record of every pipeline state transition for a document.
    Forms the audit trail visible in GET /documents/{id}/status.
    """
    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id"), nullable=False
    )
    state: Mapped[str] = mapped_column(String, nullable=False)
    agent: Mapped[str | None] = mapped_column(String, nullable=True)
    # Which agent triggered this transition
    detail: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[Document] = relationship(
        "Document", back_populates="pipeline_events"
    )
