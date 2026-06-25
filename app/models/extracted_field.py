from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ExtractedField(Base):
    """
    Per-field extraction result for a Document Instance.
    Each field carries its value, confidence score, extraction model,
    and full provenance — satisfying FR-027f (field-level lineage).
    """
    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id"), nullable=False
    )

    # Field identity
    field_name: Mapped[str] = mapped_column(String, nullable=False)
    field_value: Mapped[str | None] = mapped_column(String, nullable=True)

    # Field type — controls how field_value is interpreted
    field_type: Mapped[str] = mapped_column(String, default="scalar")
    # "scalar" — a single string value (default for all header/summary fields)
    # "table"  — field_value is a JSON array of row dicts, e.g. line items,
    #             shipping charges, packages. The AI names the table freely
    #             (e.g. "line_items", "shipping_charges", "hazmat_entries").

    # Extraction provenance (FR-027f)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    extraction_model: Mapped[str] = mapped_column(String, nullable=False)
    # e.g. "regex-tier1", "regex-tier2-ocr", "claude-sonnet-4-6-vision"
    extraction_method: Mapped[str] = mapped_column(String, nullable=False)
    # TEXT_LAYER | OCR | AI_VISION
    extraction_tier: Mapped[int] = mapped_column(Integer, default=1)
    # 1 = PDF text layer · 2 = OCR · 3 = AI vision API

    # Human correction (FR-UEE-004)
    human_corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    corrected_value: Mapped[str | None] = mapped_column(String, nullable=True)
    corrected_by: Mapped[str | None] = mapped_column(String, nullable=True)
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Match check outcome (FR-027f)
    used_in_match: Mapped[bool] = mapped_column(Boolean, default=False)
    match_result: Mapped[str | None] = mapped_column(String, nullable=True)
    # PASS | WITHIN_TOLERANCE | FAIL | None

    # ── Spatial provenance — where on the page was this field found? ──────────
    # Populated during extraction; enables the viewer to jump straight to the
    # right page and draw an exact highlight without a live text search.
    extraction_page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 0-indexed page number within the PDF.

    extraction_bbox_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON [x0, y0, x1, y1] as FRACTIONS of page dimensions (0.0–1.0).
    # e.g. [0.12, 0.34, 0.55, 0.38]
    # Stored as fractions so they are DPI-independent; the viewer scales them.

    # ── AddressAgent — structured address parse result ────────────────────────
    address_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Populated by AddressAgent for fields identified as address blocks.
    # JSON: {
    #   "name":          "Refteck Solutions Limited",
    #   "line1":         "43 Brighton Road",
    #   "line2":         null,
    #   "city":          "Coulsdon",
    #   "state":         "Surrey",
    #   "postcode":      "CR5 1NL",
    #   "country":       "United Kingdom",
    #   "country_code":  "GB",
    #   "raw":           "Refteck Solutions Limited\n43\nCoulsdon...",
    #   "verified":      true,
    #   "verification_source": "postcodes.io",  — "postcodes.io" | "google_maps" | null
    #   "lat":           51.3207,
    #   "lng":           -0.1395,
    #   "admin_district": "Croydon",
    #   "parse_error":   null
    # }

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="extracted_fields"
    )
