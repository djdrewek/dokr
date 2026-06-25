from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExtractedFieldOut(BaseModel):
    field_name: str
    field_value: str | None
    field_type: str = "scalar"       # "scalar" | "table"
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_model: str
    extraction_method: str
    human_corrected: bool
    corrected_value: str | None = None
    used_in_match: bool
    match_result: str | None = None
    extracted_at: datetime

    class Config:
        from_attributes = True


class TableOut(BaseModel):
    """A named table extracted from a document (e.g. line_items, shipping_charges)."""
    table_name: str
    columns: list[str]               # column names, derived from first row
    rows: list[dict[str, Any]]       # one dict per row
    row_count: int
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str


class DocumentFieldsOut(BaseModel):
    document_id: str
    document_class: str | None
    document_class_name: str | None
    variant_key: str | None
    learning_stage: str | None
    field_count: int
    avg_confidence: float | None
    fields: list[ExtractedFieldOut]
    tables: list[TableOut] = Field(default_factory=list)
    # Structured tables found in the document (line items, charges, packages, etc.)
    # Each table is a named array of row objects. Empty list if no tables were extracted.

    # Page sampling transparency — available once extraction has run
    pages_total: int | None = None
    # Total pages in the source PDF.
    pages_sampled: int | None = None
    # Number of pages whose text was passed to the AI.
    pages_skipped: int | None = None
    # Pages excluded from extraction (pages_total - pages_sampled).
    # Non-zero means some document content was not seen — fields on skipped pages
    # may be absent. This shrinks as PageProfileAgent learns the variant's layout.
    page_profile_stage: str | None = None
    # "none"      — no profile learned yet (first few documents)
    # "learning"  — profile building (< MIN_INSTANCES confirmed)
    # "confident" — skip list active; confirmed dead pages excluded from every run


class FieldCorrectionIn(BaseModel):
    """Body for PATCH /documents/{id}/fields/{field_name}/correct (FR-027f)."""
    corrected_value: str = Field(
        ...,
        description="The human-verified correct value for this field.",
        min_length=1,
    )
    corrected_by: str = Field(
        ...,
        description="Identifier of the reviewer making the correction (e.g. email or user ID).",
        min_length=1,
    )

    @field_validator("corrected_value", "corrected_by")
    @classmethod
    def no_whitespace_only(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("Value must not be whitespace-only.")
        return v.strip()


class FieldCorrectionOut(BaseModel):
    """Response for PATCH /documents/{id}/fields/{field_name}/correct."""
    document_id: str
    field_name: str
    original_value: str | None
    corrected_value: str
    corrected_by: str
    corrected_at: datetime
    variant_id: str | None
    learning_stage_before: str | None
    learning_stage_after: str | None
    confirmed_instance_count: int | None
