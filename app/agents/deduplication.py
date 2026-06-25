"""
Deduplication Agent — Section 7 of the PRD.

Three-tier duplicate detection:

  Tier 1 — Exact duplicate (SHA-256 match)
    Already handled in POST /submit before the pipeline starts.
    If a document reaches this agent with the same SHA-256 as an existing
    record, it is marked EXACT_DUPLICATE and processing stops.

  Tier 2 — Content duplicate (SimHash Hamming distance ≤ 3)
    Visually identical or trivially reformatted document (e.g. a PDF
    re-exported with different metadata but identical content).
    Marked CONTENT_DUPLICATE. The original document record is linked.

  Tier 3 — Near duplicate (SimHash Hamming distance ≤ 10)
    Same document with minor field changes — an amendment, revision,
    or re-send. Marked NEAR_DUPLICATE. Both the original and the diff
    of extracted field values are surfaced for human review.

SimHash comparison is done against all documents in the same Document Class
with a COMPLETED or NEEDS_REVIEW status (i.e. previously processed documents).
This bounds the search set to semantically comparable documents and avoids
false positives across classes (a TML PO and a DHL AWB will differ structurally).

Production note: replace the linear scan with an LSH (Locality-Sensitive Hashing)
index stored in Redis or a vector DB for sub-millisecond lookup at scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import pypdf
import io

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.pipeline.states import PipelineState
from app.utils.simhash import (
    CONTENT_DUPLICATE_THRESHOLD,
    NEAR_DUPLICATE_THRESHOLD,
    classify_similarity,
    hamming_distance,
    simhash,
    normalise_for_hashing,
)


@dataclass
class DeduplicationResult:
    is_duplicate: bool
    duplicate_type: str | None  # EXACT_DUPLICATE | CONTENT_DUPLICATE | NEAR_DUPLICATE | None
    original_document_id: str | None
    hamming_distance: int | None
    fingerprint: int         # computed fingerprint for this document
    field_diff: list[dict] | None  # for NEAR_DUPLICATE: list of {field, original, incoming}


class DeduplicationAgent(BaseAgent):
    """
    Runs SimHash-based content deduplication for a document after classification.

    Called by the pipeline runner between RECEIVED and CLASSIFYING.
    Computes the SimHash of the PDF text, stores it on the document record,
    and compares against previously processed documents in the same class.
    """

    name = "DeduplicationAgent"

    def deduplicate(self, doc: Document, pdf_bytes: bytes) -> DeduplicationResult:
        # Step 1: extract and normalise PDF text for hashing
        text = _extract_text(pdf_bytes)
        fingerprint = simhash(text)

        # Step 2: store fingerprint on this document (enables future comparisons)
        doc.content_simhash = _to_signed64(fingerprint)

        # Step 3: scan existing documents in same class for near-duplicates
        if not doc.document_class_id:
            # Can't scope the search without a class — skip dedup, proceed
            return DeduplicationResult(
                is_duplicate=False,
                duplicate_type=None,
                original_document_id=None,
                hamming_distance=None,
                fingerprint=fingerprint,
                field_diff=None,
            )

        terminal_statuses = [
            PipelineState.COMPLETED,
            PipelineState.NEEDS_REVIEW,
            PipelineState.CONTENT_DUPLICATE,
            PipelineState.NEAR_DUPLICATE,
        ]

        candidates = (
            self.db.query(Document)
            .filter(
                Document.document_class_id == doc.document_class_id,
                Document.id != doc.id,
                Document.content_simhash.isnot(None),
                Document.status.in_(terminal_statuses),
            )
            .all()
        )

        best_distance: int | None = None
        best_match: Document | None = None

        for candidate in candidates:
            candidate_fp = _from_signed64(candidate.content_simhash)
            dist = hamming_distance(fingerprint, candidate_fp)

            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_match = candidate

        # Step 4: classify
        if best_distance is None or best_match is None:
            return DeduplicationResult(
                is_duplicate=False,
                duplicate_type=None,
                original_document_id=None,
                hamming_distance=None,
                fingerprint=fingerprint,
                field_diff=None,
            )

        dup_type = classify_similarity(best_distance)
        if dup_type is None:
            return DeduplicationResult(
                is_duplicate=False,
                duplicate_type=None,
                original_document_id=None,
                hamming_distance=best_distance,
                fingerprint=fingerprint,
                field_diff=None,
            )

        # Step 5: for NEAR_DUPLICATE, build a field-level diff
        field_diff = None
        if dup_type == "NEAR_DUPLICATE":
            field_diff = _build_field_diff(doc, best_match, self.db)

        # Record duplicate linkage on the document
        doc.original_document_id = best_match.id
        doc.duplicate_hamming_distance = best_distance

        return DeduplicationResult(
            is_duplicate=True,
            duplicate_type=dup_type,
            original_document_id=best_match.id,
            hamming_distance=best_distance,
            field_diff=field_diff,
            fingerprint=fingerprint,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF and normalise for SimHash computation."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return normalise_for_hashing(" ".join(parts))
    except Exception:
        return ""


def _build_field_diff(
    incoming: Document,
    original: Document,
    db: Session,
) -> list[dict]:
    """
    Compare extracted fields between the incoming document and its original.
    Returns a list of {field_name, original_value, incoming_value} for fields
    that differ. Fields present only in one document are included with None
    for the absent side.
    """
    from app.models.extracted_field import ExtractedField

    def field_map(doc: Document) -> dict[str, str]:
        fields = db.query(ExtractedField).filter(ExtractedField.document_id == doc.id).all()
        return {
            f.field_name: (f.corrected_value if f.human_corrected else f.field_value)
            for f in fields
        }

    orig_map = field_map(original)
    inc_map = field_map(incoming)
    all_fields = set(orig_map) | set(inc_map)

    diffs = []
    for field_name in sorted(all_fields):
        orig_val = orig_map.get(field_name)
        inc_val = inc_map.get(field_name)
        if orig_val != inc_val:
            diffs.append({
                "field_name": field_name,
                "original_value": orig_val,
                "incoming_value": inc_val,
            })

    return diffs


def _to_signed64(value: int) -> int:
    """Convert unsigned 64-bit int to signed (SQLite stores BigInteger as signed)."""
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def _from_signed64(value: int) -> int:
    """Convert signed 64-bit int back to unsigned for Hamming comparison."""
    if value < 0:
        return value + (1 << 64)
    return value
