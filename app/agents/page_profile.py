"""
PageProfileAgent — learns which pages of a DocumentVariant are always empty.

After each successful extraction the agent:
  1. Attributes every extracted field to the page(s) where its value appears.
  2. Updates a per-variant contribution map: for each page that was sampled,
     did it contribute at least one extracted field this run?
  3. Once MIN_INSTANCES confirmed documents have been processed, pages whose
     contribution_rate is 0 are added to confident_skip.

On the next document of the same variant, ExtractionAgent calls
get_confident_skip() and passes those page indices to _smart_page_sample(),
which excludes them entirely — reducing tokens consumed and API cost.

The profile is stored in DocumentVariant.page_profile_json:
  {
    "instances_seen":  7,
    "page_data": {
      "0":  { "present_in": 7, "contributed_in": 7, "fields_seen": ["po_number"] },
      "3":  { "present_in": 7, "contributed_in": 0, "fields_seen": [] },
      "62": { "present_in": 7, "contributed_in": 7, "fields_seen": ["line_items"] }
    },
    "confident_skip":        [3, 4, 5, 6, 7, 8],
    "last_updated":          "2026-06-22T10:30:00",
    "tokens_saved_estimate": 18000,
    "cost_saved_usd":        0.054
  }
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document, DocumentVariant
from app.models.extracted_field import ExtractedField

if TYPE_CHECKING:
    from app.agents.extraction import PageSampleMeta

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
MIN_INSTANCES       = 3    # minimum confirmed docs before any page is marked skippable
SKIP_ZERO_RATE      = 0.0  # pages with ≤ this contribution rate become candidates
MIN_PRESENT_TO_SKIP = 3    # page must have been sampled at least this many times
# Approximate cost used for the "tokens saved" estimate ($3 / MTok for Sonnet text)
COST_PER_TOKEN      = 3.0 / 1_000_000
AVG_CHARS_PER_PAGE  = 1_800  # heuristic average; 6 chars ≈ 1 token
CHARS_PER_TOKEN     = 6


class PageProfileAgent(BaseAgent):
    """
    Builds and maintains the page contribution profile for each DocumentVariant.

    Primary entry points
    --------------------
    record_and_update(doc, fields, pdf_bytes, page_meta)
        Call this after a successful extraction. Records which pages contributed
        fields for this document and updates the variant's profile.

    get_confident_skip(variant_id) -> list[int]
        Returns the list of 0-indexed page numbers that can be safely skipped
        for the given variant (empty list if no profile exists yet).
    """

    name = "PageProfileAgent"

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_and_update(
        self,
        doc: Document,
        fields: list[ExtractedField],
        pdf_bytes: bytes | None,
        page_meta: "PageSampleMeta",
    ) -> dict:
        """
        Record page attribution for `doc` and refresh the variant's profile.

        Returns the updated profile dict (useful for logging / tests).
        Does nothing and returns {} if the doc has no variant assigned.
        """
        if not doc.variant_id:
            return {}

        variant = (
            self.db.query(DocumentVariant)
            .filter(DocumentVariant.id == doc.variant_id)
            .first()
        )
        if not variant:
            return {}

        pages_sampled: list[int] = page_meta.pages_used
        if not pages_sampled:
            return {}

        # ── Attribute fields to pages ─────────────────────────────────────────
        attribution = self._attribute_fields_to_pages(
            pdf_bytes, pages_sampled, fields
        )

        # ── Load or initialise profile ────────────────────────────────────────
        profile      = json.loads(variant.page_profile_json or "{}")
        instances    = profile.get("instances_seen", 0) + 1
        page_data    = profile.get("page_data", {})

        for page_idx in pages_sampled:
            key = str(page_idx)
            contributed = bool(attribution.get(key))
            if key not in page_data:
                page_data[key] = {
                    "present_in":    0,
                    "contributed_in": 0,
                    "fields_seen":   [],
                }
            page_data[key]["present_in"] += 1
            if contributed:
                page_data[key]["contributed_in"] += 1
                existing = set(page_data[key]["fields_seen"])
                existing.update(attribution[key])
                page_data[key]["fields_seen"] = sorted(existing)

        # ── Compute confident_skip ────────────────────────────────────────────
        confident_skip: list[int] = []
        if instances >= MIN_INSTANCES:
            for page_key, stats in page_data.items():
                if stats["present_in"] < MIN_PRESENT_TO_SKIP:
                    continue
                rate = stats["contributed_in"] / stats["present_in"]
                if rate <= SKIP_ZERO_RATE:
                    confident_skip.append(int(page_key))

        confident_skip.sort()

        # ── Estimate tokens (and cost) saved by the skip list ─────────────────
        # Each skipped page × avg chars / chars_per_token × cost_per_token × instances
        tokens_per_page      = AVG_CHARS_PER_PAGE / CHARS_PER_TOKEN
        tokens_saved         = int(len(confident_skip) * tokens_per_page * instances)
        cost_saved_usd       = round(tokens_saved * COST_PER_TOKEN, 4)

        # ── Persist ───────────────────────────────────────────────────────────
        updated_profile = {
            "instances_seen":        instances,
            "page_data":             page_data,
            "confident_skip":        confident_skip,
            "last_updated":          datetime.utcnow().isoformat(),
            "tokens_saved_estimate": tokens_saved,
            "cost_saved_usd":        cost_saved_usd,
        }
        variant.page_profile_json = json.dumps(updated_profile)
        self.db.commit()

        logger.info(
            "PageProfileAgent: variant=%s instances=%d confident_skip=%s "
            "est_tokens_saved=%d cost_saved=$%.4f",
            doc.variant_id, instances, confident_skip, tokens_saved, cost_saved_usd,
        )
        return updated_profile

    def get_confident_skip(self, variant_id: str) -> list[int]:
        """
        Return the list of page indices that can be safely skipped for this variant.
        Returns [] if no profile exists or fewer than MIN_INSTANCES have been seen.
        """
        variant = (
            self.db.query(DocumentVariant)
            .filter(DocumentVariant.id == variant_id)
            .first()
        )
        if not variant or not variant.page_profile_json:
            return []
        try:
            profile = json.loads(variant.page_profile_json)
            return profile.get("confident_skip", [])
        except Exception:
            return []

    def get_profile_stage(self, variant_id: str | None) -> str:
        """
        Return a human-readable stage label for API consumers:
          "none"      — no profile yet
          "learning"  — profile exists but skip list is empty (too few instances)
          "confident" — skip list active
        """
        if not variant_id:
            return "none"
        skip = self.get_confident_skip(variant_id)
        variant = (
            self.db.query(DocumentVariant)
            .filter(DocumentVariant.id == variant_id)
            .first()
        )
        if not variant or not variant.page_profile_json:
            return "none"
        try:
            profile = json.loads(variant.page_profile_json)
            if profile.get("instances_seen", 0) < MIN_INSTANCES:
                return "learning"
            return "confident" if skip else "learning"
        except Exception:
            return "none"

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _attribute_fields_to_pages(
        self,
        pdf_bytes: bytes | None,
        pages_sampled: list[int],
        fields: list[ExtractedField],
    ) -> dict[str, list[str]]:
        """
        For each sampled page, find which extracted field values appear in its text.

        Uses exact-substring matching — sufficient for reference numbers, dates,
        amounts, and party names. Values under 4 chars are skipped to avoid
        false positives (e.g. currency codes appearing on every page).

        Returns { "page_idx_str": [field_name, ...] }.
        If pdf_bytes is unavailable, falls back to "all fields on all pages."
        """
        if not pdf_bytes or not fields:
            # Fallback: attribute all fields to all sampled pages
            names = [f.field_name for f in fields if f.field_type != "table"]
            return {str(p): names for p in pages_sampled}

        # Extract per-page text
        per_page_text: dict[int, str] = {}
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            for idx in pages_sampled:
                if 0 <= idx < len(reader.pages):
                    t = reader.pages[idx].extract_text() or ""
                    per_page_text[idx] = t.lower()
        except Exception as exc:
            logger.debug("PageProfileAgent: page text extraction failed: %s", exc)
            names = [f.field_name for f in fields if f.field_type != "table"]
            return {str(p): names for p in pages_sampled}

        attribution: dict[str, list[str]] = {str(p): [] for p in pages_sampled}

        scalar_fields = [
            f for f in fields
            if f.field_type != "table"
            and f.field_value
            and len(f.field_value.strip()) >= 4
        ]

        for ef in scalar_fields:
            val_lower = ef.field_value.strip().lower()
            # Search each sampled page; attribute to the first page that contains the value
            for page_idx in sorted(pages_sampled):
                page_text = per_page_text.get(page_idx, "")
                if page_text and val_lower in page_text:
                    attribution[str(page_idx)].append(ef.field_name)
                    break  # each field attributed to its first occurrence page

        # Table fields: search first row's values (a table almost certainly spans
        # its source page, so any column value match is good enough)
        table_fields = [
            f for f in fields
            if f.field_type == "table" and f.field_value
        ]
        for ef in table_fields:
            try:
                rows = json.loads(ef.field_value)
                if not rows:
                    continue
                # Check first row's values
                first_row_vals = [
                    str(v).strip().lower()
                    for v in rows[0].values()
                    if v and len(str(v).strip()) >= 4
                ]
                for page_idx in sorted(pages_sampled):
                    page_text = per_page_text.get(page_idx, "")
                    if page_text and any(v in page_text for v in first_row_vals):
                        attribution[str(page_idx)].append(ef.field_name)
                        break
            except Exception:
                pass

        return attribution
