"""
VariantDiscoveryAgent — post-extraction variant identification.

Three-level hierarchy:
    DocumentClass  (dc_015 = RFQ)
      └── Issuer  (tata-steel, ssab-emea, unknown)
            └── Format variant  (field set A, field set B)

Variant identity  =  issuer_slug  +  field_fingerprint
Two docs from the same issuer with ≥OVERLAP_THRESHOLD field overlap → same variant.
Same issuer, different field set → new variant (different template in use).

This agent runs AFTER extraction so it can read the extracted fields to:
  a) identify the issuer (from supplier_name / issuer / entity fields)
  b) compute the field fingerprint (sorted CSV of extracted field names)
  c) match against existing variants or create a new one

Learning stage advancement (per variant):
  doc_count ≥  3  → LEARNING         (seen enough to start generalising)
  doc_count ≥ 10  → LEARNED_PROPOSED (auto-propose schema for operator review)
  operator confirms → LEARNED         (schema locked)
  doc_count ≥ 50  → OPTIMISED        (generate fast regex patterns)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document, DocumentVariant
from app.utils.ids import generate_variant_id

logger = logging.getLogger(__name__)

# ── Learning-stage thresholds ──────────────────────────────────────────────────
LEARNING_THRESHOLD  = 3      # docs seen → LEARNING
PROPOSED_THRESHOLD  = 10     # docs seen → LEARNED_PROPOSED
OPTIMISED_THRESHOLD = 50     # docs seen → OPTIMISED

# ── Multi-layer matching thresholds ────────────────────────────────────────────
#
# Variant identity is decided by a four-layer cascade.  Each layer provides an
# independent signal; the final accept/reject weighs all of them:
#
#   Layer 1 — AI-extracted field fingerprint (double-filtered: no line-item numbers,
#             no ALL-CAPS content sub-keys).  Stable on clean text-layer PDFs.
#             Unreliable on image-only PDFs where OCR/vision may yield sparse fields.
#
#   Layer 2 — Raw PDF label fingerprint: PyMuPDF colon-label scan on the raw text
#             layer.  Independent of the AI extractor.  Works on any PDF that has
#             a text layer (even if the AI struggles).  Stored per-variant in
#             structural_profile_json["raw_label_fp"].
#
#   Layer 3 — Structural profile: page count (±1), page-1 headings, footer text.
#             Already tracked by StructuralProfileAgent.  Cheap boolean check.
#
#   Layer 4 — Haiku AI confirm, with optional page-1 image for image-heavy PDFs.
#             Invoked when Layer 1+2 signals are borderline, Layer 3 fails,
#             OR when the incoming doc is image-only (layers 1+2 may be empty).
#
# Decision table (after issuer guard):
#   strong field OR strong raw   + structural ok  →  ACCEPT  (no AI call)
#   weak field   AND weak raw    + structural ok  →  ACCEPT  (corroborated)
#   anything weaker / struct fail / image PDF     →  Layer-4 AI confirm
#   AI confirm YES                               →  ACCEPT
#   AI confirm NO (or no match found at all)     →  create new variant
#
_STRONG_FIELD_THRESH = 0.85   # Layer 1 strong — confident field match
_WEAK_FIELD_THRESH   = 0.60   # Layer 1 weak   — plausible, needs corroboration
_STRONG_RAW_THRESH   = 0.75   # Layer 2 strong — confident raw-label match
_WEAK_RAW_THRESH     = 0.50   # Layer 2 weak   — plausible, needs corroboration

# Fields used to identify the TEMPLATE OWNER (the company whose format this is).
# Checked against both verbatim labels and snake_case (for backwards compatibility
# with docs processed before the verbatim-name change).
_ISSUER_FIELD_PRIORITY = [
    # Verbatim label forms
    "Issuer", "Issuing Company", "Buyer", "Buyer Name", "Purchaser",
    "Purchaser Name", "From", "From Company", "Company", "Company Name", "Entity",
    # snake_case fallbacks (older documents)
    "issuer", "issuing_company", "buyer_name", "purchaser_name",
    "from_company", "company_name", "entity",
]


class VariantDiscoveryAgent(BaseAgent):
    """
    Identifies or creates the DocumentVariant for a document.

    Call AFTER extraction. Reads doc.extracted_fields to determine issuer
    and field fingerprint, then matches against existing variants.
    """

    name = "VariantDiscoveryAgent"

    def discover(self, doc: Document, pdf_bytes: bytes | None = None) -> str | None:
        """
        Find or create the variant for this document.

        Four-layer cascade:
          L1 — AI-extracted field fingerprint (double-filtered)
          L2 — Raw PDF label fingerprint (PyMuPDF colon-label scan, stored per-variant)
          L3 — Structural profile check (page count, headings)
          L4 — Haiku AI confirm with optional page-1 image (borderline / image PDFs)

        Returns the variant ID (sets doc.variant_key externally).
        """
        if not doc.document_class_id:
            return None

        # ── L1: AI field fingerprint ───────────────────────────────────────────
        issuer_slug     = self._extract_issuer_slug(doc)
        fingerprint     = self._compute_fingerprint(doc)
        incoming_fields = set(fingerprint.split(",")) if fingerprint else set()

        # ── L2: Raw PDF label fingerprint (PyMuPDF, no AI) ────────────────────
        # Works on any text-layer PDF, independent of extraction quality.
        raw_labels: frozenset[str] = frozenset()
        if pdf_bytes:
            raw_labels = _compute_raw_label_fingerprint(pdf_bytes)

        # Is this effectively an image-only PDF?
        # (raw_labels will be empty when there's no meaningful text layer)
        is_image_doc = (
            getattr(doc, "extraction_tier", None) in ("OCR", "AI_VISION")
            or (pdf_bytes is not None and len(raw_labels) < 3)
        )

        # ── Candidate scan ─────────────────────────────────────────────────────
        candidates = (
            self.db.query(DocumentVariant)
            .filter(
                DocumentVariant.document_class_id == doc.document_class_id,
                DocumentVariant.active.is_(True),
            )
            .all()
        )

        # Use -1.0 so that even a zero-signal image-doc candidate (score=0.0) wins
        best_match   = None
        best_score   = -1.0
        best_field_j = 0.0
        best_raw_j   = 0.0

        for candidate in candidates:
            # Issuer guard — skip cross-company comparisons when both are known
            cand_issuer = (candidate.issuer_slug or "unknown").strip()
            if (
                issuer_slug != "unknown"
                and cand_issuer != "unknown"
                and issuer_slug != cand_issuer
            ):
                continue

            # L1 — AI field Jaccard
            field_j = 0.0
            if candidate.field_fingerprint:
                known = set(candidate.field_fingerprint.split(","))
                if incoming_fields or known:
                    field_j = len(incoming_fields & known) / len(incoming_fields | known)
            elif not incoming_fields:
                field_j = 1.0   # both empty → treat as identical

            # L2 — Raw label Jaccard (stored in structural_profile_json)
            raw_j = 0.0
            try:
                cprof = json.loads(candidate.structural_profile_json or "{}")
                craw  = frozenset(cprof.get("raw_label_fp") or [])
            except Exception:
                craw  = frozenset()
                cprof = {}
            if raw_labels and craw:
                raw_j = len(raw_labels & craw) / len(raw_labels | craw)

            # For non-image docs: require at least one weak signal before proceeding.
            # For image-only docs: both signals will be 0 — we still want to forward
            # this candidate to L4 AI confirm (with the page-1 image) rather than
            # quietly creating a duplicate variant.  Use page-count proximity as a
            # minimum plausibility filter (skip if variant expects very different count).
            if field_j < _WEAK_FIELD_THRESH and raw_j < _WEAK_RAW_THRESH:
                if not is_image_doc:
                    continue
                # Image doc plausibility check: page count within ±3 of variant mode
                pc_mode = (cprof.get("page_count") or {}).get("mode")
                doc_pages = doc.pages_total or 0
                if pc_mode is not None and doc_pages > 0 and abs(doc_pages - pc_mode) > 3:
                    continue
                # Passes plausibility — keep with score 0.0 for L4 to decide

            # Use the stronger signal to rank candidates; 0.0 for image-doc fallbacks
            combined = max((x for x in [field_j, raw_j] if x > 0), default=0.0)
            if combined > best_score:
                best_match   = candidate
                best_score   = combined
                best_field_j = field_j
                best_raw_j   = raw_j

        # ── L3 + L4 decision on best candidate ────────────────────────────────
        if best_match:
            field_j = best_field_j
            raw_j   = best_raw_j

            is_strong       = field_j >= _STRONG_FIELD_THRESH or raw_j >= _STRONG_RAW_THRESH
            is_corroborated = field_j >= _WEAK_FIELD_THRESH   and raw_j >= _WEAK_RAW_THRESH

            # L3 — structural profile check
            structural_ok = self._check_structural(doc, best_match)

            # Decide whether L4 AI confirm is needed
            needs_confirm = (
                not (is_strong or is_corroborated)  # signals too weak to trust alone
                or not structural_ok                 # structural mismatch (page count etc.)
                or is_image_doc                      # image PDF — signals unreliable
            )

            if needs_confirm:
                logger.info(
                    "VariantDiscoveryAgent: requesting L4 AI confirm for variant %s "
                    "(field_j=%.2f raw_j=%.2f strong=%s corr=%s struct_ok=%s image=%s)",
                    best_match.id, field_j, raw_j,
                    is_strong, is_corroborated, structural_ok, is_image_doc,
                )
                confirmed = self._zero_shot_confirm(doc, best_match, pdf_bytes=pdf_bytes)
                if not confirmed:
                    logger.info(
                        "VariantDiscoveryAgent: L4 rejected variant %s for doc %s "
                        "— creating new variant",
                        best_match.id, doc.id,
                    )
                    best_match = None

        # ── Accept match ───────────────────────────────────────────────────────
        if best_match:
            best_match.doc_count      = (best_match.doc_count or 0) + 1
            best_match.updated_at     = datetime.utcnow()
            best_match.learning_stage = _compute_stage(best_match)
            # Enrich stored field fingerprint if incoming is richer
            if incoming_fields and len(incoming_fields) > len(
                set((best_match.field_fingerprint or "").split(","))
            ):
                best_match.field_fingerprint = fingerprint
            # Accumulate raw label fingerprint on the variant
            if raw_labels:
                _update_raw_label_fp(best_match, raw_labels)
            self.db.commit()
            doc.variant_key = best_match.variant_key
            return best_match.id

        # ── Create new variant ─────────────────────────────────────────────────
        variant_key   = _build_variant_key(issuer_slug, fingerprint)
        variant_label = _build_variant_label(doc, issuer_slug, candidates)

        new_variant = DocumentVariant(
            id=generate_variant_id(),
            document_class_id=doc.document_class_id,
            variant_key=variant_key,
            issuer_slug=issuer_slug,
            field_fingerprint=fingerprint,
            variant_label=variant_label,
            learning_stage="ZERO_SHOT",
            doc_count=1,
            confirmed_instance_count=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.db.add(new_variant)
        self.db.flush()   # obtain ID before storing raw labels
        if raw_labels:
            _update_raw_label_fp(new_variant, raw_labels)
        self.db.commit()

        doc.variant_key = variant_key
        return new_variant.id

    # ── Issuer identification ──────────────────────────────────────────────────

    def _extract_issuer_slug(self, doc: Document) -> str:
        """
        Find the issuer name from extracted fields, in priority order.
        Checks both verbatim labels (e.g. "Buyer Name") and snake_case fallbacks
        (for backwards compatibility with docs processed before verbatim naming).
        Returns a normalised slug: "tata-steel", "ssab-emea", "unknown".
        """
        field_map: dict[str, str] = {
            ef.field_name.strip(): (ef.field_value or "")
            for ef in (doc.extracted_fields or [])
            if ef.field_name
        }

        for field in _ISSUER_FIELD_PRIORITY:
            value = field_map.get(field, "").strip()
            if value and len(value) > 2:
                slug = _slugify(value)
                # Avoid using the buyer (Tata Steel) as the issuer on RFQs —
                # if entity == issuer that's the buyer not the supplier; skip.
                if slug not in ("tata-steel", "tata-steel-limited", "tata-steel-uk"):
                    return slug[:80]
        return "unknown"

    # ── Field fingerprint ──────────────────────────────────────────────────────

    def _compute_fingerprint(self, doc: Document) -> str:
        """
        Sorted CSV of verbatim field names that have a value, excluding:

        1. Line-item row fields ("Item 1 Description", "Line 2 Qty") — they vary
           in count between document instances.

        2. ALL-CAPS content sub-keys ("PLANT NAME", "ITEM TYPE", "MAKE",
           "PART NO/DRAWING NO") — these are embedded key:value pairs extracted
           from inside text blocks (e.g. the Long Text section of a Tata Steel
           RFQ). They vary by item type within the same template form, not by
           template, and must not pollute the structural fingerprint.

        Only stable, mixed-case template labels remain after these two filters.
        These are identical across all instances of the same form and give a
        reliable Jaccard signal for template matching.

        e.g. fields  → ["Delivery Date", "Item 1 Qty", "PLANT NAME", "Total Amount"]
             fingerprint → "Delivery Date,Total Amount"
        """
        names = sorted(
            ef.field_name.strip()
            for ef in (doc.extracted_fields or [])
            if ef.field_name and ef.field_name.strip() and ef.field_value
            and not _is_line_item_field(ef.field_name.strip())
            and not _is_content_subkey(ef.field_name.strip())
        )
        return ",".join(names)

    # ── Structural profile checks ──────────────────────────────────────────────

    def _check_structural(self, doc: Document, variant: DocumentVariant) -> bool:
        """
        Returns True if doc's structural features are consistent with the variant's
        learned profile.  Returns True (pass) when the profile is immature (< 3 instances)
        so we never block matching on insufficient data.

        Current checks:
          • Page count: doc.pages_total must be within ±1 of the variant's mode.
          • (More checks — headings, footer — will be added as the profile matures.)
        """
        try:
            from app.agents.structural_profile import StructuralProfileAgent
            expected = StructuralProfileAgent.get_expected_page_count(variant)
        except Exception:
            return True   # Can't import or read profile — don't block

        if expected is None:
            return True   # Profile not mature enough yet

        doc_pages = doc.pages_total or 0
        if doc_pages == 0:
            return True   # Page count not recorded — can't check

        if abs(doc_pages - expected) > 1:
            return False  # Page count is off by more than 1 — structural mismatch

        return True

    def _zero_shot_confirm(
        self,
        doc: Document,
        variant: DocumentVariant,
        pdf_bytes: bytes | None = None,
    ) -> bool:
        """
        Layer-4 AI confirmation — ask Haiku whether this document belongs to the
        same template as the candidate variant.

        When pdf_bytes is provided, page 1 of the incoming document is rendered and
        sent as an image so the model can reason about visual layout.  This is the
        primary signal for image-only PDFs where text extraction is unreliable.

        Returns True (accept) or False (reject).
        Defaults to True on any API / render error so the pipeline is never blocked.
        """
        try:
            profile = json.loads(variant.structural_profile_json or "{}")
            pc_info = profile.get("page_count", {})

            variant_fields = sorted(set((variant.field_fingerprint or "").split(",")) - {""})
            doc_fields     = sorted(
                ef.field_name.strip()
                for ef in (doc.extracted_fields or [])
                if ef.field_name and ef.field_name.strip()
                and not _is_line_item_field(ef.field_name.strip())
                and not _is_content_subkey(ef.field_name.strip())
            )
            missing  = sorted(set(variant_fields) - set(doc_fields))
            extra    = sorted(set(doc_fields) - set(variant_fields))

            expected_mode = pc_info.get("mode", "unknown")
            doc_pages     = doc.pages_total or 0
            known_hdgs    = list((profile.get("page1_headings") or {}).keys())[:5]
            known_ftrs    = list((profile.get("footer_lines") or {}).keys())[:3]
            known_raw     = sorted(profile.get("raw_label_fp") or [])[:15]

            text_prompt = (
                "Determine if a new document belongs to the same printed template form "
                "as a known variant. Focus on STRUCTURAL layout — the form labels, "
                "section headings, and page arrangement — not the specific field values.\n\n"
                f"Known template:\n"
                f"  Typical page count : {expected_mode}\n"
                f"  Page-1 headings    : {known_hdgs or 'not yet learned'}\n"
                f"  Footer text        : {known_ftrs or 'not yet learned'}\n"
                f"  Form labels        : {known_raw or variant_fields[:15] or 'not yet learned'}\n\n"
                f"New document:\n"
                f"  Page count         : {doc_pages}\n"
                f"  Filename           : {doc.file_name or 'unknown'}\n"
                f"  Fields vs template — missing: {missing[:8] or 'none'}\n"
                f"  Fields vs template — extra  : {extra[:8] or 'none'}\n\n"
            )

            # Build message content — optionally add page-1 image of the incoming doc
            content: list[dict] = []
            used_vision = False
            if pdf_bytes:
                try:
                    import base64 as _b64
                    import fitz as _fitz
                    fdoc = _fitz.open(stream=pdf_bytes, filetype="pdf")
                    page = fdoc[0]
                    # 1× scale (72 DPI) — compact; enough to read labels & layout
                    pix  = page.get_pixmap(matrix=_fitz.Matrix(1.0, 1.0), alpha=False)
                    img  = pix.tobytes("jpeg")
                    fdoc.close()
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _b64.standard_b64encode(img).decode(),
                        },
                    })
                    text_prompt = (
                        "Page 1 of the new document is shown above.\n\n" + text_prompt
                    )
                    used_vision = True
                except Exception as img_exc:
                    logger.debug(
                        "VariantDiscoveryAgent: could not render page-1 image: %s", img_exc
                    )

            text_prompt += (
                "Based on the structural evidence"
                + (" and the page image above" if used_vision else "")
                + ", does this document use the SAME printed form / template as the known variant?\n"
                "Answer YES (same template) or NO (different template). One word only."
            )
            content.append({"type": "text", "text": text_prompt})

            from anthropic import Anthropic
            client = Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": content}],
            )
            answer = msg.content[0].text.strip().upper()
            logger.info(
                "VariantDiscoveryAgent L4 confirm: variant=%s doc_pages=%s → %s%s",
                variant.id, doc_pages, answer,
                " [vision]" if used_vision else " [text-only]",
            )
            return answer.startswith("Y")

        except Exception as exc:
            logger.warning(
                "VariantDiscoveryAgent: L4 confirm failed for variant %s: %s",
                variant.id, exc,
            )
            return True   # Default: accept — never block the pipeline on an API error

    # ── Schema confirmation (called from confirm-fields endpoint) ──────────────

    def confirm_schema(self, variant_id: str, schema: dict) -> None:
        """
        Write operator-confirmed field schema to this variant.
        Advances learning stage to LEARNED immediately.
        """
        variant = (
            self.db.query(DocumentVariant)
            .filter(DocumentVariant.id == variant_id)
            .first()
        )
        if not variant:
            return
        variant.field_schema_json = json.dumps(schema)
        variant.learning_stage    = "LEARNED"
        variant.updated_at        = datetime.utcnow()
        self.db.commit()

    def increment_confirmed_instance(self, variant_id: str) -> None:
        """Called when a human confirms an extraction as correct."""
        variant = (
            self.db.query(DocumentVariant)
            .filter(DocumentVariant.id == variant_id)
            .first()
        )
        if not variant:
            return
        variant.confirmed_instance_count = (variant.confirmed_instance_count or 0) + 1
        variant.updated_at    = datetime.utcnow()
        variant.learning_stage = _compute_stage(variant)
        self.db.commit()


# ── Helpers ────────────────────────────────────────────────────────────────────

# Detects isolated integers in a field name — the signal for a line-item row field.
# "Item 1 Description" → match  (line-item row, exclude from fingerprint)
# "Line 12 Part No"    → match  (line-item row, exclude from fingerprint)
# "Item 3"             → match  (line-item row, exclude from fingerprint)
# "B2B Reference"      → no match (digit embedded in word token, keep)
# "VAT19 Code"         → no match (digit embedded in word token, keep)
# "H2 Grade"           → no match (digit embedded in word token, keep)
_LINE_ITEM_NUMBER_RE = re.compile(r"(?<!\w)\d+(?!\w)")


def _is_line_item_field(name: str) -> bool:
    """Return True if this field name looks like a numbered line-item row.

    Such fields vary in count between document instances and must be excluded
    from the variant fingerprint.  Field names are never modified — this is
    purely a filter decision.
    """
    return bool(_LINE_ITEM_NUMBER_RE.search(name))


# Matches strings whose word-tokens are ALL uppercase (digits/short tokens excluded).
# e.g. "PLANT NAME" → True, "ITEM TYPE" → True, "PART NO/DRAWING NO" → True
#      "Material Number" → False, "Drawing No." → False, "RFQ No." → False
_ALL_CAPS_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")   # find real word-tokens


def _is_content_subkey(name: str) -> bool:
    """Return True if this field name is an ALL-CAPS content sub-key.

    ALL-CAPS field names (e.g. "PLANT NAME", "ITEM TYPE", "MAKE",
    "PART NO/DRAWING NO", "SPECIAL FEATURES") are typically embedded key:value
    pairs extracted from within free-text blocks (like the Long Text section of
    a Tata Steel RFQ).  They reflect the CONTENT of a particular item, not the
    STRUCTURE of the template form, so they vary across different instances of
    the same template and must be excluded from the fingerprint.

    Mixed-case template labels ("Material Number", "RFQ No.", "Drawing No.")
    always return False and are kept.
    """
    tokens = _ALL_CAPS_TOKEN_RE.findall(name)
    if not tokens:
        return False           # no real word tokens found — keep it
    return all(t == t.upper() for t in tokens)


def _compute_raw_label_fingerprint(pdf_bytes: bytes) -> frozenset[str]:
    """
    Extract structural form labels directly from the PDF text layer via PyMuPDF.
    No AI — scans for lines of the form 'Label Text :' or 'Label Text :-'.

    Returns a frozenset of lowercased, normalised label strings.  This fingerprint
    is independent of AI extraction quality and works on any PDF that has a text
    layer.  On fully-image PDFs with no text layer it returns an empty frozenset.

    Automatically excludes:
      • ALL-CAPS labels (content sub-keys like 'ITEM NAME', 'PLANT NAME')
      • Labels longer than 7 words (sentences / prose, not a label)
      • Purely numeric strings
    """
    try:
        import fitz as _fitz
        labels: set[str] = set()
        doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for page in doc:
                text = page.get_text("text")
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Match "Label Text :" or "Label Text :-" with optional trailing value
                    m = (
                        re.match(r"^(.{2,60}?)\s*:-?\s*$", line)
                        or re.match(r"^(.{2,60}?)\s*:-?\s+\S", line)
                    )
                    if not m:
                        continue
                    label = m.group(1).strip()
                    if not label:
                        continue
                    # Skip ALL-CAPS labels (content sub-keys, not template structure)
                    word_tokens = re.findall(r"[A-Za-z]{2,}", label)
                    if word_tokens and all(t == t.upper() for t in word_tokens):
                        continue
                    # Skip long labels (prose)
                    if len(label.split()) > 7:
                        continue
                    # Skip purely numeric
                    if re.match(r"^\d+[.\s]*$", label):
                        continue
                    labels.add(label.lower())
        finally:
            doc.close()
        return frozenset(labels)
    except Exception as exc:
        logger.warning("_compute_raw_label_fingerprint failed: %s", exc)
        return frozenset()


def _update_raw_label_fp(variant: "DocumentVariant", raw_labels: frozenset[str]) -> None:
    """
    Accumulate raw_labels into the variant's structural_profile_json["raw_label_fp"].
    The stored set only grows — each new document can add labels seen for the first
    time, giving a progressively richer template fingerprint over time.
    """
    try:
        profile = json.loads(variant.structural_profile_json or "{}")
    except Exception:
        profile = {}
    existing = frozenset(profile.get("raw_label_fp") or [])
    merged   = existing | raw_labels
    if len(merged) > len(existing):
        profile["raw_label_fp"]         = sorted(merged)
        profile["raw_label_fp_updated"] = datetime.utcnow().isoformat()
        variant.structural_profile_json = json.dumps(profile)


def _slugify(text: str) -> str:
    """Normalise company name to slug: "TATA STEEL LIMITED" → "tata-steel-limited"."""
    text = text.lower().strip()
    # Strip common legal suffixes that don't differentiate the company
    text = re.sub(r"\b(limited|ltd\.?|gmbh|inc\.?|corp\.?|s\.?a\.?|b\.?v\.?|ag|llc)\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation → space
    text = re.sub(r"\s+", "-", text.strip())  # whitespace → hyphen
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:80]


def _build_variant_key(issuer_slug: str, fingerprint: str) -> str:
    """
    Composite key: "{issuer_slug}__{first_4_fields}"
    Example: "tata-steel__delivery_date,grade,material_number,specification"
    Keeps it human-readable while still being unique enough.
    """
    # Use first 4 non-trivial fingerprint fields for readability
    fields_short = ",".join(fingerprint.split(",")[:4]) if fingerprint else "generic"
    return f"{issuer_slug}__{fields_short}"[:200]


def _build_variant_label(doc: Document, issuer_slug: str, existing: list) -> str:
    """
    Human-readable label for the new variant.
    Format letter is assigned across ALL variants for this class (A, B, C …),
    not per-issuer — since issuer_slug is no longer a matching discriminator.

    Label strategy:
      - Known reliable issuer (issuer / from_company / buyer_name etc.) →
        "<Issuer> <Class> — Format A"
      - Unknown issuer (no reliable template-owner field found, e.g. the only
        name on the PO is the recipient) →
        "<Class> — Format A"   (clean, doesn't expose wrong company name)
    """
    letter = chr(ord("A") + len(existing))   # A, B, C … across all variants

    class_name = doc.document_class.name if doc.document_class else ""

    if issuer_slug == "unknown":
        return f"{class_name} — Format {letter}" if class_name else f"Format {letter}"

    display = issuer_slug.replace("-", " ").title()
    return (
        f"{display} {class_name} — Format {letter}"
        if class_name
        else f"{display} — Format {letter}"
    )


def _compute_stage(variant: DocumentVariant) -> str:
    """Advance learning stage based on doc_count (unless already LEARNED/OPTIMISED)."""
    current = variant.learning_stage or "ZERO_SHOT"
    # Never regress from operator-confirmed states
    if current in ("LEARNED", "OPTIMISED"):
        return current
    count = variant.doc_count or 0
    if count >= OPTIMISED_THRESHOLD:
        return "OPTIMISED"
    if count >= PROPOSED_THRESHOLD:
        return "LEARNED_PROPOSED"
    if count >= LEARNING_THRESHOLD:
        return "LEARNING"
    return "ZERO_SHOT"
