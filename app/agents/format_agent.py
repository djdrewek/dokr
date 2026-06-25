"""
FormatAgent — learns the optimal format type for every extracted field in a variant.

For each field it:
  1. Collects recent ExtractedField values from the DB (up to MAX_SAMPLE)
  2. Runs pattern matching to infer a format type + confidence score
  3. For ambiguous fields (confidence < PATTERN_THRESHOLD) batches them into a
     single Haiku call for audit
  4. Writes an actionable format_hint back to variant.field_schema_json

Because the extraction prompt already reads format_hint (via hint_data.get("format_hint"))
improving these hints automatically tightens extraction — no other code change required.

Format types
────────────
  integer       30, 60 — "INTEGER — return only the number (e.g. 30). No units."
  decimal       1234.56, 0.5 — "DECIMAL number (e.g. 1234.56)."
  currency_amt  1,234.56, €2,500 — "Amount as digits (e.g. 1234.56). No symbol."
  currency_code EUR, GBP, USD — "ISO 4217 three-letter currency code (e.g. EUR)."
  date          15/03/2026, 2026-03-15 — "Date in {fmt} format (e.g. {ex})."
  datetime      2026-03-15T10:30:00 — "ISO datetime (e.g. {ex})."
  code          TUAC32609V, INV/2026/001 — "Reference code — copy verbatim (e.g. {ex})."
  incoterm      FOB, CIF, DAP — "Incoterm code (e.g. {ex})."
  country_code  GB, DE, IN — "ISO 3166-1 two-letter country code (e.g. {ex})."
  percentage    5.5, 5.5% — "Percentage — digits only, no % sign (e.g. {ex})."
  boolean       Yes, No — "One of: Yes / No."
  enum          APPROVED, PENDING — "One of the fixed values: {values}."
  name          Tata Steel UK — "Name — copy verbatim (e.g. {ex})."
  text          (default) — "Free text — copy verbatim from the document."
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
MAX_SAMPLE         = 30    # max values collected per field from DB
MIN_SAMPLE         = 3     # minimum values needed before inferring format
PATTERN_THRESHOLD  = 0.70  # pattern confidence ≥ this → skip Haiku audit
MIN_VARIANT_DOCS   = 3     # variant must have at least this many confirmed docs

# Known Incoterms (ICC 2020)
_INCOTERMS = frozenset([
    "EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP",
    "FAS", "FOB", "CFR", "CIF",
])

# Common ISO 4217 currency codes (not exhaustive, but covers >95% of trade docs)
_CURRENCY_CODES = frozenset([
    "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "AUD", "CAD", "SEK", "NOK",
    "DKK", "SGD", "HKD", "INR", "KRW", "MXN", "BRL", "ZAR", "NZD", "AED",
    "SAR", "TRY", "PLN", "CZK", "HUF", "RUB", "THB", "MYR", "IDR", "PHP",
    "ILS", "PKR", "NGN", "EGP", "QAR", "KWD", "BHD", "OMR",
])

# ISO 3166-1 alpha-2 country codes (common subset)
_COUNTRY_CODES = frozenset([
    "GB", "US", "DE", "FR", "IT", "ES", "NL", "BE", "SE", "NO", "DK", "FI",
    "PL", "CZ", "HU", "AT", "CH", "PT", "IE", "CN", "IN", "JP", "KR", "AU",
    "NZ", "CA", "MX", "BR", "ZA", "AE", "SA", "TR", "RU", "UA", "RO", "BG",
    "HR", "SK", "SI", "GR", "LT", "LV", "EE", "LU", "MT", "CY", "IS", "LI",
    "SG", "HK", "TH", "MY", "ID", "PH", "VN", "BD", "PK", "NG", "EG",
])


# ── Pattern classifiers ───────────────────────────────────────────────────────

def _classify_value(v: str) -> tuple[str, float]:
    """
    Classify a single string value into a format type.
    Returns (type_name, confidence) where confidence is 0.0–1.0.
    A higher confidence means the pattern match was unambiguous.
    """
    v = v.strip()
    if not v:
        return ("text", 0.0)

    vu = v.upper()

    # ── Incoterms (very specific — check early) ───────────────────────────────
    # Allow "FOB Shanghai", "CIF Mumbai" — first token must be incoterm
    first_token = vu.split()[0] if " " in vu else vu
    if first_token in _INCOTERMS and len(v) <= 30:
        return ("incoterm", 1.0)

    # ── Currency code (exact 3-letter ISO) ───────────────────────────────────
    if vu in _CURRENCY_CODES:
        return ("currency_code", 1.0)

    # ── Country code (exact 2-letter ISO) ────────────────────────────────────
    if len(v) == 2 and v.isupper() and vu in _COUNTRY_CODES:
        return ("country_code", 1.0)

    # ── Boolean ───────────────────────────────────────────────────────────────
    if vu in ("YES", "NO", "TRUE", "FALSE", "Y", "N", "1", "0",
              "SIGNED", "UNSIGNED", "CHECKED", "UNCHECKED"):
        return ("boolean", 0.95)

    # ── Datetime (ISO 8601) ───────────────────────────────────────────────────
    if re.match(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", v):
        return ("datetime", 1.0)

    # ── Date patterns ─────────────────────────────────────────────────────────
    if re.fullmatch(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", v):
        return ("date", 1.0)
    if re.fullmatch(r"\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}", v):
        return ("date", 1.0)
    # Partial: month name present
    if re.search(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
        v, re.IGNORECASE
    ) and re.search(r"\d{4}", v):
        return ("date", 0.90)

    # ── Percentage ────────────────────────────────────────────────────────────
    if re.fullmatch(r"\d+\.?\d*\s*%", v):
        return ("percentage", 1.0)
    if re.fullmatch(r"\d+\.\d+", v) and float(v) <= 100.0 and "." in v:
        # Could be percentage or decimal — lower confidence
        return ("decimal", 0.60)

    # ── Currency amount ───────────────────────────────────────────────────────
    if re.fullmatch(r"[£€$¥₹₩][\d,\s]+\.?\d*", v):
        return ("currency_amt", 1.0)
    if re.fullmatch(r"[\d,]+\.\d{2}", v) and "," in v:
        return ("currency_amt", 0.85)
    # Amount followed by currency code: "1234.56 EUR"
    if re.fullmatch(r"[\d,\.]+\s*[A-Z]{3}", v):
        amount_part, code_part = v.rsplit(None, 1)
        if code_part.upper() in _CURRENCY_CODES:
            return ("currency_amt", 0.90)

    # ── Pure integer ──────────────────────────────────────────────────────────
    # Allow comma-thousands: "1,234" also matches
    stripped_int = v.replace(",", "").replace(" ", "")
    if re.fullmatch(r"\d+", stripped_int) and not v.startswith("0"):
        # Long integers (8+ digits) are more likely reference codes
        if len(stripped_int) >= 8:
            return ("code", 0.75)
        return ("integer", 0.95)

    # ── Pure decimal ─────────────────────────────────────────────────────────
    if re.fullmatch(r"\d+\.\d+", stripped_int):
        return ("decimal", 0.85)

    # ── Reference codes ───────────────────────────────────────────────────────
    # Alphanumeric with slash/dash separators, no spaces or very short
    if re.fullmatch(r"[A-Z0-9][A-Z0-9\/\-\.]{2,29}", v.upper()) and not v.isdigit():
        # Pure uppercase letters only → might be a name abbreviation
        if re.fullmatch(r"[A-Z]+", v.upper()) and len(v) <= 6:
            return ("code", 0.65)  # ambiguous with enum/code
        return ("code", 0.85)

    # ── Fallback: short all-caps or enum candidate ─────────────────────────────
    if v.isupper() and 2 <= len(v) <= 20 and " " not in v:
        return ("code", 0.65)

    return ("text", 0.50)


def _infer_format_for_values(values: list[str]) -> tuple[str, float, str]:
    """
    Given a list of raw values for a single field, infer the most likely
    format type and return (type_name, confidence, format_hint).

    Returns a lower confidence when the sample is noisy/mixed.
    """
    if not values:
        return ("text", 0.0, "")

    # Classify each value
    typed: list[tuple[str, float]] = [_classify_value(v) for v in values]
    type_counts: Counter = Counter(t for t, _ in typed)
    most_common_type, most_common_count = type_counts.most_common(1)[0]
    majority_frac = most_common_count / len(values)

    avg_conf = sum(c for t, c in typed if t == most_common_type) / most_common_count

    # If < 60% of values agree on a type, mark as ambiguous
    if majority_frac < 0.60:
        return ("text", 0.40, "")

    # Enum detection: if majority_frac >= 0.60 AND the field has few distinct values
    # that REPEAT across documents (not unique reference codes).
    unique_vals = list({v.strip() for v in values})
    val_counts = Counter(v.strip() for v in values)
    # At least 2 distinct values must appear more than once (true enum behaviour)
    repeated_unique = sum(1 for cnt in val_counts.values() if cnt >= 2)
    if (
        most_common_type in ("text", "code", "boolean")
        and len(unique_vals) <= 7
        and len(values) >= 4
        and repeated_unique >= 2           # ← real enum: values recur
        and all(len(v) <= 30 for v in unique_vals)
    ):
        # Likely an enum
        return ("enum", 0.80, "")

    confidence = round(avg_conf * majority_frac, 3)
    return (most_common_type, confidence, "")


# ── Hint generators ───────────────────────────────────────────────────────────

_DATE_FMT_MAP = {
    # separator: format string
    "/": "DD/MM/YYYY",
    "-": "YYYY-MM-DD",
    ".": "DD.MM.YYYY",
}


def _date_fmt(example: str) -> str:
    for sep, fmt in _DATE_FMT_MAP.items():
        if sep in example:
            # Disambiguate YYYY-MM-DD vs DD-MM-YYYY
            if sep == "-" and len(example.split("-")[0]) == 4:
                return "YYYY-MM-DD"
            return fmt
    return "DD/MM/YYYY"


def _build_hint(ftype: str, values: list[str], enum_vals: list[str] | None = None) -> str:
    """Generate an actionable extraction hint string from a format type + examples."""
    ex = values[0] if values else ""

    if ftype == "integer":
        return f"INTEGER — return only the number (e.g. {ex}). No units, labels, or text."
    if ftype == "decimal":
        return f"DECIMAL number (e.g. {ex}). Digits and decimal point only."
    if ftype == "currency_amt":
        # Try to strip the symbol for the example
        clean = re.sub(r"[£€$¥₹₩\s]", "", ex).strip(",")
        return f"Currency amount — digits only, no symbol (e.g. {clean or ex})."
    if ftype == "currency_code":
        return f"ISO 4217 three-letter currency code only (e.g. {ex})."
    if ftype == "date":
        fmt = _date_fmt(ex)
        return f"Date in {fmt} format (e.g. {ex}). No time component."
    if ftype == "datetime":
        return f"ISO 8601 datetime (e.g. {ex})."
    if ftype == "code":
        return f"Reference code — copy verbatim (e.g. {ex})."
    if ftype == "incoterm":
        return f"Incoterm code only (e.g. {ex}). Two to four letters."
    if ftype == "country_code":
        return f"ISO 3166-1 two-letter country code only (e.g. {ex})."
    if ftype == "percentage":
        clean = ex.rstrip("% ").strip()
        return f"Percentage — digits only, no % sign (e.g. {clean or ex})."
    if ftype == "boolean":
        return "Boolean — return exactly: Yes or No."
    if ftype == "enum":
        vals_str = " / ".join(enum_vals[:6]) if enum_vals else ex
        return f"One of the fixed values: {vals_str}."
    if ftype == "name":
        return f"Name — copy verbatim (e.g. {ex})."
    # text / fallback
    return "Copy verbatim from the document."


# ── Main agent class ──────────────────────────────────────────────────────────

class FormatAgent:
    """
    Learns field format types for a DocumentVariant and writes actionable
    format_hint values into variant.field_schema_json.

    Usage (called from dashboard confirm-fields or manually):
        agent = FormatAgent(db)
        result = agent.run_for_variant(variant_id)
    """

    name = "FormatAgent"

    def __init__(self, db: Session, config: dict | None = None):
        self.db = db
        self.config = config or {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_for_variant(self, variant_id: str) -> dict:
        """
        Run format inference for all fields in a variant.

        Returns a summary dict:
        {
          "fields_updated": 8,
          "fields_unchanged": 2,
          "haiku_audited": 3,
          "field_types": {"payment_terms": "integer", ...},
          "ran_at": "2026-06-22T10:30:00"
        }
        """
        from app.models.document import DocumentVariant
        from app.models.extracted_field import ExtractedField
        from app.models.document import Document

        variant: Optional[DocumentVariant] = self.db.query(DocumentVariant).filter_by(
            id=variant_id
        ).first()
        if not variant:
            logger.warning("FormatAgent: variant %s not found", variant_id)
            return {"error": "variant_not_found"}

        # Load current field schema
        schema: dict = {}
        if variant.field_schema_json:
            try:
                schema = json.loads(variant.field_schema_json)
            except Exception:
                schema = {}

        if not schema:
            logger.info("FormatAgent: variant %s has no field_schema_json — skipping", variant_id)
            return {"fields_updated": 0, "fields_unchanged": 0, "note": "no_schema"}

        # Collect ExtractedField values per field for this variant
        # We join Document to filter by variant_id
        rows = (
            self.db.query(ExtractedField.field_name, ExtractedField.field_value)
            .join(Document, ExtractedField.document_id == Document.id)
            .filter(Document.variant_id == variant_id)
            .filter(ExtractedField.field_type == "scalar")
            .filter(ExtractedField.field_value.isnot(None))
            .filter(ExtractedField.field_value != "")
            .order_by(ExtractedField.id.desc())
            .limit(MAX_SAMPLE * len(schema))
            .all()
        )

        # Group by field name
        field_values: dict[str, list[str]] = {}
        for fname, fval in rows:
            if fname not in schema:
                continue
            if fname not in field_values:
                field_values[fname] = []
            if fval not in field_values[fname] and len(field_values[fname]) < MAX_SAMPLE:
                field_values[fname].append(fval)

        if not field_values:
            logger.info("FormatAgent: variant %s — no extracted values found yet", variant_id)
            return {"fields_updated": 0, "fields_unchanged": 0, "note": "no_values_yet"}

        # Pattern-classify each field
        pattern_results: dict[str, dict] = {}   # field → {type, confidence, values}
        ambiguous_fields: list[str] = []

        for fname, values in field_values.items():
            if len(values) < MIN_SAMPLE:
                # Not enough data — keep existing hint unchanged
                continue

            ftype, confidence, _ = _infer_format_for_values(values)
            unique_vals = sorted({v.strip() for v in values})

            pattern_results[fname] = {
                "type":     ftype,
                "confidence": confidence,
                "values":   values,
                "unique":   unique_vals,
            }

            if confidence < PATTERN_THRESHOLD:
                ambiguous_fields.append(fname)

        # Haiku audit for ambiguous fields
        haiku_results: dict[str, dict] = {}
        if ambiguous_fields:
            haiku_results = self._haiku_audit(ambiguous_fields, pattern_results)

        # Merge results and write hints back
        fields_updated = 0
        fields_unchanged = 0
        field_types: dict[str, str] = {}

        for fname, field_data in schema.items():
            if not isinstance(field_data, dict):
                continue

            if fname not in pattern_results:
                # Not enough data to infer — keep existing hint
                fields_unchanged += 1
                existing = field_data.get("format_type", "")
                if existing:
                    field_types[fname] = existing
                continue

            pr = pattern_results[fname]
            ftype   = pr["type"]
            values  = pr["values"]
            unique  = pr["unique"]

            # Override with Haiku result if we got one
            if fname in haiku_results:
                hr = haiku_results[fname]
                ftype = hr.get("format_type", ftype)
                hint  = hr.get("format_hint", "")
            else:
                hint = _build_hint(
                    ftype,
                    values,
                    enum_vals=unique if ftype == "enum" else None,
                )

            # Update best example values — use 3 most representative
            best_examples = _pick_examples(values, ftype)

            old_hint = field_data.get("format_hint", "")
            if hint and hint != old_hint:
                field_data["format_hint"]  = hint
                field_data["format_type"]  = ftype   # new metadata key
                field_data["example_values"] = best_examples
                fields_updated += 1
            else:
                fields_unchanged += 1

            field_types[fname] = ftype

        # Write updated schema back
        ran_at = datetime.utcnow().isoformat()

        # Store agent run metadata at top level of schema (special key)
        schema["__format_agent__"] = {
            "ran_at": ran_at,
            "fields_typed": len(field_types),
            "haiku_audited": len(haiku_results),
        }

        variant.field_schema_json = json.dumps(schema)
        try:
            self.db.commit()
        except Exception as exc:
            logger.warning("FormatAgent: commit failed for variant %s: %s", variant_id, exc)
            self.db.rollback()
            return {"error": str(exc)}

        logger.info(
            "FormatAgent: variant %s — %d updated, %d unchanged, %d haiku-audited",
            variant_id, fields_updated, fields_unchanged, len(haiku_results),
        )

        return {
            "fields_updated": fields_updated,
            "fields_unchanged": fields_unchanged,
            "haiku_audited": len(haiku_results),
            "field_types": field_types,
            "ran_at": ran_at,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Haiku audit
    # ─────────────────────────────────────────────────────────────────────────

    def _haiku_audit(
        self,
        field_names: list[str],
        pattern_results: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Send a single batched request to Haiku asking it to determine the
        format type and write an actionable extraction hint for each ambiguous field.

        Returns {field_name: {"format_type": str, "format_hint": str}}.
        Falls back silently to empty dict on any error.
        """
        from app.config import settings

        if not settings.anthropic_api_key:
            logger.debug("FormatAgent: no API key — skipping Haiku audit")
            return {}

        # Build the prompt
        field_block_lines = []
        for fname in field_names:
            pr = pattern_results.get(fname, {})
            vals = pr.get("values", [])
            sample = vals[:8]  # show up to 8 examples
            field_block_lines.append(f'  "{fname}": {json.dumps(sample)}')

        fields_block = "{\n" + ",\n".join(field_block_lines) + "\n}"

        prompt = f"""You are a data format expert analysing field values extracted from business documents (invoices, purchase orders, shipping docs).

For each field below I have listed up to 8 sample extracted values. Determine:
1. The best format type (choose ONE from: integer, decimal, currency_amt, currency_code, date, datetime, code, incoterm, country_code, percentage, boolean, enum, name, text)
2. A short, actionable extraction instruction telling an AI model EXACTLY what to return for this field.

Field samples:
{fields_block}

Rules for the extraction instruction:
- Be specific and concise (≤ 15 words)
- Include a concrete example from the samples
- For integers: say "INTEGER — return only the number (e.g. X). No units."
- For currency codes: say "ISO 4217 three-letter code only (e.g. EUR)."
- For dates: say "Date in DD/MM/YYYY format (e.g. X)."
- For codes/references: say "Reference code — copy verbatim (e.g. X)."
- For enum: say "One of the fixed values: A / B / C."

Respond with ONLY valid JSON, no prose, no markdown:
{{
  "field_name": {{"format_type": "...", "format_hint": "..."}},
  ...
}}"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            parsed: dict = json.loads(raw)

            # Validate and clean
            valid = {}
            for fname, info in parsed.items():
                if isinstance(info, dict) and "format_type" in info:
                    valid[fname] = {
                        "format_type": str(info.get("format_type", "text")),
                        "format_hint": str(info.get("format_hint", "")),
                    }
            return valid

        except Exception as exc:
            logger.warning("FormatAgent: Haiku audit failed: %s", exc)
            return {}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _pick_examples(values: list[str], ftype: str) -> list[str]:
    """
    Choose up to 3 representative example values for a field.
    For integers/decimals: pick distinct lengths.
    For codes/dates: deduplicate and pick shortest variants.
    For enum: pick all unique values (up to 5).
    """
    unique = list(dict.fromkeys(v.strip() for v in values if v.strip()))

    if ftype == "enum":
        return unique[:5]
    if ftype in ("integer", "decimal", "currency_amt"):
        # Sort by length descending — longer numbers tend to be more representative
        unique.sort(key=len, reverse=True)
        return unique[:3]
    # General: first 3 unique values
    return unique[:3]


# ── Standalone helper used by schema_learner.py ───────────────────────────────

def infer_format_hint(examples: list[str]) -> str:
    """
    Drop-in replacement for schema_learner._infer_format_hint.
    Uses the full pattern classifier instead of the old 5-line version.
    Called synchronously (no DB, no AI) — safe to use in any context.
    """
    if not examples:
        return ""
    ftype, conf, _ = _infer_format_for_values(examples)
    if ftype == "text" or conf < 0.40:
        return ""
    unique = list(dict.fromkeys(v.strip() for v in examples if v.strip()))
    return _build_hint(ftype, unique, enum_vals=unique if ftype == "enum" else None)
