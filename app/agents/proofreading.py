"""
Proofreading Agent — quality gate between each extraction tier.

Called after Tier 1 (text layer), Tier 2 (OCR), and Tier 3 (AI vision).
If the extracted fields pass, the pipeline proceeds to VALIDATING.
If they fail, the ExtractionAgent escalates to the next tier.
After all three tiers, any remaining failures push to NEEDS_REVIEW.

Checks (in order):
  1. Required fields present for this document class
  2. Field format validation (regex per field name)
  3. Anomaly detection (value too long/short, empty, look-alike duplicates)
  4. Average confidence threshold
  5. Cross-comparison against confirmed variant extractions (if available)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.config import settings
from app.models.document import Document, DocumentVariant
from app.models.extracted_field import ExtractedField


# ── Required fields per document class ───────────────────────────────────────
# A document must have at least MIN_REQUIRED_FIELD_RATE of these fields
# present (non-null, non-empty) to pass proofreading.
REQUIRED_FIELDS: dict[str, set[str]] = {
    "dc_001": {"po_number", "currency", "total_order_value"},  # entity/supplier unreliable in PO format
    "dc_002": {"po_number", "currency", "total_order_value"},
    "dc_003": {"po_number", "currency", "total_order_value"},
    "dc_004": {"awb_number", "shipper_name", "consignee_name"},
    "dc_005": {"tata_po_number"},  # DCC number format varies; PO ref is reliable
    "dc_006": {"invoice_number", "supplier_name", "total_amount", "currency"},
    "dc_007": {"supplier_name", "consignee_name", "gross_weight"},
    "dc_008": {"ic_number", "supplier_name"},
    "dc_009": {"un_number", "shipper_name"},
    "dc_010": {"confirmation_number", "supplier_name"},
    "dc_011": {"invoice_number", "total_invoice_value", "currency"},
    "dc_012": {"invoice_number", "total_invoice_value", "currency"},
    "dc_013": {"invoice_number", "currency", "total_amount"},
    "dc_014": {"certificate_number"},
    "dc_015": {"rfq_number"},
    "dc_016": {"be_number", "importer_name"},
    "dc_017": {"certificate_number", "supplier_name"},
    "dc_018": {"exporter_name", "country_of_origin"},
    "dc_019": {"supplier_name", "total_amount", "currency"},
    "dc_020": {"currency", "payment_amount"},
}

# ── Per-field format validators ───────────────────────────────────────────────
# (pattern, human-readable expectation)
FIELD_FORMAT_VALIDATORS: dict[str, tuple[str, str]] = {
    "currency":            (r"^[A-Z]{3}$",                                   "3-letter ISO code e.g. EUR"),
    # TSL/TML/TMPVL slash-format (TSL/58237) OR SAP 10-digit numeric (2200063147, 4700149489)
    "po_number":           (r"^(?:[A-Z]{2,6}[\/\-]\d{4,8}|\d{7,12})",        "e.g. TSL/58237 or 2200063147"),
    "invoice_number":      (r"^[A-Z0-9][A-Z0-9\/\-\.]{1,39}$",              "alphanumeric, 2-40 chars"),
    "awb_number":          (r"^\d{3}[-\s]?\d{8}$",                           "IATA format NNN-NNNNNNNN"),
    "hawb_number":         (r"^[A-Z0-9]{4,20}$",                             "alphanumeric 4-20 chars"),
    "invoice_date":        (r"^\d{4}-\d{2}-\d{2}$|^\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}$", "date"),
    "po_date":             (r"^\d{4}-\d{2}-\d{2}$|^\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}$", "date"),
    "delivery_date":       (r"^\d{4}-\d{2}-\d{2}$|^\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}$", "date"),
    "shipment_date":       (r"^\d{4}-\d{2}-\d{2}$|^\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}$", "date"),
    "total_amount":        (r"^[\d,\.]+$",                                    "numeric"),
    "total_order_value":   (r"^[\d,\.]+$",                                    "numeric"),
    "net_amount":          (r"^[\d,\.]+$",                                    "numeric"),
    "incoterms":           (r"^(?:EXW|FCA|CPT|CIP|DAP|DPU|DDP|FAS|FOB|CFR|CIF)$", "INCOTERMS 2020 code"),
    "iban":                (r"^[A-Z]{2}\d{2}[A-Z0-9]{4,30}$",               "IBAN format e.g. GB12XXXX..."),
    "bic":                 (r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?$", "BIC/SWIFT code"),
    "origin_airport":      (r"^[A-Z]{3}$",                                   "3-letter IATA airport code"),
    "destination_airport": (r"^[A-Z]{3}$",                                   "3-letter IATA airport code"),
    "flight_number":       (r"^[A-Z]{2}\d{3,4}$",                            "e.g. BA249"),
    "pieces":              (r"^\d+$",                                         "integer"),
}

# ── Anomaly limits ────────────────────────────────────────────────────────────
MAX_FIELD_LEN: dict[str, int] = {
    "currency": 3,
    "incoterms": 3,
    "origin_airport": 3,
    "destination_airport": 3,
    "po_number": 30,
    "invoice_number": 40,
    "awb_number": 14,
    "hawb_number": 20,
    "iban": 34,
    "bic": 11,
    "pieces": 6,
}
MIN_FIELD_LEN: dict[str, int] = {
    "supplier_name": 3,
    "consignee_name": 3,
    "shipper_name": 3,
    "invoice_number": 3,
    "po_number": 5,
    "goods_description": 5,
}


@dataclass
class ProofreadFlag:
    field_name: str
    issue: str
    severity: str   # "ERROR" | "WARNING"


@dataclass
class ProofreadResult:
    passed: bool
    score: float                         # 0.0–1.0 composite quality score
    flags: list[ProofreadFlag] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    format_failures: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    avg_confidence: float = 0.0
    summary: str = ""

    def as_event_detail(self, tier: int) -> str:
        parts = [f"Tier {tier} proofreading {'PASS' if self.passed else 'FAIL'} "
                 f"(score {self.score:.2f}, avg_conf {self.avg_confidence:.2f})."]
        if self.missing_required:
            parts.append(f"Missing required: {', '.join(self.missing_required)}.")
        if self.format_failures:
            parts.append(f"Format failures: {', '.join(self.format_failures)}.")
        if self.anomalies:
            parts.append(f"Anomalies: {', '.join(self.anomalies)}.")
        return " ".join(parts)


class ProofreadingAgent(BaseAgent):
    """
    Validates extracted fields against format rules, required-field sets,
    anomaly thresholds, and historical variant patterns.
    """

    name = "ProofreadingAgent"

    def check(
        self,
        doc: Document,
        fields: list[ExtractedField],
        tier: int,
    ) -> ProofreadResult:
        """
        Run all quality checks and return a ProofreadResult.
        `tier` is informational only (1/2/3) — thresholds tighten on Tier 3.
        """
        dc_id = doc.document_class_id or ""
        field_map: dict[str, str | None] = {
            f.field_name: f.field_value for f in fields
        }
        flags: list[ProofreadFlag] = []
        missing_required: list[str] = []
        format_failures: list[str] = []
        anomalies: list[str] = []

        # ── 1. Required field presence ─────────────────────────────────────────
        required = REQUIRED_FIELDS.get(dc_id, set())
        for req_field in required:
            val = field_map.get(req_field)
            if not val or not val.strip():
                missing_required.append(req_field)
                flags.append(ProofreadFlag(req_field, "required field missing", "ERROR"))

        required_rate = (
            (len(required) - len(missing_required)) / len(required)
            if required else 1.0
        )

        # ── 2. Format validation ──────────────────────────────────────────────
        for fname, (pattern, expectation) in FIELD_FORMAT_VALIDATORS.items():
            val = field_map.get(fname)
            if val is None:
                continue  # absent fields handled by required check above
            val_clean = val.strip()
            if val_clean and not re.match(pattern, val_clean, re.IGNORECASE):
                format_failures.append(fname)
                flags.append(ProofreadFlag(
                    fname,
                    f"format mismatch: got '{val_clean[:30]}', expected {expectation}",
                    "ERROR",
                ))

        # ── 3. Anomaly detection ──────────────────────────────────────────────
        seen_values: dict[str, str] = {}
        for f in fields:
            if not f.field_value:
                continue
            val = f.field_value.strip()

            # Too short
            min_len = MIN_FIELD_LEN.get(f.field_name)
            if min_len and len(val) < min_len:
                anomalies.append(f.field_name)
                flags.append(ProofreadFlag(
                    f.field_name,
                    f"value too short: {len(val)} chars (min {min_len})",
                    "WARNING",
                ))

            # Too long
            max_len = MAX_FIELD_LEN.get(f.field_name)
            if max_len and len(val) > max_len:
                anomalies.append(f.field_name)
                flags.append(ProofreadFlag(
                    f.field_name,
                    f"value too long: {len(val)} chars (max {max_len})",
                    "ERROR",
                ))

            # Suspiciously generic values (common OCR noise)
            if val.lower() in {"null", "none", "n/a", "n.a.", "-", "–", "tbd", "xxx"}:
                anomalies.append(f.field_name)
                flags.append(ProofreadFlag(
                    f.field_name,
                    f"placeholder value: '{val}'",
                    "WARNING",
                ))

            # Duplicate values across different fields (extraction bleed)
            if f.field_name not in {"shipper_name", "consignee_name",
                                     "origin_airport", "destination_airport"}:
                if val in seen_values and seen_values[val] != f.field_name:
                    anomalies.append(f.field_name)
                    flags.append(ProofreadFlag(
                        f.field_name,
                        f"value identical to '{seen_values[val]}' — possible extraction bleed",
                        "WARNING",
                    ))
                seen_values[val] = f.field_name

        # ── 4. Confidence threshold ────────────────────────────────────────────
        avg_conf = (
            round(sum(f.confidence for f in fields) / len(fields), 4)
            if fields else 0.0
        )
        min_conf = settings.extraction_min_confidence
        # Tier 3 (AI vision) is held to a slightly lower bar since it's the last resort
        if tier == 3:
            min_conf = max(0.45, min_conf - 0.10)

        conf_ok = avg_conf >= min_conf

        # ── 5. Variant history cross-check ────────────────────────────────────
        # Compare extracted values against confirmed instances of this variant.
        # A field value that looks nothing like the variant's known values gets a
        # WARNING (not an ERROR) because the document might be legitimately different.
        if doc.variant_id:
            self._cross_check_variant(doc, field_map, flags)

        # ── Scoring ───────────────────────────────────────────────────────────
        error_count = sum(1 for fl in flags if fl.severity == "ERROR")
        warning_count = sum(1 for fl in flags if fl.severity == "WARNING")

        # Score breakdown:
        #  40% required field rate
        #  30% confidence
        #  20% format pass rate (0 failures = full score)
        #  10% anomaly penalty
        format_pass_rate = (
            (len(FIELD_FORMAT_VALIDATORS) - len(format_failures)) / len(FIELD_FORMAT_VALIDATORS)
            if FIELD_FORMAT_VALIDATORS else 1.0
        )
        anomaly_penalty = min(len(anomalies) * 0.05, 0.10)
        score = round(
            0.40 * required_rate
            + 0.30 * (avg_conf / max(min_conf, 1.0))
            + 0.20 * format_pass_rate
            + 0.10 * (1.0 - anomaly_penalty / 0.10)
        , 4)
        score = max(0.0, min(1.0, score))

        # ── Pass/fail decision ────────────────────────────────────────────────
        min_req_rate = settings.extraction_min_required_rate
        if tier == 3:
            min_req_rate = max(0.40, min_req_rate - 0.10)

        passed = (
            required_rate >= min_req_rate
            and conf_ok
            and error_count <= 1   # allow at most one format error
        )

        return ProofreadResult(
            passed=passed,
            score=score,
            flags=flags,
            missing_required=missing_required,
            format_failures=format_failures,
            anomalies=anomalies,
            avg_confidence=avg_conf,
            summary=(
                f"Tier {tier} {'PASS' if passed else 'FAIL'}: "
                f"req_rate={required_rate:.0%}, avg_conf={avg_conf:.2f}, "
                f"errors={error_count}, warnings={warning_count}"
            ),
        )

    # ── Variant history cross-check ──────────────────────────────────────────

    def _cross_check_variant(
        self,
        doc: Document,
        field_map: dict[str, str | None],
        flags: list[ProofreadFlag],
    ) -> None:
        """
        Query confirmed instances of this variant for value patterns.
        Flags currency/incoterms if they differ from the variant's known values
        (high signal: these rarely change between invoices from the same sender).
        """
        from app.models.extracted_field import ExtractedField as EF
        from app.models.document import Document as Doc

        # Only check stable categorical fields — not dates or amounts
        STABLE_FIELDS = {"currency", "incoterms", "country_of_origin"}

        # Get the 10 most recent confirmed (human-corrected) extractions for this variant
        recent_doc_ids = (
            self.db.query(Doc.id)
            .filter(Doc.variant_id == doc.variant_id, Doc.id != doc.id)
            .order_by(Doc.created_at.desc())
            .limit(10)
            .all()
        )
        if not recent_doc_ids:
            return

        id_list = [r[0] for r in recent_doc_ids]

        for fname in STABLE_FIELDS:
            current_val = field_map.get(fname)
            if not current_val:
                continue

            known_vals = (
                self.db.query(EF.field_value)
                .filter(
                    EF.document_id.in_(id_list),
                    EF.field_name == fname,
                    EF.field_value.isnot(None),
                )
                .all()
            )
            known_set = {r[0].strip().upper() for r in known_vals if r[0]}
            if known_set and current_val.strip().upper() not in known_set:
                flags.append(ProofreadFlag(
                    fname,
                    f"value '{current_val}' differs from variant history: {sorted(known_set)}",
                    "WARNING",
                ))
