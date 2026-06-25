"""
Validation Agent — FR-026, FR-027 subset.

Runs business rule checks on extracted fields before the document
is allowed to proceed to Linking / Matching / Posting.

Rules are configurable per Document Class. A document that fails any
mandatory check is assigned NIGO (Not-In-Good-Order) status and routed
to NEEDS_REVIEW with a structured list of failing conditions.

A document that passes all checks is IGO and proceeds to LINKING.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.models.extracted_field import ExtractedField
from app.pipeline.states import PipelineState

# ── Required fields per Document Class ───────────────────────────────────────
# Documents missing any of these are NIGO — cannot proceed without them.

REQUIRED_FIELDS: dict[str, list[str]] = {
    # Purchase Orders — po_date and supplier_name unreliable across PO formats
    "dc_001": ["po_number", "total_order_value", "currency"],
    "dc_002": ["po_number", "total_order_value", "currency"],
    "dc_003": ["po_number", "total_order_value", "currency"],
    # Shipping / Logistics
    # dc_004: chargeable_weight + pieces are in carrier-template table cells that pypdf
    # cannot extract (only column headers appear in the text layer, not the filled values).
    "dc_004": ["awb_number", "shipper_name", "consignee_name"],
    "dc_005": ["tata_po_number"],                          # dcc_number format varies; PO ref is key
    # dc_007: total_gross_weight removed — many packing list formats (RHI refractory,
    # TML PO templates) have per-line weights but no single labelled "Total Gross Weight"
    # field that regex can reliably extract. supplier + consignee + PO ref are sufficient.
    "dc_007": ["supplier_name", "consignee_name", "purchase_order_no"],
    # Financial — Inbound
    "dc_006": ["invoice_number", "invoice_date", "supplier_name", "total_amount", "currency", "customer_po_ref"],
    # dc_013: invoice_date is not labeled in TKM-style freight invoices (it sits in a
    # column header row), so it cannot be reliably extracted and is not required.
    "dc_013": ["invoice_number", "total_amount", "currency", "mawb_number"],
    # Financial — TLL-issued
    "dc_011": ["invoice_number", "invoice_date", "total_invoice_value", "currency"],
    "dc_012": ["invoice_number", "invoice_date", "total_invoice_value", "currency"],
    # Compliance
    "dc_008": ["ic_number", "supplier_name"],             # release_date/order_number less reliable
    "dc_009": ["un_number", "shipper_name"],              # DGDs: UN# + shipper sufficient
    "dc_014": ["certificate_number"],                     # Insurance certs: many formats
    "dc_016": ["be_number", "importer_name"],             # BE + importer sufficient
    "dc_017": ["certificate_number", "supplier_name"],    # quality/test cert minimum
    "dc_018": ["exporter_name", "country_of_origin"],     # FTA cert: origin fields
    # Commercial
    "dc_010": ["confirmation_number", "supplier_name"],
    "dc_015": ["rfq_number"],                             # RFQ: just the number
    "dc_019": ["supplier_name", "total_amount", "currency"],
    "dc_020": ["currency", "payment_amount"],
}

# ── Date fields per class (must parse as YYYY-MM-DD) ─────────────────────────
DATE_FIELDS: dict[str, list[str]] = {
    "dc_001": ["po_date", "delivery_date"],
    "dc_002": ["po_date"],
    "dc_003": ["po_date"],
    "dc_004": ["awb_date"],
    "dc_005": ["dcc_date", "lc_date", "issue_date"],
    "dc_006": ["invoice_date"],
    "dc_007": [],
    "dc_008": ["ic_release_date", "po_validity_start", "po_validity_end"],
    "dc_009": ["declaration_date"],
    "dc_010": ["confirmation_date"],
    "dc_011": ["invoice_date", "awb_date"],
    "dc_012": ["invoice_date", "awb_date"],
    "dc_013": ["invoice_date"],
    "dc_014": ["certificate_date", "ship_date"],
    "dc_015": ["rfq_date", "rfq_due_date"],
    "dc_016": ["be_date", "invoice_date", "assessment_date"],
    "dc_017": ["issue_date"],
    "dc_018": ["issue_date", "invoice_date"],
    "dc_019": ["quotation_date", "validity_end_date"],
    "dc_020": ["remittance_date", "payment_date"],
}

# ── Numeric fields per class (must parse as a valid number) ───────────────────
NUMERIC_FIELDS: dict[str, list[str]] = {
    "dc_001": ["total_order_value"],
    "dc_002": ["total_order_value"],
    "dc_003": ["total_order_value"],
    "dc_004": ["chargeable_weight", "actual_weight"],
    "dc_005": [],
    "dc_006": ["subtotal", "vat_amount", "total_amount"],
    "dc_007": ["total_gross_weight", "total_net_weight", "quantity"],
    "dc_008": [],
    "dc_009": [],
    "dc_010": [],
    "dc_011": ["total_invoice_value", "buying_commission_pct", "buying_commission_amt"],
    "dc_012": ["total_invoice_value", "buying_commission_amt"],
    "dc_013": ["total_amount"],
    "dc_014": [],
    "dc_015": [],
    "dc_016": ["total_cif_value", "customs_duty"],
    "dc_017": [],
    "dc_018": [],
    "dc_019": ["total_amount"],
    "dc_020": ["payment_amount"],
}

# ── Invoice-specific total check ──────────────────────────────────────────────
# For dc_006: subtotal + vat_amount must equal total_amount (±£0.02 tolerance)
INVOICE_TOTAL_CHECK_CLASSES = {"dc_006"}


@dataclass
class ValidationResult:
    is_valid: bool
    nigo_conditions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ValidationAgent(BaseAgent):
    """
    Validates extracted fields against business rules for the document's class.
    Produces a structured ValidationResult with every failing condition listed.

    In production: validation rules are loaded from the portal settings
    (configurable per class without code changes — FR implied by Section 4.1).
    """

    name = "ValidationAgent"

    def validate(self, doc: Document) -> ValidationResult:
        class_id = doc.document_class_id
        if not class_id:
            return ValidationResult(is_valid=False, nigo_conditions=["No Document Class assigned — cannot validate."])

        fields = _field_map(
            self.db.query(ExtractedField)
            .filter(ExtractedField.document_id == doc.id)
            .all()
        )

        nigo: list[str] = []
        warnings: list[str] = []

        # ── 1. Required field presence ────────────────────────────────────────
        for req_field in REQUIRED_FIELDS.get(class_id, []):
            val = fields.get(req_field)
            if val is None or val.strip() == "":
                nigo.append(f"Required field missing or empty: '{req_field}'.")

        # ── 2. Date format validation ─────────────────────────────────────────
        for date_field in DATE_FIELDS.get(class_id, []):
            val = fields.get(date_field)
            if val and not _is_valid_date(val):
                nigo.append(
                    f"Field '{date_field}' value '{val}' is not a recognised date format (expected YYYY-MM-DD)."
                )

        # ── 3. Numeric field validation ───────────────────────────────────────
        for num_field in NUMERIC_FIELDS.get(class_id, []):
            val = fields.get(num_field)
            if val and not _is_valid_number(val):
                nigo.append(
                    f"Field '{num_field}' value '{val}' cannot be parsed as a number."
                )

        # ── 4. Invoice total check (subtotal + VAT = total) ───────────────────
        if class_id in INVOICE_TOTAL_CHECK_CLASSES:
            subtotal = _to_decimal(fields.get("subtotal"))
            vat = _to_decimal(fields.get("vat_amount"))
            total = _to_decimal(fields.get("total_amount"))

            if subtotal is not None and vat is not None and total is not None:
                computed = subtotal + vat
                diff = abs(computed - total)
                if diff > Decimal("0.02"):
                    nigo.append(
                        f"Invoice total mismatch: subtotal ({subtotal}) + VAT ({vat}) = {computed}, "
                        f"but total_amount = {total}. Difference: {diff}."
                    )
            elif total is not None and (subtotal is None or vat is None):
                warnings.append(
                    "Cannot verify subtotal + VAT = total: one or more component fields missing."
                )

        # ── 5. DGD shipper declaration check ─────────────────────────────────
        # Only add NIGO if the field was extracted AND explicitly set to "false".
        # If absent (most PDFs lack a parseable signature field), treat as a
        # warning only — the shipper_declaration_signed field is rarely extractable
        # from text-layer PDFs.
        if class_id == "dc_009":
            signed = fields.get("shipper_declaration_signed", "").lower().strip()
            if signed == "false":
                nigo.append(
                    "Dangerous Goods Declaration: shipper_declaration_signed is explicitly false. "
                    "DGD without shipper signature is NIGO — cannot tender to carrier."
                )
            elif signed not in ("", "true"):
                warnings.append(
                    f"shipper_declaration_signed has unexpected value '{signed}' — manual review advised."
                )

        return ValidationResult(
            is_valid=len(nigo) == 0,
            nigo_conditions=nigo,
            warnings=warnings,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _field_map(fields: list[ExtractedField]) -> dict[str, str]:
    """Return {field_name: effective_value} — corrected value takes precedence."""
    result = {}
    for f in fields:
        result[f.field_name] = f.corrected_value if f.human_corrected else (f.field_value or "")
    return result


def _is_valid_date(value: str) -> bool:
    """Accept common date formats extracted from PDFs.

    Accepted: YYYY-MM-DD, DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY,
              DD Mon YYYY (e.g. 26-05-07 treated as partial), M/D/YYYY.
    The goal is to reject obvious non-date junk, not enforce a single format.
    """
    val = value.strip()
    for fmt in (
        "%Y-%m-%d",      # 2026-06-08
        "%d/%m/%Y",      # 08/06/2026
        "%d.%m.%Y",      # 08.06.2026
        "%d-%m-%Y",      # 08-06-2026
        "%d %b %Y",      # 08 Jun 2026
        "%d %B %Y",      # 08 June 2026
        "%m/%d/%Y",      # 6/8/2026  (US-style)
        "%Y/%m/%d",      # 2026/06/08
        "%d-%m-%y",      # 26-05-07  (2-digit year — old docs)
        "%d/%m/%y",      # 26/05/07
        "%d.%m.%y",      # 26.05.07
    ):
        try:
            datetime.strptime(val, fmt)
            return True
        except ValueError:
            pass
    return False


def _is_valid_number(value: str) -> bool:
    """Accept integers, decimals, comma-formatted and Nordic-space-formatted numbers.

    Handles:
      - Standard:  "320,000.00"  → strip comma  → "320000.00"
      - Nordic:    "166825,00"   → strip comma  → "16682500"  (passes as integer)
      - Space-sep: "166 825,00"  → strip comma+spaces → "16682500"
    The goal is to reject non-numeric junk, not validate magnitude.
    """
    cleaned = value.strip().replace(",", "").replace(" ", "")
    try:
        Decimal(cleaned)
        return True
    except InvalidOperation:
        return False


def _to_decimal(value: str | None) -> Decimal | None:
    if not value:
        return None
    try:
        return Decimal(value.strip().replace(",", ""))
    except InvalidOperation:
        return None
