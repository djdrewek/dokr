"""
Matching Agent — seven-check three-way match (FR-Section 11).

Compares the Invoice against the Purchase Order and any supporting
documents (Packing List, Inspection Certificate) that share the same
ShipmentRecord.

Checks (each produces PASS/FAIL/SKIP if missing data):
  1. Supplier name match        — Invoice supplier == PO supplier
  2. Currency match             — Invoice currency == PO currency
  3. Amount within tolerance    — Invoice total ≤ PO total × (1 + tolerance)
  4. PO reference on invoice    — Invoice references the correct PO number
  5. Date order                 — Invoice date after PO date (sanity check)
  6. Inspection cert present    — For PROCESS treatment: IC must exist in shipment
  7. Packing list present       — For PROCESS treatment: PL must exist in shipment

MATCH_TOLERANCE: invoices may exceed PO by up to 2% to allow for
freight/insurance uplift on CIF/DAP terms.

match_result:
  PASS         — all applicable checks pass
  PASS_PARTIAL — one or more checks SKIP'd (missing doc) but none FAIL
  FAIL         — one or more checks FAIL

In production: tolerance values are configurable per Document Class
in the portal settings. Field names used in each check are pulled from
the class extraction template's canonical field map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.models.extracted_field import ExtractedField
from app.models.shipment import ShipmentRecord
from app.pipeline.states import PipelineState

MATCH_TOLERANCE = Decimal("0.02")   # 2% uplift allowed — global default (overridden per class via config)

# Document classes that are considered POs
PO_CLASSES = {"dc_001", "dc_002", "dc_003"}
# Document classes that are invoices (inbound)
INVOICE_CLASSES = {"dc_006", "dc_011", "dc_012", "dc_013"}
# Document classes that are inspection certs
IC_CLASSES = {"dc_008"}
# Document classes that are packing lists
PL_CLASSES = {"dc_007"}


@dataclass
class CheckResult:
    name: str
    status: str     # PASS | FAIL | SKIP
    detail: str


@dataclass
class MatchResult:
    outcome: str                # PASS | PASS_PARTIAL | FAIL
    checks: list[CheckResult] = field(default_factory=list)
    po_document_id: str | None = None
    invoice_document_id: str | None = None
    summary: str = ""


class MatchingAgent(BaseAgent):
    """
    Runs the seven-check three-way match for the shipment containing this document.
    """

    name = "MatchingAgent"

    def match(self, doc: Document) -> MatchResult:
        if not doc.shipment_id:
            return MatchResult(
                outcome="SKIP",
                summary="Document not linked to a ShipmentRecord — no match possible.",
            )

        shipment = (
            self.db.query(ShipmentRecord)
            .filter(ShipmentRecord.id == doc.shipment_id)
            .first()
        )
        if not shipment:
            return MatchResult(
                outcome="SKIP",
                summary="ShipmentRecord not found — no match possible.",
            )

        # Load all docs in this shipment
        all_doc_ids = shipment.document_ids or []
        all_docs = (
            self.db.query(Document)
            .filter(Document.id.in_(all_doc_ids))
            .all()
        )

        # Find PO and Invoice documents
        po_doc = next((d for d in all_docs if d.document_class_id in PO_CLASSES), None)
        inv_doc = next((d for d in all_docs if d.document_class_id in INVOICE_CLASSES), None)
        has_ic = any(d.document_class_id in IC_CLASSES for d in all_docs)
        has_pl = any(d.document_class_id in PL_CLASSES for d in all_docs)

        if not po_doc or not inv_doc:
            return MatchResult(
                outcome="PASS_PARTIAL",
                summary=(
                    f"Three-way match deferred: shipment has "
                    f"{len(all_doc_ids)} document(s) but is missing a "
                    f"{'PO' if not po_doc else 'Invoice'}. "
                    "Match will run when the full set arrives."
                ),
                po_document_id=po_doc.id if po_doc else None,
                invoice_document_id=inv_doc.id if inv_doc else None,
            )

        po_fields  = _field_map(self.db, po_doc.id)
        inv_fields = _field_map(self.db, inv_doc.id)

        checks: list[CheckResult] = []

        # ── Check 1: Supplier name ────────────────────────────────────────────
        po_supplier  = po_fields.get("supplier_name", "")
        inv_supplier = inv_fields.get("supplier_name", inv_fields.get("freight_agent", ""))
        if po_supplier and inv_supplier:
            # Fuzzy: check if one contains the other (handles abbreviations)
            match_ok = (
                _normalise(po_supplier) in _normalise(inv_supplier)
                or _normalise(inv_supplier) in _normalise(po_supplier)
            )
            checks.append(CheckResult(
                name="supplier_name",
                status="PASS" if match_ok else "FAIL",
                detail=f"PO: '{po_supplier}' vs Invoice: '{inv_supplier}'.",
            ))
        else:
            checks.append(CheckResult(
                name="supplier_name",
                status="SKIP",
                detail="Supplier name missing from PO or Invoice.",
            ))

        # ── Check 2: Currency ─────────────────────────────────────────────────
        po_ccy  = po_fields.get("currency", "")
        inv_ccy = inv_fields.get("currency", "")
        if po_ccy and inv_ccy:
            checks.append(CheckResult(
                name="currency",
                status="PASS" if po_ccy.upper() == inv_ccy.upper() else "FAIL",
                detail=f"PO: {po_ccy} vs Invoice: {inv_ccy}.",
            ))
        else:
            checks.append(CheckResult(
                name="currency",
                status="SKIP",
                detail="Currency missing from PO or Invoice.",
            ))

        # ── Check 3: Amount within tolerance ──────────────────────────────────
        # Tolerance is configurable per document class in app.config.Settings.
        from app.config import settings
        _tol = Decimal(str(settings.match_tolerance_for(inv_doc.document_class_id)))

        po_amount  = _to_decimal(po_fields.get("total_order_value"))
        inv_amount = _to_decimal(
            inv_fields.get("total_amount")
            or inv_fields.get("total_invoice_value")
        )
        if po_amount and inv_amount:
            ceiling = po_amount * (1 + _tol)
            ok = inv_amount <= ceiling
            checks.append(CheckResult(
                name="amount_tolerance",
                status="PASS" if ok else "FAIL",
                detail=(
                    f"Invoice {inv_amount} vs PO {po_amount} "
                    f"(ceiling {ceiling:.2f}, tolerance {_tol:.0%}). "
                    + ("PASS" if ok else f"OVERAGE: {inv_amount - po_amount:.2f}")
                ),
            ))
        else:
            checks.append(CheckResult(
                name="amount_tolerance",
                status="SKIP",
                detail="Amount field(s) missing from PO or Invoice.",
            ))

        # ── Check 4: PO reference on invoice ─────────────────────────────────
        po_number = po_fields.get("po_number", "")
        inv_po_ref = (
            inv_fields.get("customer_po_ref")
            or inv_fields.get("customer_po_number")
            or inv_fields.get("tll_reference")
            or inv_fields.get("tll_po_number")
            or ""
        )
        if po_number and inv_po_ref:
            ref_ok = _normalise(po_number) in _normalise(inv_po_ref) or _normalise(inv_po_ref) in _normalise(po_number)
            checks.append(CheckResult(
                name="po_reference",
                status="PASS" if ref_ok else "FAIL",
                detail=f"PO number '{po_number}' vs invoice ref '{inv_po_ref}'.",
            ))
        else:
            checks.append(CheckResult(
                name="po_reference",
                status="SKIP",
                detail="PO number or invoice PO reference not found.",
            ))

        # ── Check 5: Date order (invoice after PO) ────────────────────────────
        po_date  = po_fields.get("po_date", "")
        inv_date = inv_fields.get("invoice_date", "")
        if po_date and inv_date:
            try:
                from datetime import date
                po_d  = date.fromisoformat(po_date)
                inv_d = date.fromisoformat(inv_date)
                ok = inv_d >= po_d
                checks.append(CheckResult(
                    name="date_order",
                    status="PASS" if ok else "FAIL",
                    detail=f"PO date {po_date}, Invoice date {inv_date}.",
                ))
            except ValueError:
                checks.append(CheckResult(
                    name="date_order",
                    status="SKIP",
                    detail="Could not parse PO date or Invoice date for comparison.",
                ))
        else:
            checks.append(CheckResult(
                name="date_order",
                status="SKIP",
                detail="PO date or Invoice date missing.",
            ))

        # ── Check 6: Inspection certificate present ───────────────────────────
        checks.append(CheckResult(
            name="inspection_cert",
            status="PASS" if has_ic else "SKIP",
            detail=(
                "Inspection Certificate found in shipment."
                if has_ic
                else "No Inspection Certificate yet received for this shipment."
            ),
        ))

        # ── Check 7: Packing list present ────────────────────────────────────
        checks.append(CheckResult(
            name="packing_list",
            status="PASS" if has_pl else "SKIP",
            detail=(
                "Packing List found in shipment."
                if has_pl
                else "No Packing List yet received for this shipment."
            ),
        ))

        # ── Determine outcome ─────────────────────────────────────────────────
        has_fail = any(c.status == "FAIL" for c in checks)
        has_skip = any(c.status == "SKIP" for c in checks)

        if has_fail:
            outcome = "FAIL"
        elif has_skip:
            outcome = "PASS_PARTIAL"
        else:
            outcome = "PASS"

        fail_names = [c.name for c in checks if c.status == "FAIL"]
        skip_names = [c.name for c in checks if c.status == "SKIP"]
        pass_count = sum(1 for c in checks if c.status == "PASS")

        summary = (
            f"Three-way match {outcome}. "
            f"{pass_count}/{len(checks)} checks passed. "
            + (f"FAIL: {', '.join(fail_names)}. " if fail_names else "")
            + (f"SKIP (awaiting docs): {', '.join(skip_names)}." if skip_names else "")
        )

        return MatchResult(
            outcome=outcome,
            checks=checks,
            po_document_id=po_doc.id,
            invoice_document_id=inv_doc.id,
            summary=summary,
        )


def _field_map(db: Session, document_id: str) -> dict[str, str]:
    fields = db.query(ExtractedField).filter(ExtractedField.document_id == document_id).all()
    return {
        f.field_name: (f.corrected_value if f.human_corrected else f.field_value or "")
        for f in fields
    }


def _normalise(s: str) -> str:
    return s.lower().replace(" ", "").replace("/", "").replace("-", "")


def _to_decimal(val: str | None) -> Decimal | None:
    if not val:
        return None
    try:
        return Decimal(val.strip().replace(",", ""))
    except InvalidOperation:
        return None
