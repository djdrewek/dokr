"""
Linking Agent — builds and maintains the shipment document graph.

After validation passes, this agent extracts reference keys from the
document's fields and looks for an existing ShipmentRecord that shares
any of those keys. If found, the document is added to it. If not, a
new ShipmentRecord is created.

Reference priority (most → least specific):
  1. tata_po_number, po_number, customer_po_ref, customer_po_number, tll_po_number, tll_reference
  2. awb_number, mawb_number, hawb_number, awb_bl_number
  3. invoice_number (used only as a fallback — common across many docs)

A document is always linked to exactly one ShipmentRecord.

In production: this agent would publish a graph update event to Azure
Service Bus and use a distributed cache to handle concurrent arrivals of
the same shipment's documents.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.models.extracted_field import ExtractedField
from app.models.shipment import ShipmentRecord
from app.pipeline.states import PipelineState

# Field names in priority order for reference key extraction
REFERENCE_FIELDS_PRIORITY = [
    # PO-family (highest confidence linking key)
    "tata_po_number",
    "po_number",
    "customer_po_ref",
    "customer_po_number",
    "tll_reference",
    "tll_po_number",
    # Airway bill family
    "awb_number",
    "mawb_number",
    "hawb_number",
    "awb_bl_number",
    # Invoice as fallback
    "invoice_number",
    "confirmation_number",
]


@dataclass
class LinkingResult:
    shipment_id: str
    is_new_shipment: bool
    reference_key: str
    all_keys: list[str]
    document_count: int


class LinkingAgent(BaseAgent):
    """
    Assigns the document to a ShipmentRecord, creating one if needed.
    """

    name = "LinkingAgent"

    def link(self, doc: Document) -> LinkingResult:
        # Pull all extracted field values for this document
        fields = {
            ef.field_name: (ef.corrected_value if ef.human_corrected else ef.field_value)
            for ef in self.db.query(ExtractedField)
            .filter(ExtractedField.document_id == doc.id)
            .all()
        }

        # Collect all non-empty reference values, keeping priority order
        ref_keys = []
        for fname in REFERENCE_FIELDS_PRIORITY:
            val = fields.get(fname, "")
            if val and val.strip() and val.strip() not in ref_keys:
                ref_keys.append(val.strip())

        # Primary key = first (highest-priority) reference found
        primary_key = ref_keys[0] if ref_keys else f"unkeyed:{doc.id}"

        # Search for existing shipment by any of our reference keys
        existing = None
        for key in ref_keys:
            # Check if any existing shipment already holds this key
            candidates = (
                self.db.query(ShipmentRecord)
                .filter(ShipmentRecord.reference_key == key)
                .all()
            )
            if candidates:
                existing = candidates[0]
                break

            # Also check the all_reference_keys JSON array
            candidates_json = self.db.query(ShipmentRecord).all()
            for shp in candidates_json:
                if key in (shp.all_reference_keys or []):
                    existing = shp
                    break
            if existing:
                break

        from datetime import datetime
        from ulid import ULID

        is_new = existing is None

        if is_new:
            shipment = ShipmentRecord(
                id=f"shp_{ULID()}",
                reference_key=primary_key,
                all_reference_keys=ref_keys,
                document_ids=[doc.id],
                class_summary={doc.document_class_id: 1} if doc.document_class_id else {},
                status="OPEN",
            )
            self.db.add(shipment)
        else:
            shipment = existing
            # Merge reference keys
            merged_keys = list(shipment.all_reference_keys or [])
            for k in ref_keys:
                if k not in merged_keys:
                    merged_keys.append(k)
            shipment.all_reference_keys = merged_keys

            # Add document to list
            doc_ids = list(shipment.document_ids or [])
            if doc.id not in doc_ids:
                doc_ids.append(doc.id)
            shipment.document_ids = doc_ids

            # Update class summary
            summary = dict(shipment.class_summary or {})
            cls = doc.document_class_id or "unknown"
            summary[cls] = summary.get(cls, 0) + 1
            shipment.class_summary = summary
            shipment.updated_at = datetime.utcnow()

        self.db.flush()

        # Write shipment_id back to the document
        doc.shipment_id = shipment.id
        self.db.commit()

        return LinkingResult(
            shipment_id=shipment.id,
            is_new_shipment=is_new,
            reference_key=primary_key,
            all_keys=ref_keys,
            document_count=len(shipment.document_ids),
        )
