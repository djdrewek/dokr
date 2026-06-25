"""
ShipmentRecord — groups all documents that belong to the same physical shipment.

A shipment is identified by a shared reference key derived from the document's
extracted fields (typically the Tata PO number, AWB, or invoice reference).

One shipment record can accumulate many documents over time as they arrive:
  PO → DCC → Packing List → AWB → Inspection Cert → Supplier Invoice
  → Insurance Cert → FTA Cert → Test Cert → Freight Invoice → Bill of Entry
  → TLL Sales Invoice → TLL A2 Invoice

The MatchingAgent runs once the minimum required set for three-way matching
is present: a PO (dc_001/002/003) + at least one Invoice (dc_006/011).

match_result:
  None         — match not yet run (waiting for full document set)
  PASS         — all checks within tolerance (IGO)
  PASS_PARTIAL — some checks bypassed (missing docs) — supervisor flag
  FAIL         — one or more checks outside tolerance (NIGO)
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ShipmentRecord(Base):
    __tablename__ = "shipment_records"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # Format: shp_<ULID>

    # The primary linking key used to group documents (e.g. "TSL/58237")
    reference_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # All reference keys that map to this shipment (PO + AWB + invoice refs)
    all_reference_keys: Mapped[list] = mapped_column(JSON, default=list)

    # Ordered list of document IDs linked to this shipment
    document_ids: Mapped[list] = mapped_column(JSON, default=list)

    # Counts of each document class received
    class_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    # e.g. {"dc_001": 1, "dc_006": 2, "dc_007": 1}

    # Three-way match outcome
    match_result: Mapped[str | None] = mapped_column(String, nullable=True)
    # None | "PASS" | "PASS_PARTIAL" | "FAIL"

    match_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Structured match report per check

    # ERP posting reference (populated by PostingAgent)
    erp_reference: Mapped[str | None] = mapped_column(String, nullable=True)
    erp_posted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # SharePoint filing path (populated by FilingAgent)
    sharepoint_path: Mapped[str | None] = mapped_column(String, nullable=True)
    sharepoint_filed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Status of the shipment record overall
    status: Mapped[str] = mapped_column(String, default="OPEN")
    # OPEN | MATCHED | POSTED | FILED | COMPLETE

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
