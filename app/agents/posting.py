"""
Posting Agent — submits the document payload to the ERP system.

Target systems (configured per document class in portal settings):
  - Business Central (BC): REST API, journal line creation
  - Xero:                   REST API, invoice/bill creation
  - SAP:                    BAPI calls (future)

For the scaffold: the agent builds a realistic ERP payload, logs it to
the pipeline event trail, and stores a synthetic reference number on the
ShipmentRecord. No live HTTP call is made — swap _post_to_erp() for a
real httpx call when BC/Xero credentials are configured.

ERP routing by document class treatment:
  - PROCESS treatment classes → post to ERP
  - STORE treatment classes → skip (shouldn't reach this agent)

Payload shape follows Business Central's Purchase Invoice API contract.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document, DocumentClass
from app.models.extracted_field import ExtractedField
from app.models.shipment import ShipmentRecord
from app.pipeline.states import PipelineState


@dataclass
class PostingResult:
    success: bool
    erp_reference: str | None
    payload: dict
    detail: str


class PostingAgent(BaseAgent):
    """
    Builds an ERP payload from extracted fields and posts it.
    """

    name = "PostingAgent"

    def post(self, doc: Document) -> PostingResult:
        fields = _field_map(self.db, doc.id)
        class_id = doc.document_class_id or "unknown"

        # Build payload based on document class family
        if class_id in ("dc_006", "dc_013"):
            payload = _build_purchase_invoice_payload(doc, fields, class_id)
        elif class_id in ("dc_011", "dc_012"):
            payload = _build_sales_invoice_payload(doc, fields, class_id)
        elif class_id in ("dc_001", "dc_002", "dc_003"):
            payload = _build_po_acknowledgement_payload(doc, fields, class_id)
        else:
            payload = _build_generic_payload(doc, fields, class_id)

        # Attempt live ERP post if BC credentials are configured; else use stub
        from app.config import settings
        if settings.bc_api_url and settings.bc_api_key:
            erp_ref, detail = _post_to_bc(payload, settings.bc_api_url, settings.bc_api_key, settings.bc_company, class_id)
            if erp_ref is None:
                # Live call failed — return failure so runner routes to NEEDS_REVIEW
                return PostingResult(
                    success=False,
                    erp_reference=None,
                    payload=payload,
                    detail=detail,
                )
        else:
            # Stub mode: generate synthetic reference, log that no credentials are set
            erp_ref = f"ERP-{class_id.upper()}-{uuid.uuid4().hex[:8].upper()}"
            detail = (
                f"ERP posting stubbed (no BC credentials configured). "
                f"Synthetic reference: {erp_ref}. "
                f"Payload: {len(json.dumps(payload))} bytes. "
                f"Document class: {class_id}. "
                "Set BC_API_URL + BC_API_KEY in .env to enable live posting."
            )

        # Update ShipmentRecord if linked
        if doc.shipment_id:
            shipment = (
                self.db.query(ShipmentRecord)
                .filter(ShipmentRecord.id == doc.shipment_id)
                .first()
            )
            if shipment:
                shipment.erp_reference = erp_ref
                shipment.erp_posted_at = datetime.utcnow()
                shipment.status = "POSTED"
                self.db.commit()

        return PostingResult(
            success=True,
            erp_reference=erp_ref,
            payload=payload,
            detail=detail,
        )


# ── Payload builders ──────────────────────────────────────────────────────────

def _build_purchase_invoice_payload(doc: Document, fields: dict, class_id: str) -> dict:
    return {
        "type": "Purchase Invoice",
        "documentType": "Invoice",
        "vendorNumber": fields.get("supplier_vat", "") or fields.get("supplier_name", ""),
        "vendorInvoiceNumber": fields.get("invoice_number", ""),
        "invoiceDate": fields.get("invoice_date", ""),
        "postingDate": datetime.utcnow().strftime("%Y-%m-%d"),
        "currencyCode": fields.get("currency", "EUR"),
        "amount": fields.get("total_amount", ""),
        "vatAmount": fields.get("vat_amount", "0.00"),
        "paymentTermsCode": fields.get("payment_terms", "NET30"),
        "shipToAddress": fields.get("delivery_address", ""),
        "purchaseOrderNumber": fields.get("customer_po_ref", ""),
        "externalDocumentNumber": doc.id,
        "dokrDocumentId": doc.id,
        "dokrClass": class_id,
    }


def _build_sales_invoice_payload(doc: Document, fields: dict, class_id: str) -> dict:
    return {
        "type": "Sales Invoice",
        "documentType": "Invoice",
        "customerNumber": fields.get("customer_name", ""),
        "invoiceNumber": fields.get("invoice_number", ""),
        "invoiceDate": fields.get("invoice_date", ""),
        "postingDate": datetime.utcnow().strftime("%Y-%m-%d"),
        "currencyCode": fields.get("currency", "GBP"),
        "totalAmount": fields.get("total_invoice_value", ""),
        "buyingCommissionPct": fields.get("buying_commission_pct", ""),
        "awbBlNumber": fields.get("awb_bl_number", ""),
        "portOfLoading": fields.get("port_of_loading", ""),
        "portOfDischarge": fields.get("port_of_discharge", ""),
        "externalDocumentNumber": doc.id,
        "dokrDocumentId": doc.id,
        "dokrClass": class_id,
    }


def _build_po_acknowledgement_payload(doc: Document, fields: dict, class_id: str) -> dict:
    return {
        "type": "Purchase Order Update",
        "documentType": "PO Acknowledgement",
        "purchaseOrderNumber": fields.get("po_number", ""),
        "vendorNumber": fields.get("supplier_name", ""),
        "confirmedDeliveryDate": fields.get("delivery_date", ""),
        "currencyCode": fields.get("currency", "EUR"),
        "confirmedAmount": fields.get("total_order_value", ""),
        "externalDocumentNumber": doc.id,
        "dokrDocumentId": doc.id,
        "dokrClass": class_id,
    }


def _build_generic_payload(doc: Document, fields: dict, class_id: str) -> dict:
    return {
        "type": "Document Notification",
        "documentClass": class_id,
        "documentId": doc.id,
        "fields": fields,
        "postingDate": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def _post_to_bc(
    payload: dict,
    bc_api_url: str,
    bc_api_key: str,
    bc_company: str,
    class_id: str,
) -> tuple[str | None, str]:
    """
    POST payload to Business Central REST API.

    Returns (erp_reference, detail_message).
    erp_reference is None on failure.

    BC purchase invoice endpoint:
      POST /companies(name='{company}')/purchaseInvoices
    BC sales invoice endpoint:
      POST /companies(name='{company}')/salesInvoices

    The BC API returns the created record including its 'number' field
    which we use as the ERP reference.
    """
    import httpx

    doc_type = payload.get("type", "")
    if "Purchase" in doc_type:
        endpoint = f"{bc_api_url}/companies('{bc_company}')/purchaseInvoices"
    elif "Sales" in doc_type:
        endpoint = f"{bc_api_url}/companies('{bc_company}')/salesInvoices"
    else:
        endpoint = f"{bc_api_url}/companies('{bc_company}')/generalLedgerEntries"

    headers = {
        "Authorization": f"Bearer {bc_api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Dokr/1.0",
    }

    try:
        resp = httpx.post(endpoint, json=payload, headers=headers, timeout=15)
        if resp.is_success:
            data = resp.json()
            erp_ref = data.get("number") or data.get("id") or f"BC-{class_id.upper()}-{uuid.uuid4().hex[:8].upper()}"
            return erp_ref, (
                f"ERP posting successful (live BC). Reference: {erp_ref}. "
                f"HTTP {resp.status_code}. Endpoint: {endpoint}."
            )
        else:
            return None, (
                f"ERP posting failed. HTTP {resp.status_code} from {endpoint}. "
                f"Response: {resp.text[:200]}."
            )
    except httpx.TimeoutException:
        return None, f"ERP posting timeout (>15s). Endpoint: {endpoint}."
    except Exception as exc:
        return None, f"ERP posting exception: {type(exc).__name__}: {exc}. Endpoint: {endpoint}."


def _field_map(db: Session, document_id: str) -> dict[str, str]:
    fields = db.query(ExtractedField).filter(ExtractedField.document_id == document_id).all()
    return {
        f.field_name: (f.corrected_value if f.human_corrected else f.field_value or "")
        for f in fields
    }
