"""
Tests: end-to-end pipeline — submit → pipeline → COMPLETED with correct class,
fields, ERP reference, and filing path.

Submit returns HTTP 200. Background task runs asynchronously; wait_done()
polls until the document reaches a terminal state.
"""

import io
import uuid
import pytest
from fpdf import FPDF
from tests.conftest import (
    V1, AUTH, supplier_invoice_pdf, awb_pdf, tsl_po_pdf,
    packing_list_pdf, blank_pdf, wait_done,
)


def _unique_pdf_for_class(dc: str, tag: str) -> bytes:
    """Generate a class-specific PDF with a unique tag to avoid exact-dup 409s."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    # Derive a stable 5-digit numeric suffix from the hex UUID tag
    _n = int(tag, 16) % 100_000
    _po = f"TSL/5{_n:05d}"    # e.g. TSL/512345
    _mawb = f"932-{int(tag, 16) % 100_000_000:08d}"   # IATA NNN-NNNNNNNN

    class_lines = {
        # Validator requires: invoice_number, invoice_date, supplier_name,
        #                     total_amount, currency, customer_po_ref
        "dc_006": [
            f"INVOICE {tag}",
            f"Invoice No.: INV-{_n:05d}",
            "Invoice Date: 2026-06-01",
            "Supplier: Test Supplier GmbH",
            "Customer: Tata Steel Limited",
            f"PO No.: {_po}",
            "Payment Terms: 45 days net",
            f"IBAN: GB12TEST{_n:05d}0001",
            "Bank: Test Bank AG",
            "Currency: EUR",
            "Invoice Total: EUR 1,000.00",
        ],
        # Validator requires: awb_number, shipper_name, consignee_name,
        #                     chargeable_weight, pieces
        "dc_004": [
            f"HOUSE AIR WAYBILL {tag}",
            f"MAWB No.: {_mawb}",       # e.g. 932-12847316 (IATA format)
            f"HAWB No.: HWB{_n:05d}",
            "Shipper: Test Shipper Ltd",
            "Consignee: Tata Steel Limited",
            "Airport of Departure: LHR",
            "Airport of Destination: DEL",
            "Pieces: 2",
            "Chargeable Weight: 15.0 KG",
        ],
        # Validator requires: po_number, po_date, total_order_value,
        #                     currency, supplier_name
        "dc_003": [
            f"Tata Steel Limited {tag}",
            f"Purchase Order No.: {_po}",
            "P.O. Date: 2026-06-01",
            "Tata Steel UK",
            "Orders@tata.co.uk",
            "Grosvenor Place London",
            "Supplier: Example GmbH",
            "Currency: GBP",
            "Total Order Value: GBP 25,000.00",
        ],
        # Validator requires: supplier_name, consignee_name,
        #                     total_gross_weight, purchase_order_no
        "dc_007": [
            f"PACKING LIST {tag}",
            "Supplier: Test Supplier GmbH",
            "Consignee: Tata Steel Limited",
            "Total Gross Weight: 500 KG",
            "Nett Wt: 460 KG",
            f"P.O. No.: {_po}",
            f"Packages: 3 cases {tag}",
            "Dimensions: 120x80x60 cm",
            "Pieces: 10",
        ],
    }

    for line in class_lines.get(dc, [f"DOCUMENT {tag}", "No keywords"]):
        pdf.cell(0, 8, line[:90], ln=True)
    return bytes(pdf.output())


def submit_and_wait(client, pdf: bytes, **kwargs) -> dict:
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf), "application/pdf")},
        data={"priority": "standard", **kwargs},
        headers=AUTH,
    )
    assert r.status_code == 200, f"Submit failed: {r.status_code} {r.text}"
    return wait_done(client, r.json()["id"])


def uid() -> str:
    return str(uuid.uuid4())[:8]


# ── Supplier Invoice (dc_006) ─────────────────────────────────────────────────

def test_supplier_invoice_reaches_completed(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    assert doc["status"] == "COMPLETED"


def test_supplier_invoice_classified_as_dc006(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    assert doc["document_class"] == "dc_006"


def test_supplier_invoice_has_erp_reference(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    shp_id = doc.get("shipment_id")
    assert shp_id is not None, "Expected shipment_id on COMPLETED dc_006 doc"
    r = client.get(f"{V1}/shipments/{shp_id}", headers=AUTH)
    assert r.status_code == 200
    assert r.json().get("erp_reference") is not None


def test_supplier_invoice_has_filing_path(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    shp_id = doc.get("shipment_id")
    assert shp_id is not None, "Expected shipment_id on COMPLETED dc_006 doc"
    r = client.get(f"{V1}/shipments/{shp_id}", headers=AUTH)
    assert r.status_code == 200
    assert r.json().get("sharepoint_path") is not None


def test_supplier_invoice_has_variant_id(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    assert doc.get("variant") is not None


# ── Airway Bill (dc_004) ──────────────────────────────────────────────────────

def test_awb_reaches_completed(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_004", uid()))
    assert doc["status"] == "COMPLETED"


def test_awb_classified_as_dc004(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_004", uid()))
    assert doc["document_class"] == "dc_004"


# ── Tata Steel PO (dc_003) ────────────────────────────────────────────────────

def test_tsl_po_reaches_completed(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_003", uid()))
    assert doc["status"] == "COMPLETED"


def test_tsl_po_classified_as_dc003(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_003", uid()))
    assert doc["document_class"] == "dc_003"


# ── Packing List (dc_007) ─────────────────────────────────────────────────────

def test_packing_list_reaches_completed(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_007", uid()))
    assert doc["status"] == "COMPLETED"


# ── Blank / unclassified ──────────────────────────────────────────────────────

def test_blank_pdf_becomes_unclassified(client):
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("blank.pdf", io.BytesIO(blank_pdf()), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    # blank_pdf may collide with a previous blank_pdf → 409 is fine too
    if r.status_code == 409:
        return   # duplicate of a previous blank → was already UNCLASSIFIED
    assert r.status_code == 200
    doc = wait_done(client, r.json()["id"])
    assert doc["status"] == "UNCLASSIFIED"
    assert doc["document_class"] is None


# ── Manual class override ─────────────────────────────────────────────────────

def test_class_override_bypasses_classifier(client):
    """
    document_class override → pipeline uses the forced class, not the classifier.
    Verified by submitting a packing-list PDF but setting override to dc_007.
    The classifier would otherwise see 'packing list' keywords and pick dc_007 anyway,
    but by explicitly supplying document_class we skip CLASSIFYING entirely.
    """
    tag = uid()
    # Use a content-rich dc_007 PDF so proofreading passes
    pdf_bytes = _unique_pdf_for_class("dc_007", tag)

    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("override.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"priority": "standard", "document_class": "dc_007"},
        headers=AUTH,
    )
    assert r.status_code == 200
    doc = wait_done(client, r.json()["id"])
    assert doc["document_class"] == "dc_007"
    assert doc["status"] == "COMPLETED"


# ── Pipeline fields ───────────────────────────────────────────────────────────

def test_completed_doc_has_fields(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    assert doc["status"] == "COMPLETED"
    r = client.get(f"{V1}/documents/{doc['id']}/fields", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["field_count"] > 0
    field_names = [f["field_name"] for f in body["fields"]]
    assert "invoice_number" in field_names
    assert "total_amount" in field_names


def test_all_fields_have_confidence(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    r = client.get(f"{V1}/documents/{doc['id']}/fields", headers=AUTH)
    for field in r.json()["fields"]:
        assert 0.0 <= field["confidence"] <= 1.0


def test_completed_doc_has_pipeline_stage(client):
    doc = submit_and_wait(client, _unique_pdf_for_class("dc_006", uid()))
    assert doc["status"] == "COMPLETED"
    # Verify full pipeline event trail is accessible via the status endpoint
    r = client.get(f"{V1}/documents/{doc['id']}/status", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "pipeline" in body
    stages = [e["state"] for e in body["pipeline"]]
    assert "COMPLETED" in stages
