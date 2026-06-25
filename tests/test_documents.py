"""
Tests: document submission, listing, retrieval, and validation errors.

Note: Submit endpoint returns HTTP 200 (not 202). Exact duplicates are
rejected immediately with HTTP 409 at submit time.
"""

import io
import time
import uuid
import pytest
from fpdf import FPDF
from tests.conftest import V1, AUTH, supplier_invoice_pdf, awb_pdf, blank_pdf, wait_done


def _unique_pdf(tag: str) -> bytes:
    """Generate a unique supplier-invoice-like PDF using a UUID tag so each
    test submission doesn't collide with others via the exact-dup check."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in [
        f"INVOICE {tag}",
        f"Invoice No.: INV-{tag}",
        "Tata Steel Limited",
        "Purchaser: Tata Steel Limited",
        "Payment Terms: 45 days",
        f"IBAN: GB12{tag[:8].upper()}0001",
        "VAT 0%",
        "Bank: Test Bank AG",
        "Total: EUR 1000.00",
    ]:
        pdf.cell(0, 8, line[:90], ln=True)
    return bytes(pdf.output())


def submit(client, pdf: bytes, priority: str = "standard", **extra_data):
    data = {"priority": priority, **extra_data}
    return client.post(
        f"{V1}/documents/submit",
        files={"file": ("test.pdf", io.BytesIO(pdf), "application/pdf")},
        data=data,
        headers=AUTH,
    )


# ── Submit ────────────────────────────────────────────────────────────────────

def test_submit_returns_200_with_id(client):
    r = submit(client, _unique_pdf(str(uuid.uuid4())[:8]))
    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["status"] == "RECEIVED"
    assert body["priority"] == "standard"


def test_submit_express_priority(client):
    r = submit(client, _unique_pdf(str(uuid.uuid4())[:8]), priority="express")
    assert r.status_code == 200
    assert r.json()["priority"] == "express"


def test_submit_invalid_priority(client):
    r = submit(client, _unique_pdf(str(uuid.uuid4())[:8]), priority="turbo")
    assert r.status_code == 422


def test_submit_non_pdf_rejected(client):
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    assert r.status_code == 422
    assert "invalid_file_type" in r.text


def test_submit_empty_file_rejected(client):
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    assert r.status_code == 422


def test_submit_non_pdf_bytes_rejected(client):
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("fake.pdf", io.BytesIO(b"NOT A PDF AT ALL"), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    assert r.status_code == 422
    assert "invalid_pdf" in r.text


def test_submit_with_metadata(client):
    """Extra key-value metadata can be attached to a document via the metadata field."""
    import json as _json
    r = submit(
        client,
        _unique_pdf(str(uuid.uuid4())[:8]),
        metadata=_json.dumps({"reference": "PO-TEST-REF-001"}),
    )
    assert r.status_code == 200
    assert r.json()["metadata"]["reference"] == "PO-TEST-REF-001"


def test_exact_duplicate_returns_409(client):
    """Submitting the same PDF bytes twice → second returns 409 immediately."""
    pdf = _unique_pdf("DEDUP-DOCS-001")
    r1 = submit(client, pdf)
    assert r1.status_code == 200
    r2 = submit(client, pdf)
    assert r2.status_code == 409
    body = r2.json()
    assert body["detail"]["error"] == "exact_duplicate"
    assert body["detail"]["original_document_id"] == r1.json()["id"]


# ── List ─────────────────────────────────────────────────────────────────────

def test_list_documents(client):
    r = client.get(f"{V1}/documents/", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body
    assert "total" in body
    assert isinstance(body["documents"], list)


def test_list_documents_pagination(client):
    r = client.get(f"{V1}/documents/?page=1&page_size=2", headers=AUTH)
    assert r.status_code == 200
    assert len(r.json()["documents"]) <= 2


def test_list_documents_filter_by_status(client):
    r = client.get(f"{V1}/documents/?status=COMPLETED", headers=AUTH)
    assert r.status_code == 200
    for item in r.json()["documents"]:
        assert item["status"] == "COMPLETED"


# ── Get ───────────────────────────────────────────────────────────────────────

def test_get_document_by_id(client):
    r = submit(client, _unique_pdf(str(uuid.uuid4())[:8]))
    doc_id = r.json()["id"]
    r2 = client.get(f"{V1}/documents/{doc_id}", headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["id"] == doc_id


def test_get_nonexistent_document(client):
    r = client.get(f"{V1}/documents/00000000000000000000000000", headers=AUTH)
    assert r.status_code == 404


# ── Fields endpoint ───────────────────────────────────────────────────────────

def test_fields_endpoint_after_pipeline(client):
    """After pipeline completes the fields endpoint returns field_count > 0."""
    r = submit(client, _unique_pdf(str(uuid.uuid4())[:8]))
    doc_id = r.json()["id"]
    doc = wait_done(client, doc_id)
    if doc["status"] == "UNCLASSIFIED":
        pytest.skip("Document was not classified — fields not available")
    r2 = client.get(f"{V1}/documents/{doc_id}/fields", headers=AUTH)
    assert r2.status_code == 200
    body = r2.json()
    assert "field_count" in body
    assert body["field_count"] > 0
    assert "fields" in body
    assert isinstance(body["fields"], list)


def test_fields_endpoint_on_unclassified_returns_409(client):
    """Blank PDF → UNCLASSIFIED → fields endpoint → 409."""
    r = submit(client, blank_pdf())
    doc_id = r.json()["id"]
    wait_done(client, doc_id)
    r2 = client.get(f"{V1}/documents/{doc_id}/fields", headers=AUTH)
    assert r2.status_code == 409
