"""
Tests: deduplication — exact duplicates (instant 409), near-duplicates, amendments.
"""

import io
import uuid
from fpdf import FPDF
from tests.conftest import V1, AUTH, supplier_invoice_pdf, awb_pdf, wait_done


def _make_pdf(lines: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.cell(0, 8, line[:90], ln=True)
    return bytes(pdf.output())


def _submit(client, pdf: bytes, **kwargs):
    data = {"priority": "standard", **kwargs}
    return client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf), "application/pdf")},
        data=data,
        headers=AUTH,
    )


def uid() -> str:
    return str(uuid.uuid4())[:8]


# ── Exact duplicates ──────────────────────────────────────────────────────────

def test_exact_duplicate_returns_409(client):
    """
    Submitting the identical bytes twice → second submit returns HTTP 409
    immediately with error=exact_duplicate.
    """
    tag = uid()
    pdf = _make_pdf([
        f"INVOICE {tag}", f"Invoice No.: DEDUP-{tag}",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", "Payment Terms: 30 days",
        f"IBAN: GB99{tag.upper()}", "Bank: Test Bank"
    ])
    r1 = _submit(client, pdf)
    assert r1.status_code == 200
    original_id = r1.json()["id"]

    r2 = _submit(client, pdf)
    assert r2.status_code == 409
    body = r2.json()
    assert body["detail"]["error"] == "exact_duplicate"
    assert body["detail"]["original_document_id"] == original_id


def test_exact_duplicate_response_includes_original_status(client):
    tag = uid()
    pdf = _make_pdf([
        f"INVOICE {tag}", f"Invoice No.: DEDUP2-{tag}",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", f"IBAN: GB99{tag.upper()}X", "Bank: DupBank"
    ])
    r1 = _submit(client, pdf)
    assert r1.status_code == 200

    r2 = _submit(client, pdf)
    assert r2.status_code == 409
    detail = r2.json()["detail"]
    assert "original_status" in detail


def test_exact_duplicate_doesnt_affect_first_doc(client):
    """The original document should still be queryable after a dup rejection."""
    tag = uid()
    pdf = _make_pdf([
        f"INVOICE {tag}", f"Invoice No.: DEDUP3-{tag}",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", f"IBAN: GB99{tag.upper()}Y", "Bank: OrigBank"
    ])
    r1 = _submit(client, pdf)
    original_id = r1.json()["id"]

    _submit(client, pdf)  # 409 — ignore

    r3 = client.get(f"{V1}/documents/{original_id}", headers=AUTH)
    assert r3.status_code == 200
    assert r3.json()["id"] == original_id


# ── Near-duplicates ───────────────────────────────────────────────────────────

def test_near_duplicate_pipeline_states(client):
    """
    Two PDFs with nearly identical content should both be accepted (200) and
    the second may end up CONTENT_DUPLICATE, NEAR_DUPLICATE, or COMPLETED.
    """
    tag = uid()
    base_lines = [
        f"INVOICE {tag}", f"Invoice No.: NEAR-{tag}",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", "Payment Terms: 30 days",
        f"IBAN: GB99NEAR{tag.upper()}", "Bank: Near Dup Bank",
        "Amount: EUR 1000"
    ]
    variant_lines = base_lines[:-1] + ["Amount: EUR 1001"]  # tiny change

    r1 = _submit(client, _make_pdf(base_lines))
    assert r1.status_code == 200
    wait_done(client, r1.json()["id"])

    r2 = _submit(client, _make_pdf(variant_lines))
    assert r2.status_code == 200
    doc2 = wait_done(client, r2.json()["id"])
    assert doc2["status"] in {
        "COMPLETED", "CONTENT_DUPLICATE", "NEAR_DUPLICATE", "NEEDS_REVIEW"
    }


# ── Amendments ────────────────────────────────────────────────────────────────

def test_amendment_accepted_and_linked(client):
    """
    Submitting a revised document with distinct bytes is accepted without dup
    rejection and processes to a terminal state independently of the original.
    """
    tag = uid()
    pdf1 = _make_pdf([
        f"INVOICE {tag}", f"Invoice No.: AMEND-{tag}",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", f"IBAN: GB99AMEND{tag.upper()}", "Bank: AmendBank",
        "Total: EUR 500"
    ])
    r1 = _submit(client, pdf1)
    assert r1.status_code == 200
    original_id = r1.json()["id"]
    wait_done(client, original_id)

    tag2 = uid()
    pdf2 = _make_pdf([
        f"INVOICE {tag}-REV", f"Invoice No.: AMEND-{tag}-REV",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", f"IBAN: GB99AMEND{tag2.upper()}", "Bank: AmendBankRev",
        "Total: EUR 550"
    ])
    r2 = _submit(client, pdf2)
    assert r2.status_code == 200
    amended_id = r2.json()["id"]
    assert amended_id != original_id
    doc = wait_done(client, amended_id)
    # Amended docs have distinct bytes — should not be rejected as exact dup
    assert doc["status"] in {"COMPLETED", "CONTENT_DUPLICATE", "NEAR_DUPLICATE", "NEEDS_REVIEW"}


# ── Different documents both complete ─────────────────────────────────────────

def test_different_document_types_dont_dedup(client):
    """An invoice PDF and an AWB PDF are genuinely different and both complete."""
    tag1, tag2 = uid(), uid()

    # Rich enough to pass proofreading AND validation:
    # dc_006 requires invoice_number, invoice_date, supplier_name,
    # total_amount, currency, customer_po_ref — all present below.
    _n1 = int(tag1, 16) % 10000
    inv_pdf = _make_pdf([
        f"INVOICE {tag1}",
        f"Invoice No.: DIFF-INV-{tag1.upper()}",
        "Invoice Date: 2026-06-01",
        "Supplier: DiffSupplier GmbH",
        "Customer: Tata Steel Limited",
        f"PO No.: TSL/5{_n1:04d}",
        "Payment Terms: 30 days net",
        f"IBAN: GB99DIFF{tag1.upper()}",
        "Bank: DiffBank1",
        "Currency: EUR",
        "Invoice Total: EUR 12500.00",
    ])
    # Rich enough: awb_number (IATA format) + shipper_name + consignee_name
    # are the 3 required dc_004 fields.
    awb_pdf_bytes = _make_pdf([
        f"HOUSE AIR WAYBILL {tag2}",
        "MAWB No.: 932-00000991",
        f"HAWB No.: HWB{tag2.upper()}",
        "Shipper: DiffShipper Ltd",
        "Consignee: Tata Steel Limited",
        "Departure Airport: LHR",
        "Destination Airport: DEL",
        "Pieces: 4",
        "Gross Weight: 250 KG",
    ])

    r1 = _submit(client, inv_pdf)
    r2 = _submit(client, awb_pdf_bytes)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["id"] != r2.json()["id"]

    doc1 = wait_done(client, r1.json()["id"])
    doc2 = wait_done(client, r2.json()["id"])
    assert doc1["status"] == "COMPLETED"
    assert doc2["status"] == "COMPLETED"
