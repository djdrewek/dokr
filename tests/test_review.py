"""
Tests: NEEDS_REVIEW queue and retry/approve endpoints.

Review queue: GET /v1/review/  → {"total", "page", "page_size", "pages", "items"}
Retry:        POST /v1/review/{id}/retry
Approve:      POST /v1/review/{id}/approve
"""

import io
import uuid
from tests.conftest import V1, AUTH, wait_done


def uid() -> str:
    return str(uuid.uuid4())[:8]


def _unique_invoice(client, tag: str):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in [
        f"INVOICE {tag}", f"Invoice No.: REV-{tag}",
        "Tata Steel Limited", "Purchaser: Tata Steel Limited",
        "VAT 0%", f"IBAN: GB99REV{tag.upper()}", "Bank: ReviewBank",
    ]:
        pdf.cell(0, 8, line[:90], ln=True)
    pdf_bytes = bytes(pdf.output())
    return client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )


def _create_require_approval(client) -> int:
    r = client.post(
        f"{V1}/instructions",
        json={"action": "REQUIRE_APPROVAL", "document_class_id": "dc_006", "active": True},
        headers=AUTH,
    )
    assert r.status_code == 201
    return r.json()["id"]


def _deactivate(client, instr_id: int):
    client.patch(f"{V1}/instructions/{instr_id}", json={"active": False}, headers=AUTH)


# ── Review queue ──────────────────────────────────────────────────────────────

def test_review_queue_returns_200(client):
    r = client.get(f"{V1}/review/", headers=AUTH)
    assert r.status_code == 200


def test_review_queue_response_shape(client):
    r = client.get(f"{V1}/review/", headers=AUTH)
    body = r.json()
    assert "total" in body
    assert "items" in body
    assert "page" in body
    assert isinstance(body["items"], list)


def test_review_queue_requires_auth(client):
    r = client.get(f"{V1}/review/")
    assert r.status_code == 401


# ── Force document into NEEDS_REVIEW ─────────────────────────────────────────

def test_needs_review_document_appears_in_queue(client):
    instr_id = _create_require_approval(client)
    try:
        r = _unique_invoice(client, uid())
        assert r.status_code == 200
        doc = wait_done(client, r.json()["id"])
        assert doc["status"] == "NEEDS_REVIEW"

        r2 = client.get(f"{V1}/review/", headers=AUTH)
        queue_ids = [item["id"] for item in r2.json()["items"]]
        assert doc["id"] in queue_ids
    finally:
        _deactivate(client, instr_id)


# ── Retry ─────────────────────────────────────────────────────────────────────

def test_retry_endpoint(client):
    """POST /documents/{id}/retry re-queues a FAILED document."""
    instr_id = _create_require_approval(client)
    try:
        r = _unique_invoice(client, uid())
        assert r.status_code == 200
        doc_id = r.json()["id"]
        doc = wait_done(client, doc_id)
        assert doc["status"] == "NEEDS_REVIEW"

        # Deactivate instruction first, then reject to move to FAILED
        _deactivate(client, instr_id)
        instr_id = None

        # Reject the document to put it in FAILED state
        r_reject = client.post(
            f"{V1}/review/{doc_id}/reject",
            json={"rejected_by": "test@tata.co.uk", "reason": "test rejection for retry"},
            headers=AUTH,
        )
        assert r_reject.status_code == 200

        # Now retry from FAILED
        r2 = client.post(f"{V1}/documents/{doc_id}/retry", headers=AUTH)
        assert r2.status_code == 200
        body = r2.json()
        assert "document_id" in body
        assert body["document_id"] == doc_id
        assert body["action"] == "retried"
    finally:
        if instr_id:
            _deactivate(client, instr_id)


def test_retry_nonexistent_doc_returns_404(client):
    r = client.post(f"{V1}/documents/00000000000000000000000000/retry", headers=AUTH)
    assert r.status_code == 404


def test_retry_requires_auth(client):
    r = client.post(f"{V1}/documents/any-id/retry")
    assert r.status_code == 401


# ── Approve ───────────────────────────────────────────────────────────────────

def test_approve_needs_review_document(client):
    """POST /review/{id}/approve re-queues to a specified stage."""
    instr_id = _create_require_approval(client)
    doc_id = None
    try:
        r = _unique_invoice(client, uid())
        assert r.status_code == 200
        doc_id = r.json()["id"]
        doc = wait_done(client, doc_id)
        assert doc["status"] == "NEEDS_REVIEW"

        _deactivate(client, instr_id)
        instr_id = None

        r2 = client.post(
            f"{V1}/review/{doc_id}/approve",
            json={"target_stage": "COMPLETED", "approved_by": "test@tata.co.uk"},
            headers=AUTH,
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["action"] == "approved"
        assert body["document_id"] == doc_id
    finally:
        if instr_id:
            _deactivate(client, instr_id)


# ── Human correction ──────────────────────────────────────────────────────────

def test_human_correction_endpoint(client):
    """PATCH /documents/{id}/fields/{field_name}/correct stores a human correction."""
    from tests.conftest import rich_invoice_pdf
    tag = uid()
    pdf_bytes = rich_invoice_pdf(tag)

    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    assert r.status_code == 200
    doc_id = r.json()["id"]
    doc = wait_done(client, doc_id)
    assert doc["status"] == "COMPLETED"

    r2 = client.patch(
        f"{V1}/documents/{doc_id}/fields/invoice_number/correct",
        json={"corrected_value": "CORRECTED-99999", "corrected_by": "test@tata.co.uk"},
        headers=AUTH,
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["corrected_value"] == "CORRECTED-99999"
    assert body["corrected_by"] == "test@tata.co.uk"
