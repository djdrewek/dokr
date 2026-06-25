"""
Tests: variant discovery and the variants listing endpoint.

Response shape: {"total": int, "variants": [...]}
"""

import io
import uuid
from fpdf import FPDF
from tests.conftest import V1, AUTH, wait_done, rich_invoice_pdf


def uid() -> str:
    return str(uuid.uuid4())[:8]


def _submit_and_wait(client, lines: list[str]) -> dict:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.cell(0, 8, line[:90], ln=True)
    pdf_bytes = bytes(pdf.output())

    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    assert r.status_code == 200
    return wait_done(client, r.json()["id"])


def _submit_bytes_and_wait(client, pdf_bytes: bytes) -> dict:
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    assert r.status_code == 200
    return wait_done(client, r.json()["id"])


# ── Variants list ─────────────────────────────────────────────────────────────

def test_variants_endpoint_returns_200(client):
    r = client.get(f"{V1}/variants", headers=AUTH)
    assert r.status_code == 200


def test_variants_response_shape(client):
    r = client.get(f"{V1}/variants", headers=AUTH)
    body = r.json()
    assert "total" in body
    assert "variants" in body
    assert isinstance(body["variants"], list)


def test_variants_total_matches_list(client):
    r = client.get(f"{V1}/variants", headers=AUTH)
    body = r.json()
    assert body["total"] == len(body["variants"])


def test_variants_endpoint_requires_auth(client):
    r = client.get(f"{V1}/variants")
    assert r.status_code == 401


# ── Variant assigned to document ──────────────────────────────────────────────

def test_completed_document_has_variant(client):
    tag = uid()
    # Use rich_invoice_pdf — has all ValidationAgent-required dc_006 fields:
    # invoice_number, invoice_date, supplier_name, total_amount, currency, customer_po_ref
    doc = _submit_bytes_and_wait(client, rich_invoice_pdf(tag))
    assert doc["status"] == "COMPLETED"
    assert doc.get("variant") is not None


def test_variant_appears_in_list(client):
    """After completing a doc, its variant should appear in GET /variants."""
    tag = uid()
    doc = _submit_bytes_and_wait(client, rich_invoice_pdf(tag))
    assert doc.get("variant") is not None

    r = client.get(f"{V1}/variants", headers=AUTH)
    variant_ids = [v["id"] for v in r.json()["variants"]]
    assert doc["variant"] in variant_ids


def test_variant_has_learning_stage(client):
    """Each variant should have a learning_stage field."""
    r = client.get(f"{V1}/variants", headers=AUTH)
    variants = r.json()["variants"]
    assert len(variants) >= 1
    for v in variants:
        assert "learning_stage" in v
        assert v["learning_stage"] in ("ZERO_SHOT", "LEARNING", "LEARNED", "OPTIMISED")


def test_filter_variants_by_class(client):
    """GET /variants?document_class_id=dc_006 returns only dc_006 variants."""
    r = client.get(f"{V1}/variants?document_class_id=dc_006", headers=AUTH)
    assert r.status_code == 200
    for v in r.json()["variants"]:
        assert v["document_class_id"] == "dc_006"


def test_variant_detail_endpoint(client):
    """GET /variants/{id} returns the full variant record."""
    r = client.get(f"{V1}/variants", headers=AUTH)
    variants = r.json()["variants"]
    if not variants:
        return

    v_id = variants[0]["id"]
    r2 = client.get(f"{V1}/variants/{v_id}", headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["id"] == v_id


def test_variant_not_found(client):
    r = client.get(f"{V1}/variants/00000000000000000000000000", headers=AUTH)
    assert r.status_code == 404
