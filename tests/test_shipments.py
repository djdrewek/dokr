"""
Tests: shipments endpoint — list and detail.

Response shape: {"total": int, "page": int, "page_size": int, "pages": int, "shipments": [...]}
"""

import io
import uuid
from fpdf import FPDF
from tests.conftest import V1, AUTH, wait_done


def uid() -> str:
    return str(uuid.uuid4())[:8]


def _submit_and_wait(client, dc: str, tag: str) -> dict:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    lines = {
        "dc_006": [
            f"INVOICE {tag}", f"Invoice No.: SHIP-{tag}",
            "Tata Steel Limited", "Purchaser: Tata Steel Limited",
            "VAT 0%", f"IBAN: GB99SHIP{tag.upper()}", "Bank: ShipBank",
        ],
        "dc_004": [
            f"HOUSE AIR WAYBILL {tag}", f"AWB No.: WAC-SHIP-{tag}",
            f"HAWB No.: SHIP{tag[:6]}", "MAWB No.: 932-SHIP-001",
            "Shipper: Test", "Consignee: Tata", "Departure: LHR", "Destination: DEL",
        ],
    }
    for line in lines.get(dc, [f"DOCUMENT {tag}"]):
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


# ── List ──────────────────────────────────────────────────────────────────────

def test_shipments_endpoint_returns_200(client):
    r = client.get(f"{V1}/shipments", headers=AUTH)
    assert r.status_code == 200


def test_shipments_response_shape(client):
    r = client.get(f"{V1}/shipments", headers=AUTH)
    body = r.json()
    assert "total" in body
    assert "shipments" in body
    assert "page" in body
    assert "page_size" in body
    assert isinstance(body["shipments"], list)


def test_shipments_total_matches_list(client):
    r = client.get(f"{V1}/shipments", headers=AUTH)
    body = r.json()
    assert len(body["shipments"]) <= body["total"]


def test_shipments_endpoint_requires_auth(client):
    r = client.get(f"{V1}/shipments")
    assert r.status_code == 401


def test_shipments_pagination(client):
    r = client.get(f"{V1}/shipments?page=1&page_size=1", headers=AUTH)
    assert r.status_code == 200
    assert len(r.json()["shipments"]) <= 1


# ── Detail ────────────────────────────────────────────────────────────────────

def test_shipment_detail_not_found(client):
    r = client.get(f"{V1}/shipments/00000000000000000000000000", headers=AUTH)
    assert r.status_code == 404


def test_shipment_detail_has_required_fields(client):
    """If any shipments exist, check they have required fields."""
    r = client.get(f"{V1}/shipments", headers=AUTH)
    shipments = r.json()["shipments"]
    if not shipments:
        return  # No shipments yet — skip field check

    s_id = shipments[0]["id"]
    r2 = client.get(f"{V1}/shipments/{s_id}", headers=AUTH)
    assert r2.status_code == 200
    body = r2.json()
    assert "id" in body
    assert "created_at" in body


# ── After pipeline ────────────────────────────────────────────────────────────

def test_completed_docs_may_create_shipments(client):
    """
    After completing a PROCESS-class document, verify the shipments endpoint
    is still healthy (may or may not create a ShipmentRecord depending on
    whether the document has a PO reference).
    """
    _submit_and_wait(client, "dc_006", uid())
    _submit_and_wait(client, "dc_004", uid())

    r = client.get(f"{V1}/shipments", headers=AUTH)
    assert r.status_code == 200
    # Total should be >= 0; endpoint must respond correctly
    assert isinstance(r.json()["total"], int)
