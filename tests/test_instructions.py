"""
Tests: instruction CRUD and pipeline effects.

The instructions schema uses `action` (not `instruction_type`) for the
instruction type field. Response shape: {"total": int, "instructions": [...]}
"""

import io
import uuid
from tests.conftest import V1, AUTH, supplier_invoice_pdf, wait_done


def uid() -> str:
    return str(uuid.uuid4())[:8]


def _unique_invoice(client, tag: str) -> dict:
    """Submit a unique supplier invoice (all required fields) and return the response."""
    from tests.conftest import rich_invoice_pdf
    pdf_bytes = rich_invoice_pdf(tag)
    r = client.post(
        f"{V1}/documents/submit",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"priority": "standard"},
        headers=AUTH,
    )
    return r


def create_instruction(client, action: str, document_class_id: str = "dc_006",
                       **extra) -> dict:
    payload = {
        "action": action,
        "document_class_id": document_class_id,
        "active": True,
        **extra,
    }
    r = client.post(f"{V1}/instructions", json=payload, headers=AUTH)
    assert r.status_code == 201, r.text
    return r.json()


def deactivate(client, instr_id: int):
    client.patch(f"{V1}/instructions/{instr_id}", json={"active": False}, headers=AUTH)


# ── Create ────────────────────────────────────────────────────────────────────

def test_create_instruction(client):
    r = client.post(
        f"{V1}/instructions",
        json={"action": "FLAG_WARNING", "document_class_id": "dc_006", "active": True},
        headers=AUTH,
    )
    assert r.status_code == 201
    body = r.json()
    assert "id" in body
    assert body["action"] == "FLAG_WARNING"
    assert body["active"] is True


def test_create_instruction_requires_auth(client):
    r = client.post(
        f"{V1}/instructions",
        json={"action": "FLAG_WARNING", "document_class_id": "dc_006"},
    )
    assert r.status_code == 401


def test_invalid_action_rejected(client):
    r = client.post(
        f"{V1}/instructions",
        json={"action": "MADE_UP_TYPE", "document_class_id": "dc_006"},
        headers=AUTH,
    )
    assert r.status_code == 422


# ── List ──────────────────────────────────────────────────────────────────────

def test_list_instructions_response_shape(client):
    create_instruction(client, "FLAG_WARNING", "dc_007")
    r = client.get(f"{V1}/instructions", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "total" in body
    assert "instructions" in body
    assert isinstance(body["instructions"], list)
    assert body["total"] >= 1


def test_list_instructions_filter_by_class(client):
    create_instruction(client, "FLAG_WARNING", "dc_008")
    r = client.get(f"{V1}/instructions?document_class_id=dc_008", headers=AUTH)
    assert r.status_code == 200
    for item in r.json()["instructions"]:
        assert item["document_class_id"] == "dc_008"


# ── Get / Update / Delete ─────────────────────────────────────────────────────

def test_get_instruction_by_id(client):
    instr = create_instruction(client, "SKIP_MATCHING", "dc_009")
    r = client.get(f"{V1}/instructions/{instr['id']}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["id"] == instr["id"]


def test_get_nonexistent_instruction(client):
    r = client.get(f"{V1}/instructions/999999999", headers=AUTH)
    assert r.status_code == 404


def test_update_instruction_active(client):
    instr = create_instruction(client, "FLAG_WARNING", "dc_010")
    r = client.patch(
        f"{V1}/instructions/{instr['id']}",
        json={"active": False},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert r.json()["active"] is False


def test_delete_instruction(client):
    instr = create_instruction(client, "FLAG_WARNING", "dc_011")
    r = client.delete(f"{V1}/instructions/{instr['id']}", headers=AUTH)
    assert r.status_code == 204
    r2 = client.get(f"{V1}/instructions/{instr['id']}", headers=AUTH)
    assert r2.status_code == 404


# ── Pipeline effects ──────────────────────────────────────────────────────────

def test_skip_posting_instruction(client):
    """SKIP_POSTING should not prevent COMPLETED (just skips ERP post step)."""
    instr = create_instruction(client, "SKIP_POSTING", "dc_006")
    try:
        r = _unique_invoice(client, uid())
        assert r.status_code == 200
        doc = wait_done(client, r.json()["id"])
        assert doc["status"] == "COMPLETED"
    finally:
        deactivate(client, instr["id"])


def test_require_approval_instruction(client):
    """REQUIRE_APPROVAL should halt the document in NEEDS_REVIEW."""
    instr = create_instruction(client, "REQUIRE_APPROVAL", "dc_006")
    try:
        r = _unique_invoice(client, uid())
        assert r.status_code == 200
        doc = wait_done(client, r.json()["id"])
        assert doc["status"] == "NEEDS_REVIEW"
    finally:
        deactivate(client, instr["id"])


def test_flag_warning_still_completes(client):
    """FLAG_WARNING should add a warning but still reach COMPLETED."""
    instr = create_instruction(client, "FLAG_WARNING", "dc_006",
                               description="Test warning flag")
    try:
        r = _unique_invoice(client, uid())
        assert r.status_code == 200
        doc = wait_done(client, r.json()["id"])
        assert doc["status"] == "COMPLETED"
    finally:
        deactivate(client, instr["id"])
