"""
Tests: authentication — missing key, wrong format, wrong value, valid key.
"""

import io
from tests.conftest import V1, AUTH, supplier_invoice_pdf


def _submit(client, headers: dict, pdf: bytes | None = None):
    if pdf is None:
        pdf = supplier_invoice_pdf()
    return client.post(
        f"{V1}/documents/submit",
        files={"file": ("test.pdf", io.BytesIO(pdf), "application/pdf")},
        data={"priority": "standard"},
        headers=headers,
    )


def test_no_auth_returns_401(client):
    r = _submit(client, headers={})
    assert r.status_code == 401
    assert "missing_api_key" in str(r.json())


def test_bad_format_returns_401(client):
    r = _submit(client, headers={"Authorization": "Bearer notavalidprefix_abc"})
    assert r.status_code == 401
    assert "invalid_api_key_format" in str(r.json())


def test_wrong_key_returns_401(client):
    r = _submit(client, headers={"Authorization": "Bearer dk_live_wrongkey12345"})
    assert r.status_code == 401
    assert "invalid_api_key" in str(r.json())


def test_valid_key_accepted(client):
    r = _submit(client, headers=AUTH)
    # 200 = accepted; 409 = already submitted this exact PDF (also fine — auth passed)
    assert r.status_code in (200, 409)


def test_document_classes_requires_auth(client):
    r = client.get(f"{V1}/document_classes")
    assert r.status_code == 401


def test_document_classes_with_auth(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    assert r.status_code == 200
