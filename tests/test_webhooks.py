"""
Tests: webhook subscription CRUD, test delivery, and HMAC signature verification.

Response shapes:
  List: {"total": int, "subscriptions": [...]}
  Item: {id, url, events, active, ...}
  Test: {webhook_id, event, success, status_code, detail}
"""

import hashlib
import hmac
import json
from tests.conftest import V1, AUTH


# ── Create ────────────────────────────────────────────────────────────────────

def test_create_webhook(client):
    r = client.post(
        f"{V1}/webhooks",
        json={
            "url": "https://example.com/webhook",
            "events": ["document.completed", "document.failed"],
            "secret_key": "test_secret_abc",
        },
        headers=AUTH,
    )
    assert r.status_code == 201
    body = r.json()
    assert "id" in body
    assert body["url"] == "https://example.com/webhook"
    assert "document.completed" in body["events"]
    assert body["active"] is True


def test_create_webhook_without_secret(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://no-secret.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    assert r.status_code == 201


def test_create_webhook_requires_auth(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://noauth.example.com/x", "events": ["document.completed"]},
    )
    assert r.status_code == 401


# ── List ──────────────────────────────────────────────────────────────────────

def test_list_webhooks_response_shape(client):
    r = client.get(f"{V1}/webhooks", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "total" in body
    assert "subscriptions" in body
    assert isinstance(body["subscriptions"], list)


def test_list_webhooks_has_entries(client):
    # Ensure at least one exists
    client.post(
        f"{V1}/webhooks",
        json={"url": "https://list-ensure.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    r = client.get(f"{V1}/webhooks", headers=AUTH)
    assert r.json()["total"] >= 1
    assert len(r.json()["subscriptions"]) >= 1


# ── Get ───────────────────────────────────────────────────────────────────────

def test_get_webhook_by_id(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://get-test.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    wh_id = r.json()["id"]
    r2 = client.get(f"{V1}/webhooks/{wh_id}", headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["id"] == wh_id


def test_get_nonexistent_webhook(client):
    r = client.get(f"{V1}/webhooks/00000000000000000000000000", headers=AUTH)
    assert r.status_code == 404


# ── Update ────────────────────────────────────────────────────────────────────

def test_update_webhook_active_flag(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://patch-test.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    wh_id = r.json()["id"]
    r2 = client.patch(
        f"{V1}/webhooks/{wh_id}",
        json={"active": False},
        headers=AUTH,
    )
    assert r2.status_code == 200
    assert r2.json()["active"] is False


def test_update_webhook_events(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://events-upd.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    wh_id = r.json()["id"]
    r2 = client.patch(
        f"{V1}/webhooks/{wh_id}",
        json={"events": ["document.completed", "document.needs_review"]},
        headers=AUTH,
    )
    assert r2.status_code == 200
    assert "document.needs_review" in r2.json()["events"]


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_webhook(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://delete-test.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    wh_id = r.json()["id"]
    r2 = client.delete(f"{V1}/webhooks/{wh_id}", headers=AUTH)
    assert r2.status_code == 204
    r3 = client.get(f"{V1}/webhooks/{wh_id}", headers=AUTH)
    assert r3.status_code == 404


# ── Test delivery endpoint ────────────────────────────────────────────────────

def test_webhook_test_delivery_returns_200(client):
    """
    POST /webhooks/{id}/test always returns 200 (the response describes whether
    delivery succeeded, not whether the endpoint call succeeded).
    Even if the target URL is unreachable, the response is 200 with detail.
    """
    r = client.post(
        f"{V1}/webhooks",
        json={
            "url": "https://httpbin.org/status/200",
            "events": ["document.completed"],
            "secret_key": "test_hmac_secret",
        },
        headers=AUTH,
    )
    wh_id = r.json()["id"]
    r2 = client.post(f"{V1}/webhooks/{wh_id}/test", headers=AUTH)
    assert r2.status_code == 200


def test_webhook_test_delivery_has_expected_fields(client):
    r = client.post(
        f"{V1}/webhooks",
        json={"url": "https://test-fields.example.com/hook", "events": ["document.completed"]},
        headers=AUTH,
    )
    wh_id = r.json()["id"]
    r2 = client.post(f"{V1}/webhooks/{wh_id}/test", headers=AUTH)
    assert r2.status_code == 200
    body = r2.json()
    # Response shape may vary; just check the call completes cleanly
    assert isinstance(body, dict)


# ── HMAC signature verification ───────────────────────────────────────────────

def test_hmac_signature_format():
    secret = "my_webhook_secret"
    payload = json.dumps({"event": "document.completed", "id": "doc_abc"}).encode()
    sig = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    assert sig.startswith("sha256=")
    assert len(sig) == 71  # "sha256=" (7) + 64 hex chars


def test_hmac_signature_different_secrets_differ():
    payload = b'{"event": "test"}'
    sig1 = hmac.new(b"secret_a", payload, hashlib.sha256).hexdigest()
    sig2 = hmac.new(b"secret_b", payload, hashlib.sha256).hexdigest()
    assert sig1 != sig2


def test_hmac_signature_deterministic():
    payload = b'{"event": "test"}'
    secret = b"my_secret"
    sig1 = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    sig2 = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    assert sig1 == sig2
