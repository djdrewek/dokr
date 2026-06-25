"""
Tests: system health, root, and OpenAPI docs endpoints (no auth required).
"""

from tests.conftest import V1


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["product"] == "Dokr"
    assert "version" in body


def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["product"] == "Dokr"
    assert "docs" in body
    assert body["api_base"] == "/v1"


def test_openapi_docs(client):
    r = client.get("/docs")
    assert r.status_code == 200


def test_openapi_json(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "Dokr API"
    # All main router groups should appear in the paths
    paths = schema["paths"]
    assert any("/documents" in p for p in paths)
    assert any("/webhooks" in p for p in paths)
    assert any("/instructions" in p for p in paths)
