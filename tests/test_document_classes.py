"""
Tests: document classes endpoint — all 20 classes are seeded and returned.

Response shape: {"total": int, "classes": [...]}
"""

from tests.conftest import V1, AUTH

EXPECTED_CLASS_IDS = {f"dc_{i:03d}" for i in range(1, 21)}


def test_document_classes_returns_200(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    assert r.status_code == 200


def test_document_classes_response_shape(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    body = r.json()
    assert "total" in body
    assert "classes" in body
    assert isinstance(body["classes"], list)


def test_document_classes_has_all_20_classes(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    classes = r.json()["classes"]
    ids = {c["id"] for c in classes}
    assert EXPECTED_CLASS_IDS.issubset(ids), (
        f"Missing classes: {EXPECTED_CLASS_IDS - ids}"
    )


def test_document_classes_total_matches_list(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    body = r.json()
    assert body["total"] == len(body["classes"])
    assert body["total"] >= 20


def test_document_classes_have_required_fields(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    for cls in r.json()["classes"]:
        assert "id" in cls
        assert "name" in cls
        assert "slug" in cls
        assert "treatment" in cls
        assert cls["treatment"] in ("PROCESS", "STORE")


def test_document_classes_treatment_split(client):
    """Check that both PROCESS and STORE are present (not all one type)."""
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    classes = r.json()["classes"]
    process_count = sum(1 for c in classes if c["treatment"] == "PROCESS")
    store_count   = sum(1 for c in classes if c["treatment"] == "STORE")
    assert process_count > 0
    assert store_count > 0


def test_specific_known_classes(client):
    r = client.get(f"{V1}/document_classes", headers=AUTH)
    by_id = {c["id"]: c for c in r.json()["classes"]}

    assert by_id["dc_006"]["slug"] == "supplier-invoice"
    assert by_id["dc_006"]["treatment"] == "PROCESS"

    assert by_id["dc_004"]["slug"] == "awb"
    assert by_id["dc_004"]["treatment"] == "PROCESS"

    assert by_id["dc_009"]["treatment"] == "STORE"   # DGD → store only
    assert by_id["dc_014"]["treatment"] == "STORE"   # Insurance → store only


def test_document_class_detail_endpoint(client):
    """GET /document_classes/{id} returns the class with its variants."""
    r = client.get(f"{V1}/document_classes/dc_006", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "dc_006"
    assert "variants" in body
    assert isinstance(body["variants"], list)


def test_document_class_detail_not_found(client):
    r = client.get(f"{V1}/document_classes/dc_999", headers=AUTH)
    assert r.status_code == 404
