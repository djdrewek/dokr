"""
Pytest configuration and shared fixtures for the Dokr API test suite.

Design decisions:
  - DATABASE_URL must be set before any app import so the module-level
    `settings` singleton picks it up. That's why it appears at the very top.
  - A single in-process TestClient is shared across the session for speed.
    Background tasks run in a background thread, so `wait_done()` polls until
    the document reaches a terminal pipeline state.
  - fpdf2 generates proper PDFs whose text can be extracted by pypdf. Each
    helper creates a document whose keywords match a specific document class.
"""

import os
import time

# ── Must be set BEFORE any app module is imported ────────────────────────────
TEST_DB_PATH = "/tmp/dokr_pytest_suite.db"
TEST_API_KEY = "dk_test_pytest_dokr_12345"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["DOKR_API_KEY"] = TEST_API_KEY

# ── Now safe to import app ─────────────────────────────────────────────────
import pytest                                      # noqa: E402
from fpdf import FPDF                              # noqa: E402
from fastapi.testclient import TestClient          # noqa: E402
from app.main import app                           # noqa: E402

# ── Shared constants ──────────────────────────────────────────────────────────
AUTH   = {"Authorization": f"Bearer {TEST_API_KEY}"}
NOAUTH = {}
V1     = "/v1"

PIPELINE_IN_PROGRESS = {
    "RECEIVED", "DEDUPLICATING", "CLASSIFYING", "EXTRACTING",
    "AI_REVIEWING", "VALIDATING", "LINKING", "MATCHING",
    "POSTING", "FILING", "NOTIFYING",
}
TERMINAL = {
    "COMPLETED", "EXACT_DUPLICATE", "CONTENT_DUPLICATE",
    "NEAR_DUPLICATE", "UNCLASSIFIED", "NEEDS_REVIEW", "FAILED",
}


# ── PDF factories ─────────────────────────────────────────────────────────────

def _pdf(lines: list[str]) -> bytes:
    """Create a minimal valid PDF from a list of text lines."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.cell(0, 8, line[:90], ln=True)  # fpdf2 max cell width guard
    return bytes(pdf.output())


def supplier_invoice_pdf() -> bytes:
    """Generates a PDF that classifies as dc_006 (Supplier Invoice)."""
    return _pdf([
        "INVOICE",
        "Invoice No.: INV-TEST-001",
        "Invoice Date: 2026-06-01",
        "Supplier: Test Supplier GmbH",
        "Customer: Tata Steel Limited",
        "Purchaser: Tata Steel Limited",
        "PO No.: TSL/58001",
        "Payment Terms: 45 days net",
        "IBAN: GB12TEST00000000000001",
        "VAT Registration: GB 123 4567 89",
        "Invoice Total: EUR 5,000.00",
        "Bank: Deutsche Bank AG",
    ])


def awb_pdf() -> bytes:
    """Generates a PDF that classifies as dc_004 (Airway Bill / House AWB)."""
    return _pdf([
        "HOUSE AIR WAYBILL",
        "MAWB No.: 932-00000001",
        "HAWB No.: HWB99999",
        "Shipper: Test Shipper Ltd",
        "Consignee: Tata Steel Limited",
        "Airport of Departure: LHR",
        "Airport of Destination: DEL",
        "Pieces: 2",
        "Chargeable Weight: 15.0 KG",
    ])


def tsl_po_pdf() -> bytes:
    """Generates a PDF that classifies as dc_003 (Tata Steel PO)."""
    return _pdf([
        "Tata Steel Limited",
        "Purchase Order No.: TSL/99000",
        "Please Supply and Deliver",
        "TSL/99000 reference",
        "Tata Steel UK",
        "Orders@tata.co.uk",
        "Grosvenor Place London",
        "Supplier: Example GmbH",
    ])


def packing_list_pdf() -> bytes:
    """Generates a PDF that classifies as dc_007 (Packing List)."""
    return _pdf([
        "PACKING LIST",
        "Consignee: Tata Steel Limited",
        "Gross Wt: 500 KG",
        "Nett Wt: 460 KG",
        "Packages: 3 cases",
        "Marks & Nos: TSL/99000",
        "Dimensions: 120x80x60 cm",
        "QTY: 10",
        "Shipping: Air",
    ])


def rich_invoice_pdf(tag: str) -> bytes:
    """
    dc_006 PDF with all ValidationAgent-required fields present:
    invoice_number, invoice_date, supplier_name, total_amount,
    currency, customer_po_ref.
    Pass a short unique tag to avoid exact-dup 409s.
    """
    n = int(tag, 16) % 99_999
    return _pdf([
        f"INVOICE {tag}",
        f"Invoice No.: INV-{n:05d}",
        "Invoice Date: 2026-06-01",
        "Supplier: Test Supplier GmbH",
        "Customer: Tata Steel Limited",
        f"PO No.: TSL/5{n:04d}",
        "Payment Terms: 45 days net",
        f"IBAN: GB12TEST{n:05d}0001",
        "Bank: Test Bank AG",
        "Currency: EUR",
        "Invoice Total: EUR 5,000.00",
    ])


def blank_pdf() -> bytes:
    """A valid but empty PDF — should result in UNCLASSIFIED."""
    pdf = FPDF()
    pdf.add_page()
    return bytes(pdf.output())


# ── Poll helper ───────────────────────────────────────────────────────────────

def wait_done(client: TestClient, doc_id: str, timeout: float = 10.0) -> dict:
    """
    Poll GET /v1/documents/{id} until the document reaches a terminal state.
    Returns the final document dict. Raises AssertionError on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"{V1}/documents/{doc_id}", headers=AUTH)
        assert r.status_code == 200, f"GET /documents/{doc_id} returned {r.status_code}"
        doc = r.json()
        if doc["status"] in TERMINAL:
            return doc
        time.sleep(0.15)
    # One final check and return (let the test assert on the status)
    r = client.get(f"{V1}/documents/{doc_id}", headers=AUTH)
    return r.json()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient. Removes any stale test DB first, then spins up
    the app. The lifespan hook runs init_db() once at startup.
    """
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    with TestClient(app) as c:
        yield c
