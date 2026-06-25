"""
Tests: ClassificationAgent unit tests — text-layer PDFs, OCR path,
manual override, and edge cases.
"""

import io
import pytest
from fpdf import FPDF

# Import the agent directly (no app or DB needed for unit tests)
from app.agents.classification import ClassificationAgent, _extract_pdf_text


class MockDoc:
    """Minimal document stub for ClassificationAgent.classify()."""
    def __init__(self, override=None):
        self.document_class_override = override


def make_pdf(lines: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.cell(0, 8, line[:90], ln=True)
    return bytes(pdf.output())


agent = ClassificationAgent(db=None)


# ── Positive classification ───────────────────────────────────────────────────

@pytest.mark.parametrize("lines,expected_class", [
    # dc_006 — Supplier Invoice
    (["INVOICE", "Invoice No.: INV-001", "Tata Steel Limited",
      "Purchaser: TSL", "VAT 0%", "Payment Terms: 30 days",
      "IBAN: GB12TEST001", "Bank: Test Bank"],
     "dc_006"),

    # dc_004 — Airway Bill
    (["HOUSE AIR WAYBILL", "HAWB No.: 12345", "MAWB No.: 932-000001",
      "AWB No.: WAC-001", "Shipper: Test", "Consignee: Tata",
      "Departure: LHR", "Destination: DEL", "Pieces: 2"],
     "dc_004"),

    # dc_007 — Packing List
    (["PACKING LIST", "Consignee: Tata Steel", "Gross Wt: 500 KG",
      "Nett Wt: 460 KG", "Packages: 3", "Marks & Nos: TSL/001",
      "Dimensions: 120x80", "Shipping: Air"],
     "dc_007"),

    # dc_015 — RFQ
    (["REQUEST FOR QUOTATION", "RFQ No.: RFQ-001", "RFQ Due Date: 2026-07-01",
      "Please quote your best offer", "Tata Steel", "Submit your best offer"],
     "dc_015"),

    # dc_016 — Bill of Entry
    (["BILL OF ENTRY", "Home Consumption", "BE No.: BE123",
      "Port Code: INDEL4", "IEC: 0388039124",
      "Importer: Tata Steel", "Assessed", "Customs"],
     "dc_016"),

    # dc_010 — Order Acknowledgement
    (["ORDER CONFIRMATION", "Order Date: 2026-06-01", "Shipment Date: 2026-07-01",
      "Salesperson: Jane Smith", "Order Acknowledgement No.: OA-001",
      "We confirm your order", "Your order reference: TSL/001"],
     "dc_010"),
])
def test_classify_text_layer(lines, expected_class):
    pdf = make_pdf(lines)
    result = agent.classify(MockDoc(), pdf)
    assert result == expected_class, f"Expected {expected_class}, got {result}"


# ── Manual override ───────────────────────────────────────────────────────────

def test_override_bypasses_keyword_scoring():
    """document_class_override should be returned without any keyword check."""
    # Blank PDF has no keywords → would normally be None
    pdf = FPDF()
    pdf.add_page()
    pdf_bytes = bytes(pdf.output())
    result = agent.classify(MockDoc(override="dc_009"), pdf_bytes)
    assert result == "dc_009"


# ── Unclassified ──────────────────────────────────────────────────────────────

def test_blank_pdf_returns_none():
    pdf = FPDF()
    pdf.add_page()
    pdf_bytes = bytes(pdf.output())
    result = agent.classify(MockDoc(), pdf_bytes)
    assert result is None


def test_irrelevant_text_returns_none():
    """Text that matches no document class keywords should return None."""
    pdf = make_pdf(["The quick brown fox jumps over the lazy dog.",
                    "Lorem ipsum dolor sit amet consectetur adipiscing elit."])
    result = agent.classify(MockDoc(), pdf)
    assert result is None


# ── TLL Invoice disambiguation ────────────────────────────────────────────────

def test_tll_client_invoice_classified_as_dc011():
    pdf = make_pdf([
        "CLIENT INVOICE",
        "Tata Limited Invoice No.: 072904",
        "Buying Commission @ 1.85%",
        "To Freight",
        "To Insurance",
        "Indian Import License",
        "Grosvenor Place, London",
        "Invoice Date: 2026-06-01",
    ])
    result = agent.classify(MockDoc(), pdf)
    assert result == "dc_011"


def test_tll_a2_invoice_classified_as_dc012():
    pdf = make_pdf([
        "CLIENT INVOICE -A2",
        "Tata Limited Invoice No.: 077910",
        "Buying Commission @ 1.85%",
        "A2 Invoice",
        "Commission",
        "Grosvenor Place, London",
    ])
    result = agent.classify(MockDoc(), pdf)
    assert result == "dc_012"


# ── Freight agent invoice disambiguation ─────────────────────────────────────

def test_freight_invoice_classified_as_dc013():
    """Invoice with MAWB-NO header should be dc_013 not dc_006."""
    pdf = make_pdf([
        "INVOICE",
        "MAWB-No.: 932-0001234",
        "HAWB-No.: 57613",
        "Freight Agent: TKM Global",
        "Shipper: Test Ltd",
        "Consignee: Tata Limited",
        "Inv-No.: TEST-001",
        "Acct-No.: ACC-001",
    ])
    result = agent.classify(MockDoc(), pdf)
    assert result == "dc_013"


# ── Text extraction ───────────────────────────────────────────────────────────

def test_extract_pdf_text_returns_lowercase():
    pdf = make_pdf(["Hello World", "TEST CONTENT"])
    text = _extract_pdf_text(pdf)
    assert text == text.lower()


def test_extract_pdf_text_normalises_whitespace():
    """pypdf sometimes inserts double spaces; extraction should normalise."""
    pdf = make_pdf(["This is a test with spacing"])
    text = _extract_pdf_text(pdf)
    # No double spaces in output
    assert "  " not in text


def test_extract_pdf_text_empty_pdf():
    pdf = FPDF()
    pdf.add_page()
    text = _extract_pdf_text(bytes(pdf.output()))
    assert text == ""
