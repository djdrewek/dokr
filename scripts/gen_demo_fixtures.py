#!/usr/bin/env python3
"""
Generate the two sample PDFs used by scripts/demo_flow.py.

    tests/fixtures/demo_po.pdf       — Purchase Order   TSL-2024-00847
    tests/fixtures/demo_invoice.pdf  — Purchase Invoice NFS-INV-2024-3891

Both share the same PO number, supplier (Nordic Freight Solutions Ltd),
currency (GBP) and net amount (£12,450.00) so the three-way match engine
produces a clean PASS.

Usage
-----
    python scripts/gen_demo_fixtures.py
"""
from __future__ import annotations

import os
from pathlib import Path

from fpdf import FPDF

HERE     = Path(__file__).parent
OUT      = HERE.parent / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


# ── Style constants ───────────────────────────────────────────────────────────
DARK  = (20,  20,  40)
MID   = (60,  60,  90)
LIGHT = (130, 130, 160)
PALE  = (240, 240, 248)

LINE_ITEMS = [
    ("1", "Ocean freight - Liverpool to Rotterdam (steel coil cargo)", "1", "Lot", "9,200.00",  "9,200.00"),
    ("2", "Port handling & stevedoring charges - Liverpool",           "1", "Lot", "1,850.00",  "1,850.00"),
    ("3", "Documentation & customs clearance (UK export)",             "1", "Set",   "650.00",    "650.00"),
    ("4", "Marine cargo insurance (0.06% of cargo value)",             "1", "Lot",   "420.00",    "420.00"),
    ("5", "Fuel surcharge (BAF)",                                      "1", "Lot",   "330.00",    "330.00"),
]

TABLE_COLS = [
    ("Line", 12), ("Description", 86), ("Qty", 14),
    ("Unit", 18), ("Unit Price (GBP)", 32), ("Total (GBP)", 24),
]


# ── Shared drawing helpers ────────────────────────────────────────────────────
def _header(pdf: FPDF, title: str) -> None:
    pdf.set_fill_color(20, 20, 60)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_xy(14, 9)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 8, title)
    pdf.set_text_color(*DARK)
    pdf.ln(4)


def _section(pdf: FPDF, label: str) -> None:
    pdf.set_fill_color(235, 235, 248)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*MID)
    pdf.cell(0, 6, f"  {label.upper()}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(*DARK)
    pdf.ln(1)


def _kv(pdf: FPDF, key: str, value: str, key_w: int = 55) -> None:
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*LIGHT)
    pdf.cell(key_w, 5, key)
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")


def _divider(pdf: FPDF) -> None:
    pdf.set_draw_color(*LIGHT)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(3)


def _table_header(pdf: FPDF) -> None:
    pdf.set_fill_color(20, 20, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8)
    for text, w in TABLE_COLS:
        pdf.cell(w, 6, text, border=0, fill=True)
    pdf.ln()
    pdf.set_text_color(*DARK)


def _table_rows(pdf: FPDF) -> None:
    for i, row in enumerate(LINE_ITEMS):
        shade = i % 2 == 0
        if shade:
            pdf.set_fill_color(*PALE)
        pdf.set_font("Helvetica", "", 8)
        for value, (_, w) in zip(row, TABLE_COLS):
            pdf.cell(w, 5.5, value, border=0, fill=shade)
        pdf.ln()


def _totals(pdf: FPDF) -> None:
    pdf.ln(3)
    _divider(pdf)
    for label, amount, size in [
        ("Net total",     "GBP  12,450.00",  9),
        ("VAT (20%)",     "GBP   2,490.00",  9),
        ("TOTAL PAYABLE", "GBP  14,940.00", 11),
    ]:
        pdf.set_x(118)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*LIGHT)
        pdf.cell(42, 5, label)
        pdf.set_text_color(20, 20, 60)
        pdf.set_font("Helvetica", "B", size)
        pdf.cell(0, 5, amount, new_x="LMARGIN", new_y="NEXT")


# ── Purchase Order ────────────────────────────────────────────────────────────
def make_po() -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(False)
    pdf.set_margins(14, 14, 14)

    _header(pdf, "PURCHASE ORDER")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*LIGHT)
    pdf.set_xy(14, 32)
    # Use canonical company name + well-known address so classifier keywords fire
    pdf.multi_cell(0, 5,
        "Tata Steel Limited  |  Reg. No. 2280000  |  30 Millbank, Grosvenor Place, London SW1P 4WY")
    pdf.ln(2)

    _section(pdf, "Order details")
    # "purchase order no" and "TSL/" are classifier keywords for dc_003
    _kv(pdf, "Purchase Order No", "TSL/2024/00847")
    _kv(pdf, "PO Date",           "15 March 2024")
    _kv(pdf, "Buyer Reference",   "PROC/2024/Q1/0847")
    _kv(pdf, "Payment Terms",     "Net 30 days from invoice date")
    _kv(pdf, "Currency",          "GBP")
    pdf.ln(3)

    _section(pdf, "Supplier")
    _kv(pdf, "Supplier Name",   "Nordic Freight Solutions Ltd")
    _kv(pdf, "Supplier Code",   "SUP-NFS-0044")
    _kv(pdf, "VAT Number",      "GB 312 4457 09")
    _kv(pdf, "Address",         "Unit 7, Dock Road, Liverpool, L3 4AQ")
    _kv(pdf, "Contact",         "accounts@nordicfreight.co.uk")
    _kv(pdf, "Orders To",       "orders@tata.co.uk")
    pdf.ln(3)

    _section(pdf, "Delivery")
    _kv(pdf, "Ship From",       "Port of Liverpool, Merseyside, UK")
    _kv(pdf, "Ship To",         "Rotterdam Container Terminal, Netherlands")
    _kv(pdf, "Incoterm",        "CIF Rotterdam")
    _kv(pdf, "Required By",     "22 March 2024")
    pdf.ln(3)

    # "please supply and deliver" is a strong dc_003 keyword
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*MID)
    pdf.multi_cell(0, 5,
        "Please supply and deliver the goods or services detailed below in accordance "
        "with Tata Steel standard terms and conditions of purchase.")
    pdf.set_text_color(*DARK)
    pdf.ln(2)

    _section(pdf, "Line items")
    _table_header(pdf)
    _table_rows(pdf)
    _totals(pdf)

    pdf.ln(6)
    _divider(pdf)
    _section(pdf, "Authorisation")
    _kv(pdf, "Approved by",     "James Cartwright, Head of Procurement")
    _kv(pdf, "Approval Date",   "15 March 2024")
    _kv(pdf, "Cost Centre",     "CC-OPS-PORT-04")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*LIGHT)
    pdf.multi_cell(0, 4,
        "Invoices must quote Purchase Order No TSL/2024/00847 and be sent to orders@tata.co.uk. "
        "No goods or services may be supplied without a valid PO reference.")

    dest = OUT / "demo_po.pdf"
    pdf.output(str(dest))
    print(f"Written: {dest}  ({os.path.getsize(dest):,} bytes)")


# ── Purchase Invoice ──────────────────────────────────────────────────────────
def make_invoice() -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(False)
    pdf.set_margins(14, 14, 14)

    _header(pdf, "PURCHASE INVOICE")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*LIGHT)
    pdf.set_xy(14, 32)
    pdf.multi_cell(0, 5,
        "Nordic Freight Solutions Ltd  |  VAT No. GB 312 4457 09  "
        "|  Unit 7, Dock Road, Liverpool, L3 4AQ")
    pdf.ln(2)

    _section(pdf, "Invoice details")
    _kv(pdf, "Invoice Number",  "NFS-INV-2024-3891")
    _kv(pdf, "Invoice Date",    "22 March 2024")
    _kv(pdf, "Due Date",        "21 April 2024")
    _kv(pdf, "PO Reference",    "TSL/2024/00847")
    _kv(pdf, "Payment Terms",   "Net 30 days")
    _kv(pdf, "Currency",        "GBP")
    pdf.ln(3)

    _section(pdf, "Billed to")
    _kv(pdf, "Customer",        "Tata Steel UK Limited")
    _kv(pdf, "Address",         "Port Talbot, Wales SA13 2NG")
    _kv(pdf, "VAT Number",      "GB 142 0012 61")
    _kv(pdf, "Contact",         "ap@tatasteeluk.com")
    pdf.ln(3)

    _section(pdf, "Services rendered")
    _table_header(pdf)
    _table_rows(pdf)
    _totals(pdf)

    pdf.ln(6)
    _divider(pdf)
    _section(pdf, "Payment details")
    _kv(pdf, "Bank",            "Barclays Bank PLC")
    _kv(pdf, "Sort Code",       "20-00-00")
    _kv(pdf, "Account Number",  "40308669")
    _kv(pdf, "IBAN",            "GB29 BARC 2000 0040 3086 69")
    _kv(pdf, "Reference",       "NFS-INV-2024-3891  /  TSL-2024-00847")
    pdf.ln(3)

    _section(pdf, "Certification")
    _kv(pdf, "Issued by",       "Nordic Freight Solutions Ltd")
    _kv(pdf, "Signatory",       "Annika Lindqvist, Finance Director")
    _kv(pdf, "Date",            "22 March 2024")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*LIGHT)
    pdf.multi_cell(0, 4,
        "This invoice is issued under Purchase Order No TSL/2024/00847 raised by Tata Steel Limited. "
        "Please remit payment by 21 April 2024. "
        "Queries: accounts@nordicfreight.co.uk  |  +44 151 430 8800.")

    dest = OUT / "demo_invoice.pdf"
    pdf.output(str(dest))
    print(f"Written: {dest}  ({os.path.getsize(dest):,} bytes)")


if __name__ == "__main__":
    make_po()
    make_invoice()
    print("Done. Run the demo with:  python scripts/demo_flow.py")
