"""
Extraction Agent — 2-tier field extraction with proofreading quality gate.

Tier 1 — PDF text layer (pypdf → AI text extraction)
  Fast, free, works on any PDF with a text layer.
  Confidence 0.75–0.95 depending on pattern specificity.

Tier 2 — AI Vision (Claude with image input → JSON)
  For scanned / image-only PDFs, mixed PDFs, complex layouts,
  handwriting, rotated text, overlaid documents, mixed scripts.
  Requires ANTHROPIC_API_KEY in environment / .env.

After each tier, ProofreadingAgent validates the fields.
If the result passes, it flows to VALIDATING.
If both tiers fail, ExtractionAgent returns None → NEEDS_REVIEW.

Confidence levels:
  Tier 1 TEXT_LAYER: regex match confidence × learning stage dampen
  Tier 2 OCR:        same × 0.90 additional OCR noise dampen
  Tier 3 AI_VISION:  0.82–0.88 base (model-inherent uncertainty)
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field as dc_field

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.agents.proofreading import ProofreadingAgent, FIELD_FORMAT_VALIDATORS
from app.config import settings
from app.models.document import Document, DocumentClass, DocumentVariant
from app.models.extracted_field import ExtractedField

logger = logging.getLogger(__name__)

# ── Fields that hold numeric amounts — trailing punctuation is stripped ────────
_NUMERIC_FIELD_NAMES: frozenset[str] = frozenset({
    "total_amount", "total_order_value", "total_invoice_value",
    "net_amount", "subtotal", "vat_amount", "payment_amount",
    "chargeable_weight", "actual_weight", "total_gross_weight", "total_net_weight",
    "buying_commission_pct", "buying_commission_amt", "customs_duty", "total_cif_value",
    "quantity", "pieces",
})

# ── Regex extraction patterns ─────────────────────────────────────────────────
# List of (pattern, group_index, base_confidence) per field name.
# First match wins. All patterns applied to lower-cased text for matching,
# then re-extracted from original-case text for the stored value.

EXTRACTION_PATTERNS: dict[str, list[tuple[str, int, float]]] = {

    "po_number": [
        # Slash-separated PO formats: TSL/58237, TML/32662, etc.
        (r"(?:p\.?o\.?\s*(?:no\.?|number|ref\.?|#)\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.92),
        (r"\b(TSL\/\d{1,2}\s?\d{4,5}(?:\/[^\s,;]{1,15})?)\b", 1, 0.95),  # handles "TSL/5 2602" space variant
        (r"\b(TSL\/\d{5,6}(?:\/\d+)?)\b", 1, 0.95),
        (r"\b(TML\/\d{5,6}(?:\/\d+)?)\b", 1, 0.95),
        (r"\b(TMPVL\/\d{5,8}\/\d+)\b", 1, 0.95),
        (r"(?:purchase\s+order\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.88),
        # TML Import Contract series: TJAC32192V, TUAC32609V, etc. (labeled "PO No :")
        (r"(?:p\.?o\.?\s*(?:no\.?|number|ref\.?|#)\s*[:\-]?\s*)([A-Z]{3,8}\d{4,10}[A-Z]?)\b", 1, 0.90),
        # Bare TJAC/TUAC/similar identifiers in text
        (r"\b((?:TJAC|TUAC|TJEC|TSAC|TAPVL|TMGMV)\d{4,9}[A-Z]?)\b", 1, 0.93),
        # SAP-style 10-digit contract/order numbers ("The contract no is. 4700155396")
        (r"(?:contract\s+no(?:\.?\s*is)?|contract\s+number)\s*[:\-\.]*\s*(\d{7,12})", 1, 0.84),
        # TSL SAP PO header: "PURCHASE ORDER NO. TATA STEEL/ 2200063147"
        (r"purchase\s+order\s+no\.?\s+tata\s+steel\s*\/\s*(\d{7,12})", 1, 0.87),
        # TSL SAP footer reference: "TATA STEEL/2200063147" (standalone on last page)
        (r"\btata\s+steel\s*\/\s*(\d{7,12})\b", 1, 0.82),
    ],

    "invoice_number": [
        (r"(?:invoice\s*(?:no\.?|number|#)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\/\-\.]{2,29})", 1, 0.91),
        # TKM freight: "I N V O I C E 30357 25801 ..." — spaced-letter heading;
        # invoice number is the FIRST numeric token after the heading.
        (r"(?:i\s+n\s+v\s+o\s+i\s+c\s+e)\s+(\d{3,10})\b", 1, 0.88),
        # "Inv-No.: 30357" — REQUIRE colon/dash so column header "Inv-No. Acct-No." is not matched.
        # Accept alpha-start to handle "INV.NO. : TXCCU260500156" (ocean freight invoices).
        (r"(?:inv[-\.]?\s*no\.?\s*[:\-]+\s*)([A-Z0-9][A-Z0-9\/\-\.]{1,29})", 1, 0.87),
        (r"(?:rechnung(?:snummer)?\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\/\-\.]{2,29})", 1, 0.87),
        # "Invoice n. , date: .../NR 25206544" — Danieli-style column header + /NR value label
        (r"(?:/NR)\s+(\d{5,12})\b", 1, 0.84),
    ],

    "invoice_date": [
        (r"(?:invoice\s*date|date\s+of\s+invoice)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.90),
        (r"(?:invoice\s*date|date\s+of\s+invoice)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.90),
        (r"(?:invoice\s*date|datum)\s*[:\-]?\s*(\d{1,2}\.\s*\w+\s+\d{4})", 1, 0.85),
        # TKM freight invoice: "I N V O I C E {inv_no} {acct_no} {shipment_no} {date}"
        # Four space-separated columns; the date is the last value on the row.
        (r"(?:i\s+n\s+v\s+o\s+i\s+c\s+e)\s+\d+\s+\d+\s+[\d\-]+\s+(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{4})", 1, 0.82),
        # Danieli "/NR {invoice_no}, {date}" — date immediately after invoice number
        (r"(?:/NR)\s+\d+,\s*(\d{2}[\/\.]\d{2}[\/\.]\d{2,4})", 1, 0.78),
    ],

    "po_date": [
        (r"(?:p\.?o\.?\s*date|order\s+date|date\s+of\s+order)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.88),
        (r"(?:p\.?o\.?\s*date|order\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.88),
        # TLL PO format: "Req No:DD/MM/YYYY Order Date:" — the date follows Req No
        (r"(?:req\s*no\.?\s*[:\-]?\s*)(\d{2}[\/\.]\d{2}[\/\.]\d{4})", 1, 0.80),
    ],

    "currency": [
        # Use lookahead (?![A-Z]) instead of trailing \b so codes like "SEK12-15weeks"
        # (no space between currency code and following digit) are still matched.
        # The leading \b ensures we don't match mid-word (e.g. inside "SEEK" → rejected
        # because \b fires before S, but then lookahead would reject "EEK" from "SEEK").
        (r"\b(EUR|GBP|USD|INR|CHF|SEK|DKK|NOK|AUD|CAD|JPY|SGD|HKD)(?![A-Z])", 1, 0.87),
    ],

    "total_amount": [
        # "invoice total / total amount / total amount due / grand total / amount due"
        (r"(?:invoice\s+total|total\s+amount(?:\s+due)?|grand\s+total|amount\s+(?:due|payable))\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.88),
        # "Total: EUR 1000.00" — require colon/dash to avoid table-header false matches
        (r"\btotal\s*[:\-]\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.82),
        (r"(?:netto\s*betrag|net\s*total)\s*[:\-]?\s*(\d[\d,\.]+)", 1, 0.83),
        (r"(?:total\s+incl\.?\s+vat)\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.85),
        # European column format: amount PRECEDES "Total Value" label
        # e.g. Danieli "253.431,00 Total Value FOB BARI PORT" (value before label)
        (r"([1-9][\d\.]+,\d{2})\s+Total\s+Value\b", 1, 0.80),
        # Ocean freight single-item invoice: "Ocean Freight 1 93.50 996521"
        # qty=1 followed by rate (the total for one line = the invoice total).
        (r"(?:ocean\s+)?freight\s+\d+\s+([\d,]+\.\d{2})\b", 1, 0.72),
        # Swedish/Nordic space-thousands format: "Price each\n166 825,00" or "166 825,00 SEK"
        # Captures NNN NNN,NN pattern (digits, optional space-group, comma-cents).
        (r"(?:price\s+each|total\s+amount|amount\s+(?:due|payable))\s*(?:[A-Z]{3}\s*)?\n?\s*(\d{1,3}(?:\s\d{3})+,\d{2})", 1, 0.80),
    ],

    "total_order_value": [
        # "Total Order Value: EUR 74,965.00" or "Total Order Value:EXWorks 74,965.00 EUR"
        (r"(?:total\s+(?:order\s+)?value|order\s+total|contract\s+(?:value|total))\s*[:\-]?\s*(?:[A-Z]{3,10}\s*)?(\d[\d,\.]+)", 1, 0.88),
        (r"(?:gesamtwert|total\s+price|order\s+value)\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.83),
        (r"\btotal\s+value\s*(?:of\s*)?(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.82),
        # Rate contracts: "approx. value will be around USD 320,000.00" or
        # "approximate value will be around EUR 5,06,053.58"
        # Note: "approx." has a period — use \.? to handle both spellings.
        (r"approx(?:imate)?\.?\s+value\s+(?:will\s+be\s+)?(?:around|of|is)\s+[A-Z]{3}\s*([\d,\.]+)", 1, 0.80),
        (r"(?:approximate\s+value\s+will\s+be\s+around)\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.80),
        # SAP multi-item PO: "Item Price @13,650.00 EUR Per 1 Set 13,650.00 EUR"
        # Captures the per-item total (last number on the item-price line).
        # Confidence is low — may only reflect one line item, not the PO total.
        # Note: @[\d,\.]+ handles comma-formatted prices (e.g. 13,650.00).
        (r"item\s+price\s+@[\d,\.]+\s+[A-Z]{3}\s+per\s+[\d,\.]+\s+\w+\s+([\d,\.]+)", 1, 0.68),
        # Large numbers with currency suffix: "515,000.00"
        (r"(?:value\s+of\s+[A-Z]{3}\s*)(\d[\d,\.]+)", 1, 0.81),
        # TSL PO format: "3,422.60 TaxNetFCA" — net total immediately before "TaxNet" label
        # Note: TaxNet is concatenated with incoterm (no space), so no \b here
        (r"([1-9][\d,]+\.\d{2})\s+(?:TaxNet|Tax\s*Net)", 1, 0.74),
    ],

    "net_amount": [
        (r"(?:net\s*amount|net\s*total|netto\s*betrag)\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.86),
    ],

    "payment_terms": [
        (r"(?:payment\s+terms?|terms\s+of\s+payment|zahlungsbedingungen)\s*[:\-]?\s*([^\n\r]{3,50})", 1, 0.84),
    ],

    "supplier_name": [
        # "To: COMPANY NAME GmbH" — used in TLL PO format
        (r"\bTo:\s+([A-Z][A-Z\s&\-\.]{2,45}(?:GmbH|Ltd\.?|Limited|AG|BV|Inc\.?|SRL|SA|KG|OHG|AS|AB|LLC|Corp\.?))", 1, 0.84),
        # "Supplier: Company Name GmbH" / "Manufacturer or the case: duisport... GmbH"
        # Suffix is mandatory in capture — prevents over-extending into address lines.
        # [^:]{0,25} allows "Manufacturer or the case:" style labels.
        (r"(?:supplier|vendor|issued\s+by|manufacturer(?:[^:]{0,25})?)\s*[:\-]\s*([A-Z][A-Za-z0-9&\.\s\-]{3,50}(?:GmbH|Ltd\.?|Limited|AG|BV|Inc\.?|S\.A\.|KG|OHG|LLC|Corp\.?))", 1, 0.82),
        # Broader fallback: no suffix required (catches "Supplier: Simpex Engineering")
        (r"(?:supplier|vendor|issued\s+by)\s*[:\-]\s*([A-Z][A-Za-z0-9&\.\s\-]{3,50})", 1, 0.80),
        # Japanese/Asian "CO., LTD." style: "YAMATO GOKIN CO., LTD." (comma breaks general pattern)
        (r"([A-Z][A-Z\s]{2,35}CO\.,\s*(?:LTD\.?|INC\.?))", 1, 0.82),
        # Company names with legal entity suffix.
        # (?!\w) after suffix prevents matching suffixes inside ordinary words.
        # SA / AS removed — too short, appear as common word substrings.
        # AB = Swedish Aktiebolag (e.g. "ABP Induction AB", "Sandvik AB")
        # S.p.A. = Italian joint-stock company (e.g. "Danieli & C. Officine meccaniche S.p.A.")
        (r"([A-Z][a-zA-Z&\.\s\-]{2,40}(?:GmbH|Ltd\.?|Limited|AG|BV|Inc\.?|SRL|KG|OHG|S\.A\.|S\.A\.U\.|S\.p\.A\.|SpA|s\.r\.o\.|GmbH\s*&\s*Co\.\s*KG|LLC|Corp\.?|\bAB\b))(?!\w)", 1, 0.75),
        # Issued by / commitment by (covers "3D Systems", "ABP Induction" etc. in quotations)
        (r"(?:commitment|offer)\s+by\s+((?:\d+[A-Z]\s+)?[A-Z][A-Za-z\d\s&\.\-]{3,40}?)\s+(?:to\s+supply|reserves?|to\s+fulfill)", 1, 0.76),
        # OCR-concatenated CamelCase company + S.A.: "PaulWurthS.A." (scanned docs)
        (r"\b([A-Z][a-z]{2,12}[A-Z][a-z]{2,12}S\.A\.)\b", 1, 0.71),
        # Website domain as supplier identifier: "www.paulwurth.lu" → "paulwurth"
        (r"\bwww\.([\w\d][\w\d\-]+)\.\w{2,}\b", 1, 0.58),
        # E-Mail domain as last-resort supplier name in quotation/OC context
        (r"(?:E-?Mail|e-?mail|contact)\s*[:\@\s]+[\w\.\+\-]+@([\w\d][\w\d\-]+)\.\w{2,}", 1, 0.60),
    ],

    "entity": [
        (r"(Tata\s+(?:Steel\s+Limited|Steel\s+UK|Motors\s+Limited|Motors\s+Passenger\s+Vehicles\s+Limited|Ltd\.?\s*(?:London)?|Limited))", 1, 0.96),
        (r"(?:buyer|purchaser|bill\s+to|ship\s+to)\s*[:\-]?\s*(Tata\s+(?:Steel|Motors|Ltd)[^\n\r]{0,40})", 1, 0.88),
    ],

    "awb_number": [
        (r"(?:(?:m)?awb|air\s+waybill|airway\s+bill)\s*(?:no\.?)?\s*[:\-]?\s*(\d{3}[-\s]\d{8})", 1, 0.93),
        (r"mawb[:\-\s]+(\d{3}[-\s]\d{8})", 1, 0.92),
        (r"\b(\d{3}[-\s]\d{4}\s*\d{4})\b", 1, 0.82),
    ],

    "hawb_number": [
        (r"(?:hawb|house\s+(?:air\s+)?waybill)\s*(?:no\.?)?\s*[:\-]?\s*([A-Z0-9]{4,20})", 1, 0.91),
        (r"hawb[:\-\s]+([A-Z0-9]{4,20})", 1, 0.90),
    ],

    "shipper_name": [
        (r"(?:shipper|consignor|sender)\s*[:\-]?\s*([A-Z][^\n\r]{3,60})", 1, 0.84),
    ],

    "consignee_name": [
        # Explicit ship-to / deliver-to with Tata entity — capture exact entity name only
        (r"(?:ship\s+to|deliver\s+to|delivery\s+address)\s*[:\-]?\s*((?:Tata|TATA)\s+(?:Steel|Motors)\s+(?:Limited|LIMITED|Ltd\.?|UK|Europe|Steel))", 1, 0.80),
        # Standalone Tata Steel entities — consignee in almost all inbound shipping docs.
        # Listed BEFORE the generic "consignee:" pattern so a blank "CONSIGNEE:" label
        # followed by an address line doesn't capture the address instead of Tata.
        (r"\b(TATA\s+STEEL\s+LIMITED)\b", 1, 0.78),
        (r"(Tata\s+Steel\s+(?:Limited|Ltd\.?|UK|Europe))", 1, 0.77),
        # Generic consignee label — fallback for non-Tata consignees
        (r"(?:consignee|receiver|recipient)\s*[:\-]?\s*([A-Z][^\n\r]{3,60})", 1, 0.84),
    ],

    "gross_weight": [
        # Capture only the numeric part; unit is consumed but not captured
        (r"(?:gross\s*(?:weight)?|gwt|g\.?w\.?)\s*[:\-]?\s*([0-9,\.]+)(?:\s*(?:kgs?|lbs?|kg))?\b", 1, 0.86),
        # "Gross Wt. (Kg): 282" — unit in parentheses before colon
        (r"gross\s+wt\.?\s*(?:\([a-z]{1,5}\))?\s*[:\-]\s*([0-9,\.]+)", 1, 0.84),
        # Japanese PL "Number of Packaging (Grand total) count net_kg gross_kg vol"
        # Columns: count | net | gross | volume — capture the SECOND number (gross)
        (r"(?:grand\s+total\))\s+\d+\s+[0-9,\.]+\s+([0-9,\.]+)", 1, 0.82),
    ],

    "country_of_origin": [
        # Require colon or dash after label to avoid matching "country of origin"
        # mid-sentence (e.g. "…with respect to production in country of origin with…").
        # Negative lookahead prevents TSL PO template "COUNTRY OF ORIGIN: Net Weight:"
        # from capturing the next column header "Net Weight" as the country value.
        (r"(?:country\s+of\s+origin|ursprungsland|origin\s+country)\s*[:\-]\s*(?!Net\s+Weight|Gross\s+Weight|Weight\b)([A-Z][a-zA-Z\s]{2,30})", 1, 0.88),
        # Address-embedded country: ", Korea" / ", Germany" at end of postal address
        # Matches APTA Form-I and other certs where country appears in exporter address.
        (r",\s*(Korea|South Korea|Japan|Germany|France|Italy|China|India|Taiwan|Vietnam|Thailand|United Kingdom|Austria|Sweden|Belgium|Netherlands|Spain|Switzerland|Czech Republic|Poland|Slovakia|Hungary|Romania|Turkey)\b", 1, 0.72),
    ],

    "pieces": [
        (r"(?:no\.?\s+of\s+pieces?|pieces?|pkgs?|packages?|collis?)\s*[:\-]?\s*(\d+)", 1, 0.85),
    ],

    "incoterms": [
        (r"\b(EXW|FCA|CPT|CIP|DAP|DPU|DDP|FAS|FOB|CFR|CIF)\b", 1, 0.90),
    ],

    "final_destination": [
        (r"(?:final\s+destination|destination|bestimmungsort)\s*[:\-]?\s*([A-Z][^\n\r]{3,40})", 1, 0.83),
    ],

    "port_of_entry": [
        (r"(?:port\s+of\s+(?:entry|discharge|arrival)|discharge\s+port)\s*[:\-]?\s*([A-Z][^\n\r]{2,30})", 1, 0.85),
    ],

    "port_of_loading": [
        (r"(?:port\s+of\s+(?:loading|departure)|loading\s+port)\s*[:\-]?\s*([A-Z][^\n\r]{2,30})", 1, 0.85),
    ],

    "mode_of_dispatch": [
        (r"(?:mode\s+of\s+(?:dispatch|transport|shipment))\s*[:\-]?\s*([A-Za-z]{3,20})", 1, 0.83),
    ],

    "delivery_date": [
        (r"(?:delivery\s+date|required\s+(?:delivery\s+)?date|lieferdatum)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.87),
        (r"(?:delivery\s+date|lieferdatum)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.87),
    ],

    "req_no": [
        (r"(?:req(?:uisition)?\s*(?:no\.?|number)\s*[:\-]?\s*)([A-Z0-9]{5,15})", 1, 0.86),
    ],

    "supplier_code": [
        (r"(?:supplier\s*(?:code|id|no\.?|number))\s*[:\-]?\s*([A-Z0-9]{3,15})", 1, 0.84),
    ],

    "iban": [
        (r"iban\s*[:\-]?\s*([A-Z]{2}\d{2}[A-Z0-9]{4,30})", 1, 0.92),
    ],

    "bic": [
        (r"(?:bic|swift)\s*[:\-]?\s*([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)", 1, 0.91),
    ],

    "vat_number": [
        (r"(?:vat\s*(?:reg\.?\s*)?(?:no\.?|number|#)\s*[:\-]?\s*)([A-Z]{2}[\d\s]{5,12})", 1, 0.88),
    ],

    "origin_airport": [
        (r"(?:airport\s+of\s+departure|departure\s+airport|origin\s+airport)\s*[:\-]?\s*([A-Z]{3})\b", 1, 0.88),
    ],

    "destination_airport": [
        (r"(?:airport\s+of\s+(?:destination|arrival)|destination\s+airport)\s*[:\-]?\s*([A-Z]{3})\b", 1, 0.88),
    ],

    "flight_number": [
        (r"(?:flight\s*(?:no\.?|number)\s*[:\-]?\s*)([A-Z]{2}\d{3,4})", 1, 0.88),
    ],

    "shipment_date": [
        (r"(?:shipment\s+date|date\s+of\s+shipment|verschiffungsdatum)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.86),
    ],

    "salesperson": [
        (r"(?:salesperson|sales\s+(?:person|rep\.?)|account\s+manager|sachbearbeiter)\s*[:\-]?\s*([A-Z][^\n\r]{3,40})", 1, 0.80),
    ],

    "test_standard": [
        (r"\b((?:acc\.?\s+to\s+)?(?:en|din|iso|astm|bs)\s*[\d\-\/\.]+)", 1, 0.85),
        (r"(?:test\s+standard|standard|specification)\s*[:\-]?\s*([A-Z]{2,5}\s*[\d\-\/\.]+)", 1, 0.83),
    ],

    "material_name": [
        (r"(?:material|product|article)\s*(?:name|description)?\s*[:\-]?\s*([A-Z][^\n\r]{3,60})", 1, 0.80),
    ],

    "exporter_name": [
        # Require colon or dash separator to prevent "Shipper/Exporter" tab-header
        # (no colon) from consuming the next cell's column labels as the value.
        (r"(?:exporter|exported\s+by|exporting\s+company)\s*[:\-]\s*([A-Z][^\n\r]{3,50})", 1, 0.83),
        # All-caps company: "HKT BEARINGS LIMITED" style (3-part name, APTA Form-I)
        (r"\b([A-Z]{2,8}\s+[A-Z]{4,15}\s+(?:LIMITED|LTD\.?))\b", 1, 0.74),
        # Email domain as last-resort exporter identifier (FTA cert with OCR noise)
        # "@hktbearings." in "sales@hktbearings. com" captures "hktbearings"
        (r"@([\w\d][\w\d\-]{3,})\.", 1, 0.58),
    ],

    "goods_description": [
        (r"(?:description\s+of\s+(?:goods?|cargo|articles?)|commodity|nature\s+of\s+goods?)\s*[:\-]?\s*([^\n\r]{5,80})", 1, 0.82),
    ],

    "certificate_number": [
        (r"(?:certificate\s*(?:no\.?|number|#))\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.88),
        # "Case-No. 639553" / "Kisten-Nr. / Case-No. 639553" — wooden crate / IPPC heat treatment certs
        (r"(?:case[-\s]*no\.?|kisten[-\s]*nr\.?)\s*(?:\/\s*case[-\s]*no\.?)?\s*[:\-]?\s*(\d{4,15})", 1, 0.83),
        # "Recording-No. (DE-NW1 49704 HT)" — IPPC treatment registration number
        (r"(?:recording[-\s]*no\.?)\s*[:\-]?\s*\(?([\w][\w\s\-]{3,30})\)?", 1, 0.78),
    ],

    "bank_name": [
        (r"(?:bank|bankverbindung|banque)\s*[:\-]?\s*([A-Z][^\n\r]{3,50})", 1, 0.80),
    ],

    "contact_email": [
        (r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,6})\b", 1, 0.90),
    ],

    "import_licence_no": [
        (r"(?:import\s+licen[cs]e?\s*(?:no\.?|number)?)\s*[:\-]?\s*([^\n\r]{3,40})", 1, 0.82),
    ],

    # ── Validation-aligned aliases ────────────────────────────────────────────
    # These use the same patterns as their base counterparts but with the
    # field names the ValidationAgent expects in its REQUIRED_FIELDS lists.

    # dc_006 — ValidationAgent requires "customer_po_ref" (buyer's PO on invoice)
    "customer_po_ref": [
        (r"(?:(?:customer|buyer|your)\s+p\.?o\.?\s*(?:no\.?|number|ref\.?|#)\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.90),
        (r"(?:p\.?o\.?\s*(?:no\.?|number|ref\.?|#)\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.86),
        (r"\b(TSL\/\d{5,6}(?:\/\d+)?)\b", 1, 0.94),
        (r"\b(TML\/\d{5,6}(?:\/\d+)?)\b", 1, 0.94),
        (r"\b(TMPVL\/\d{5,8}\/\d+)\b", 1, 0.94),
    ],

    # dc_004 — ValidationAgent requires "chargeable_weight" (numeric only, no unit)
    "chargeable_weight": [
        (r"(?:chargeable\s*(?:weight|wt\.?)|charge(?:able)?\s+wt\.?)\s*[:\-]?\s*([0-9,\.]+)(?:\s*(?:kgs?|lbs?|kg))?\b", 1, 0.88),
        (r"(?:gross\s*(?:weight|wt\.?)|gwt)\s*[:\-]?\s*([0-9,\.]+)(?:\s*(?:kgs?|lbs?|kg))?\b", 1, 0.82),
    ],

    # dc_007 — ValidationAgent requires "total_gross_weight" (numeric only, no unit)
    "total_gross_weight": [
        (r"(?:total\s+gross\s*(?:weight|wt\.?))\s*[:\-]?\s*([0-9,\.]+)(?:\s*(?:kgs?|lbs?|kg))?\b", 1, 0.88),
        # "Total Gross Wt. (Kg): 564" — unit in parentheses before colon
        (r"total\s+gross\s+wt\.?\s*(?:\([a-z]{1,5}\))?\s*[:\-]\s*([0-9,\.]+)", 1, 0.87),
        (r"(?:(?:total\s+)?gross\s*(?:weight|wt\.?)|gwt|g\.?w\.?)\s*[:\-]?\s*([0-9,\.]+)(?:\s*(?:kgs?|lbs?|kg))?\b", 1, 0.85),
        (r"gross\s+wt\.?\s*(?:\([a-z]{1,5}\))?\s*[:\-]\s*([0-9,\.]+)", 1, 0.83),
        # Japanese PL "Number of Packaging (Grand total) count net_kg gross_kg vol"
        (r"(?:grand\s+total\))\s+\d+\s+[0-9,\.]+\s+([0-9,\.]+)", 1, 0.82),
    ],

    # dc_007 — ValidationAgent requires "purchase_order_no"
    "purchase_order_no": [
        (r"(?:p\.?o\.?\s*(?:no\.?|#)\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.91),
        (r"(?:purchase\s+order\s*(?:no\.?|number|#)?\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.90),
        (r"\b(TSL\/\d{5,6}(?:\/\d+)?)\b", 1, 0.94),
        (r"\b(TML\/\d{5,6}(?:\/\d+)?)\b", 1, 0.94),
        (r"(?:marks\s*&?\s*nos?\.?\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.82),
        # "TSL ORDER NUMBER 2400002835" — labeled SAP 10-digit order number on RHI-style PLs
        (r"(?:tsl\s+order\s+number|customer\s+order\s+number)\s*[:\-]?\s*(\d{7,12})\b", 1, 0.86),
        # "AS PER PO 2400002835 DATED" — SAP PO reference embedded in goods description
        (r"(?:as\s+per\s+p\.?o\.?|per\s+p\.?o\.?)\s+(\d{7,12})\b", 1, 0.84),
    ],

    # dc_013 — ValidationAgent requires "mawb_number"
    "mawb_number": [
        # Label "MAWB-No.:" or "MAWB" followed by number; space-tolerant 8-digit block
        # e.g. "MAWB-No. : 932-7704 0515" or "MAWB 098-30811303"
        (r"(?:mawb|master\s+(?:air\s+)?waybill)\s*(?:[-\s]no\.?)?\s*[:\-]?\s*(\d{3}[-\s]?\d{4}[-\s]?\d{4})", 1, 0.93),
        # "MAWB 932-77040515" in Reference field (compact 11-char IATA format)
        (r"\bMAWB\b\s*(\d{3}[-\s]?\d{8})\b", 1, 0.91),
        # Bare IATA: NNN-NNNNNNNN (no space variant)
        (r"\b(\d{3}[-\s]\d{8})\b", 1, 0.82),
        # Space-tolerant fallback: "932-7704 0515"
        (r"\b(\d{3}[-\s]\d{4}\s\d{4})\b", 1, 0.80),
        # Ocean freight HBL: "HBL NO.:: 802026050006" — plays the same role as MAWB
        # for sea shipments; extracted here so dc_013 mawb_number requirement is satisfied.
        (r"(?:hbl\s*no\.?\s*[:\-]+\s*)(\d{6,20})", 1, 0.82),
    ],

    # ── dc_015 — RFQ ─────────────────────────────────────────────────────────
    "rfq_number": [
        # "RFQ No. : 921840750/1400149303/107" — require colon/dash to avoid matching "our RFQ No. and"
        (r"rfq\s*no\.?\s*[:\-]+\s*([A-Z0-9][A-Z0-9\/\-\.]{2,49})", 1, 0.93),
        (r"request\s+for\s+quotation\s*(?:no\.?|number|#)?\s*[:\-]+\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.89),
        (r"\b(TSL[-\/]RFQ[-\/]\d{4,8})\b", 1, 0.93),
        (r"\b(RFQ[-\/]\d{4,8})\b", 1, 0.88),
    ],
    "rfq_date": [
        # "RFQ Date :- 02.06.2026" — handle ":-" separator
        (r"rfq\s*date\s*[:\-]+\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.89),
        (r"rfq\s*date\s*[:\-]+\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.89),
        (r"date\s+of\s+rfq\s*[:\-]+\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.87),
    ],
    "rfq_due_date": [
        # "RFQ Due date :- 08.06.2026"
        (r"rfq\s+due\s+date\s*[:\-]+\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.91),
        (r"rfq\s+due\s+date\s*[:\-]+\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.91),
        (r"(?:submission\s+(?:deadline|date)|quote\s+(?:due\s+)?date|response\s+due)\s*[:\-]+\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.88),
        (r"(?:please\s+submit|offer\s+by|submit\s+by)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.84),
    ],
    "issuer": [
        (r"(?:issued\s+by|from|on\s+behalf\s+of)\s*[:\-]?\s*(Tata\s+(?:Steel|Motors)[^\n\r]{0,40})", 1, 0.88),
        (r"(Tata\s+Steel\s+(?:Limited|UK|Europe|Netherlands)[^\n\r]{0,30})", 1, 0.85),
    ],

    # ── dc_005 — DCC / Freight Booking ───────────────────────────────────────
    "dcc_number": [
        # "Dispatch Clearance Certificate No. 01 (DCC)" — number BEFORE "(DCC)"
        (r"(?:dispatch|despatch)\s+clearance\s+certificate\s*(?:no\.?|number|#)?\s*[:\-\.]*\s*(\d{1,10})\b", 1, 0.92),
        (r"(?:dcc\s*(?:no\.?|number|#|ref\.?)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\/\-\.]{1,24})", 1, 0.91),
        (r"(?:delivery\s+cost\s+confirmation\s*(?:no\.?|number|#)?\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\/\-\.]{1,24})", 1, 0.89),
        (r"(?:freight\s+booking\s*(?:no\.?|number|ref\.?)?\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.86),
        (r"\b(DCC[-\/]\d{4,8})\b", 1, 0.90),
    ],
    "dcc_date": [
        (r"(?:dcc\s+date|booking\s+date|confirmation\s+date)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.87),
        (r"(?:dcc\s+date|booking\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.87),
    ],
    "tata_po_number": [
        (r"(?:tata\s+(?:steel\s+)?p\.?o\.?\s*(?:no\.?|number|ref\.?)?\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.93),
        (r"(?:customer\s+p\.?o\.?\s*(?:no\.?|number|ref\.?)?\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.89),
        # Space-tolerant TSL/5 2602 variant: "PO No. TSL/5 2602/24000024"
        (r"\b(TSL\/\d{1,2}\s?\d{4,5}(?:\/[^\s,;]{1,15})?)\b", 1, 0.95),
        (r"\b(TSL\/\d{5,6}(?:\/\d+)?)\b", 1, 0.95),
        (r"\b(TML\/\d{5,6}(?:\/\d+)?)\b", 1, 0.95),
        (r"\b(TMPVL\/\d{5,8}\/\d+)\b", 1, 0.95),
        (r"(?:p\.?o\.?\s*(?:no\.?|number|ref\.?)?\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.85),
    ],
    "tata_entity": [
        (r"(Tata\s+(?:Steel\s+Limited|Steel\s+UK|Steel\s+Europe|Motors\s+Limited|Motors\s+Passenger\s+Vehicles\s+Limited|Limited))", 1, 0.94),
        (r"(?:consignee|customer|buyer)\s*[:\-]?\s*(Tata\s+[^\n\r]{3,50})", 1, 0.88),
    ],

    # ── dc_008 — Inspection Certificate ──────────────────────────────────────
    "ic_number": [
        (r"(?:i\.?c\.?\s*(?:no\.?|number|#|ref\.?)|inspection\s+certificate\s*(?:no\.?|number|#)?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.90),
        (r"\b(IC[-\/][A-Z0-9]{3,15})\b", 1, 0.87),
    ],
    "ic_release_date": [
        (r"(?:release\s+date|ic\s+date|inspection\s+date|date\s+of\s+inspection)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.87),
        (r"(?:release\s+date|ic\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.87),
    ],
    "order_number": [
        (r"(?:order\s*(?:no\.?|number|#|ref\.?)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.86),
        (r"(?:purchase\s+order\s*(?:no\.?|number)?\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.88),
    ],

    # ── dc_009 — DGD ─────────────────────────────────────────────────────────
    "un_number": [
        (r"(?:un\s*(?:no\.?|number|#)\s*[:\-]?\s*)(UN\s?\d{4})", 1, 0.93),
        (r"\b(UN\s?\d{4})\b", 1, 0.91),
        (r"(?:un\s*(?:no\.?|number|#)\s*[:\-]?\s*)(\d{4})", 1, 0.88),
    ],
    "proper_shipping_name": [
        (r"(?:proper\s+shipping\s+name|psn|shipping\s+name)\s*[:\-]?\s*([A-Z][^\n\r]{5,80})", 1, 0.87),
    ],
    "hazard_class": [
        (r"(?:class(?:\s+and\s+division)?|hazard\s+class|division)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)", 1, 0.88),
        (r"(?:iata\s+)?(?:class|hazard)\s*[:\-]?\s*(\d\.\d|\d)", 1, 0.85),
    ],
    "emergency_contact_1": [
        (r"(?:24\s*(?:hr|hour)\s+emergency|emergency\s+contact|emergency\s+(?:phone|tel\.?|number))\s*[:\-]?\s*([+\d\s\(\)\-\.]{7,20})", 1, 0.85),
        (r"(?:emergency)\s*[:\-]?\s*([+\d\s\(\)\-\.]{7,20})", 1, 0.80),
    ],

    # ── dc_010 — Order Acknowledgement ───────────────────────────────────────
    "confirmation_number": [
        (r"(?:order\s+(?:acknowledgement|confirmation)\s*(?:no\.?|number|#|ref\.?)|oa\s*(?:no\.?|#)|confirmation\s*(?:no\.?|number|#))\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.90),
        (r"\b(OA[-\/]\d{4,8})\b", 1, 0.88),
        (r"\b(CONF[-\/][A-Z0-9]{4,12})\b", 1, 0.86),
        # "SALES CONFIRMATION: SC74189-79008" — German/European sales agent format
        (r"sales\s+confirmation\s*[:\-]?\s*([A-Z]{1,4}\d{4,10}(?:-\w+)?)\b", 1, 0.87),
        # "Number/Date 311332/29.05.2026" — European OC header (Paul Wurth et al.)
        (r"number\s*\/\s*date\s*[:\-\s]*(\d{5,10})\b", 1, 0.85),
        # "TMT Ref. 55752287" — TMT/sender-reference label specific to some suppliers
        (r"(?:tmt|our|sender|oc)\s+ref\.?\s*(?:no\.?)?\s*[:\-]?\s*(\d{5,10})\b", 1, 0.87),
        # "Order Confirmation 55752461" — bare integer directly after heading
        (r"order\s+confirmation\s+(\d{5,10})\b", 1, 0.84),
        # OCR-concatenated: "orderconfirmation311332" (no spaces, scanned doc)
        (r"orderconfirmation(\d{5,10})", 1, 0.83),
        # "SC74189" standalone sales confirmation code
        (r"\b(SC\d{4,8}(?:-\w+)?)\b", 1, 0.84),
    ],
    "confirmation_date": [
        (r"(?:confirmation\s+date|acknowledgement\s+date|oa\s+date|order\s+date)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.87),
        (r"(?:confirmation\s+date|acknowledgement\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.87),
    ],
    "customer_po_number": [
        (r"(?:(?:customer|buyer|your)\s+p\.?o\.?\s*(?:no\.?|number|ref\.?|#)\s*[:\-]?\s*)([A-Z]{2,6}\/\d{4,8}(?:\/\d+)?)", 1, 0.91),
        (r"\b(TSL\/\d{5,6}(?:\/\d+)?)\b", 1, 0.94),
        (r"\b(TML\/\d{5,6}(?:\/\d+)?)\b", 1, 0.94),
    ],
    "customer_name": [
        (r"(?:customer|bill\s+to|sold\s+to|client)\s*[:\-]?\s*([A-Z][^\n\r]{3,50})", 1, 0.82),
        (r"(Tata\s+(?:Steel\s+Limited|Steel\s+UK|Motors\s+Limited)[^\n\r]{0,30})", 1, 0.88),
    ],

    # ── dc_011/dc_012 — TLL-issued invoices ──────────────────────────────────
    "total_invoice_value": [
        (r"(?:total\s+invoice\s+value|invoice\s+(?:total|value)|grand\s+total)\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.90),
        (r"(?:invoice\s+total|total\s+amount\s+(?:due|payable))\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.87),
        # TLL Sales Invoice totals: "VALUE COST GBP" or "VALUE GBP COST" or "VALUE CIFC EUR"
        # The COST/CIFC label marks the all-in cost, which is the total invoice value.
        (r"(\d[\d,\.]+)\s+(?:COST|CIFC)\s+(?:GBP|EUR|USD|INR)\b", 1, 0.85),
        (r"(\d[\d,\.]+)(?:GBP|EUR|USD|INR)\s+(?:COST|CIFC)\b", 1, 0.84),
        (r"\btotal\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.82),
        # A2 Commission Invoice: "TO BUYING COMMISSION @ 1.85 47.98" / "@ 1.85 393.16"
        # The amount immediately after the percentage rate is the total commission billed.
        (r"(?:to\s+)?buying\s+commission\s+@\s+[\d\.]+%?\s+([\d,\.]+)", 1, 0.82),
        # Indian service invoice (Digitide / TLL): "Total Amount After Tax: 4,408.57"
        # or "Total Amount Before Tax 4,408.57" — "due/payable" variants already handled above.
        (r"(?:total\s+(?:invoice\s+)?amount\s+(?:after|before)\s+tax)\s*[:\-]?\s*([\d,\.]+)", 1, 0.83),
    ],

    # ── dc_014 — Insurance Certificate ───────────────────────────────────────
    "certificate_date": [
        (r"(?:certificate\s+date|date\s+of\s+(?:issue|certificate|issuance))\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.88),
        (r"(?:certificate\s+date|date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.87),
    ],
    "policy_number": [
        (r"(?:policy\s*(?:no\.?|number|#)|insurance\s+policy\s*(?:no\.?|number)?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,29})", 1, 0.90),
        (r"\b(POL[-\/][A-Z0-9]{4,20})\b", 1, 0.87),
    ],
    "insurer": [
        (r"(?:insurer|underwriter|insurance\s+company|underwritten\s+by)\s*[:\-]?\s*([A-Z][^\n\r]{3,50})", 1, 0.84),
    ],

    # ── dc_017/dc_018 — Quality/Test & FTA Certificates ──────────────────────
    "issue_date": [
        (r"(?:issue\s+date|date\s+of\s+issue|issuance\s+date|date\s+issued)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.88),
        (r"(?:issue\s+date|date\s+of\s+issue)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.88),
        (r"(?:date)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.78),
    ],
    "conformance_status": [
        (r"\b(PASS|FAIL|CONFORMING|NON[-\s]CONFORMING|ACCEPTED|REJECTED)\b", 1, 0.90),
        (r"(?:conforms?\s+to|conformance|test\s+result|result)\s*[:\-]?\s*([A-Z][^\n\r]{1,30})", 1, 0.83),
    ],

    # ── dc_016 — Bill of Entry / Customs Release ──────────────────────────────
    "be_number": [
        (r"(?:b\.?e\.?\s*(?:no\.?|number|#)|bill\s+of\s+entry\s*(?:no\.?|number)?|entry\s*(?:no\.?|number)?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.90),
        (r"\b(BE[-\/]\d{4,15})\b", 1, 0.88),
    ],
    "be_date": [
        (r"(?:be\s+date|bill\s+of\s+entry\s+date|entry\s+date|assessment\s+date)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.87),
        (r"(?:be\s+date|entry\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.87),
    ],
    "importer_name": [
        (r"(?:importer|importer\s+of\s+record|imported\s+by)\s*[:\-]?\s*([A-Z][^\n\r]{3,50})", 1, 0.86),
        (r"(Tata\s+(?:Steel\s+Limited|Steel\s+UK|Motors\s+Limited|Limited))", 1, 0.83),
    ],
    "iec_number": [
        (r"(?:iec\s*(?:no\.?|number|#|code)|import\s+export\s+code)\s*[:\-]?\s*([A-Z0-9]{8,15})", 1, 0.88),
    ],

    # ── dc_019 — Quotation / RFQ Response ────────────────────────────────────
    "quotation_number": [
        (r"(?:quotation\s*(?:no\.?|number|#|ref\.?)|quote\s*(?:no\.?|number|#)|offer\s*(?:no\.?|number|#))\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.90),
        (r"\b(QT[-\/][A-Z0-9]{4,15})\b", 1, 0.87),
        (r"\b(OFF[-\/][A-Z0-9]{4,15})\b", 1, 0.85),
    ],
    "quotation_date": [
        (r"(?:quotation\s+date|quote\s+date|offer\s+date|date\s+of\s+(?:quotation|quote))\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.88),
        (r"(?:quotation\s+date|quote\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.88),
        (r"(?:date)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.77),
    ],

    # ── dc_020 — Remittance Advice ────────────────────────────────────────────
    "remittance_date": [
        (r"(?:remittance\s+date|payment\s+date|date\s+of\s+(?:payment|remittance))\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.88),
        (r"(?:remittance\s+date|payment\s+date)\s*[:\-]?\s*(\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})", 1, 0.88),
        (r"(?:value\s+date|date)\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", 1, 0.78),
    ],
    "payer_name": [
        (r"(?:payer|remitter|paying\s+(?:party|company)|payment\s+from)\s*[:\-]?\s*([A-Z][^\n\r]{3,50})", 1, 0.82),
        (r"(Tata\s+(?:Steel\s+Limited|Steel\s+UK|Motors\s+Limited|Limited))", 1, 0.83),
    ],
    "payment_amount": [
        (r"(?:payment\s+amount|amount\s+(?:paid|remitted|transferred|sent)|total\s+(?:paid|remitted))\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.89),
        (r"(?:amount|total)\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\d[\d,\.]+)", 1, 0.82),
    ],
    "reference_number": [
        (r"(?:reference\s*(?:no\.?|number|#)|ref\.?\s*(?:no\.?|number|#)?|bank\s+ref)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\.]{2,24})", 1, 0.86),
    ],
}

# ── Fields to extract per document class ─────────────────────────────────────
FIELDS_BY_CLASS: dict[str, list[str]] = {
    "dc_001": ["po_number", "po_date", "entity", "supplier_name", "supplier_code",
               "currency", "total_order_value", "incoterms", "payment_terms",
               "final_destination", "port_of_entry", "mode_of_dispatch",
               "country_of_origin", "import_licence_no", "req_no", "goods_description",
               "delivery_date"],
    "dc_002": ["po_number", "po_date", "entity", "supplier_name", "supplier_code",
               "currency", "total_order_value", "incoterms", "payment_terms",
               "final_destination", "port_of_entry", "mode_of_dispatch",
               "country_of_origin", "import_licence_no", "req_no", "goods_description"],
    "dc_003": ["po_number", "po_date", "entity", "supplier_name", "supplier_code",
               "currency", "total_order_value", "incoterms", "payment_terms",
               "goods_description", "delivery_date"],
    "dc_004": ["awb_number", "hawb_number", "shipper_name", "consignee_name",
               "origin_airport", "destination_airport", "flight_number",
               "shipment_date", "chargeable_weight", "gross_weight", "pieces",
               "goods_description"],
    "dc_005": ["dcc_number", "dcc_date", "tata_po_number", "tata_entity",
               "supplier_name", "goods_description"],
    "dc_006": ["invoice_number", "invoice_date", "supplier_name", "currency",
               "total_amount", "net_amount", "payment_terms", "iban", "bic",
               "vat_number", "bank_name", "customer_po_ref", "contact_email"],
    "dc_007": ["supplier_name", "shipper_name", "consignee_name",
               "total_gross_weight", "gross_weight", "pieces",
               "goods_description", "country_of_origin", "purchase_order_no"],
    "dc_008": ["ic_number", "ic_release_date", "order_number", "supplier_name",
               "goods_description"],
    "dc_009": ["un_number", "proper_shipping_name", "hazard_class",
               "shipper_name", "consignee_name", "emergency_contact_1",
               "goods_description"],
    "dc_010": ["confirmation_number", "confirmation_date", "supplier_name",
               "customer_po_number", "customer_name", "salesperson",
               "delivery_date"],
    "dc_011": ["invoice_number", "invoice_date", "customer_name",
               "total_invoice_value", "currency", "awb_number", "po_number"],
    "dc_012": ["invoice_number", "invoice_date", "customer_name",
               "total_invoice_value", "currency", "awb_number", "po_number"],
    "dc_013": ["invoice_number", "invoice_date", "currency", "total_amount",
               "net_amount", "mawb_number", "awb_number", "hawb_number",
               "shipper_name", "consignee_name", "payment_terms"],
    "dc_014": ["certificate_number", "certificate_date", "policy_number",
               "insurer", "goods_description", "consignee_name"],
    "dc_015": ["rfq_number", "rfq_date", "rfq_due_date", "issuer",
               "supplier_name", "goods_description"],
    "dc_016": ["be_number", "be_date", "importer_name", "iec_number",
               "supplier_name", "invoice_number", "currency", "total_amount"],
    "dc_017": ["certificate_number", "issue_date", "supplier_name",
               "conformance_status", "material_name", "test_standard"],
    "dc_018": ["exporter_name", "country_of_origin", "issue_date",
               "consignee_name", "goods_description", "incoterms"],
    "dc_019": ["quotation_number", "quotation_date", "supplier_name",
               "total_amount", "currency", "goods_description"],
    "dc_020": ["remittance_date", "payer_name", "payment_amount",
               "currency", "reference_number"],
}

# ── Per-class regex field exclusions ─────────────────────────────────────────
# Fields that the REGEX fallback should NOT attempt to extract for a given
# document class — typically because the field is context-dependent and the
# broad fallback patterns produce false positives (the AI-first path handles
# these correctly using FIELD_DESCRIPTIONS).
REGEX_SKIP_FIELDS: dict[str, set[str]] = {
    # In an RFQ the supplier is often not named explicitly; the broad company-name
    # fallback grabs "TATA STEEL LIMITED" from the body text (the buyer).
    "dc_015": {"supplier_name"},
    # In a quotation/RFQ response the issuer is always the supplier — the broad
    # issuer pattern can incorrectly pick up the buyer reference.
    "dc_019": {"issuer"},
}

# ── Semantic field descriptions for AI-based extraction ───────────────────────
# Provided to Claude so it can distinguish contextually similar fields
# (e.g. supplier vs buyer, issuer vs consignee) from raw document text.
FIELD_DESCRIPTIONS: dict[str, str] = {
    # ── Order / Contract references ──────────────────────────────────────────
    "po_number":          "Purchase order number issued BY THE BUYER (e.g. TSL/56500, TML/32662, 4700155396). NOT a supplier reference.",
    "po_date":            "Date the purchase order was placed/issued (as it appears in the document).",
    "req_no":             "Internal requisition or request number that preceded this order.",
    "import_licence_no":  "Import licence or permit number.",
    "customer_po_ref":    "The customer's own purchase order reference number (on an invoice, this is what the buyer quoted).",
    "customer_po_number": "The customer's purchase order number (used on order acknowledgements).",
    "order_number":       "Generic order or inspection order number.",
    "purchase_order_no":  "Purchase order number as referenced on packing lists.",
    "confirmation_number":"Order confirmation or acknowledgement reference number.",
    "confirmation_date":  "Date the order confirmation was issued.",
    # ── Invoice / Financial ───────────────────────────────────────────────────
    "invoice_number":     "Invoice reference number assigned by the SELLER (NOT the buyer's PO).",
    "invoice_date":       "Date this invoice was issued.",
    "currency":           "Currency code (e.g. GBP, EUR, USD, INR, AED).",
    "total_amount":       "Total monetary amount due/payable including all taxes and charges.",
    "total_invoice_value":"Total value shown on this invoice.",
    "total_order_value":  "Total value of the purchase order.",
    "net_amount":         "Net amount before taxes/extra charges.",
    "vat_amount":         "VAT or GST tax amount.",
    "payment_amount":     "Amount being paid/remitted in a remittance advice.",
    "payment_terms":      "Payment terms (e.g. 30 days net, Net 60, immediate, as per contract).",
    "iban":               "IBAN bank account number.",
    "bic":                "BIC/SWIFT bank identifier code.",
    "vat_number":         "Supplier's VAT registration number.",
    "bank_name":          "Name of the supplier's bank.",
    "contact_email":      "Contact email address shown on the document.",
    # ── Company / Party names ────────────────────────────────────────────────
    "supplier_name":      "Name of the SUPPLIER or VENDOR — the company PROVIDING goods or services. NOT the buyer. In an RFQ, this may be the company being invited to quote (may not be named explicitly, return null if absent).",
    "supplier_code":      "Buyer's internal code assigned to this supplier (e.g. V-1234, SUP-009).",
    "issuer":             "Name of the company that ISSUED or CREATED this document (the document's owner/sender).",
    "entity":             "The buyer/importer entity that placed the order (e.g. TML, TSL, TMPVL).",
    "customer_name":      "Name of the CUSTOMER (buyer/importer) company.",
    "importer_name":      "Name of the importer of record as shown on customs/bill of entry.",
    "payer_name":         "Name of the company making the payment in a remittance advice.",
    "exporter_name":      "Name of the exporter (the party shipping goods out).",
    "consignee_name":     "Name of the consignee — the party who will RECEIVE the shipment.",
    "shipper_name":       "Name of the shipper — the party who SENT the shipment.",
    "insurer":            "Name of the insurance company providing coverage.",
    "salesperson":        "Name or ID of the salesperson handling this order.",
    # ── RFQ / Quotation ───────────────────────────────────────────────────────
    "rfq_number":         "Request for Quotation reference number (a single clean identifier, no line breaks).",
    "rfq_date":           "Date the RFQ was issued.",
    "rfq_due_date":       "Deadline date/time by which quotes must be submitted.",
    "quotation_number":   "Quotation or offer reference number issued by the supplier in response to an RFQ.",
    "quotation_date":     "Date of the quotation.",
    # ── Air Freight / Logistics ───────────────────────────────────────────────
    "awb_number":         "Master Air Waybill (MAWB) number.",
    "hawb_number":        "House Air Waybill (HAWB) number.",
    "mawb_number":        "Master Air Waybill number (same as awb_number — use whichever label appears).",
    "origin_airport":     "3-letter IATA code or name of the origin airport.",
    "destination_airport":"3-letter IATA code or name of the destination airport.",
    "flight_number":      "Flight number used for the shipment.",
    "shipment_date":      "Date the shipment was dispatched.",
    "chargeable_weight":  "Chargeable weight (the higher of actual vs volumetric) in kg.",
    "gross_weight":       "Actual gross weight in kg.",
    "total_gross_weight": "Total gross weight across all packages.",
    "total_net_weight":   "Total net weight of goods (excluding packaging).",
    "pieces":             "Number of pieces / packages.",
    "incoterms":          "Incoterms code (e.g. CIF, FOB, EXW, DAP, DDP).",
    "final_destination":  "Final delivery destination city or address.",
    "port_of_entry":      "Port or airport of entry into the destination country.",
    "mode_of_dispatch":   "Mode of transport (Air, Sea, Road, Rail).",
    "country_of_origin":  "Country where the goods were manufactured or produced.",
    # ── DCC / Dispatch ───────────────────────────────────────────────────────
    "dcc_number":         "Dispatch Clearance Certificate reference number.",
    "dcc_date":           "Date of the Dispatch Clearance Certificate.",
    "tata_po_number":     "Tata purchase order number referenced on the DCC.",
    "tata_entity":        "Tata group entity name shown on the DCC.",
    # ── Inspection / Certificates ─────────────────────────────────────────────
    "ic_number":          "Inspection Certificate reference number.",
    "ic_release_date":    "Date the inspection was released/signed off.",
    "certificate_number": "Certificate reference number (inspection, insurance, quality, etc.).",
    "certificate_date":   "Date the certificate was issued.",
    "issue_date":         "Date this document was issued or signed.",
    "policy_number":      "Insurance policy number.",
    "conformance_status": "Conformance or test result status (e.g. PASS, FAIL, CONFORM).",
    "material_name":      "Name or grade of the material tested.",
    "test_standard":      "Standard or specification against which testing was performed.",
    # ── Dangerous Goods ───────────────────────────────────────────────────────
    "un_number":          "UN number for the dangerous good (e.g. UN1234).",
    "proper_shipping_name":"Proper shipping name for the dangerous good.",
    "hazard_class":       "IATA/IMDG hazard class (e.g. Class 3, Class 9).",
    "emergency_contact_1":"Emergency response contact number.",
    # ── Customs / Bill of Entry ───────────────────────────────────────────────
    "be_number":          "Bill of Entry number assigned by customs.",
    "be_date":            "Date of the Bill of Entry.",
    "iec_number":         "Importer-Exporter Code (IEC) number.",
    # ── Goods ─────────────────────────────────────────────────────────────────
    "goods_description":  "Brief description of the goods, materials, or services (1-2 sentences max).",
    # ── Financial / Remittance ────────────────────────────────────────────────
    "remittance_date":    "Date of the payment/remittance.",
    "reference_number":   "Payment reference or bank transaction reference number.",
    # ── Buying commission ─────────────────────────────────────────────────────
    "buying_commission_pct": "Buying commission percentage.",
    "buying_commission_amt": "Buying commission monetary amount.",
    "customs_duty":       "Customs duty amount.",
    "total_cif_value":    "Total CIF (Cost + Insurance + Freight) value.",
    "subtotal":           "Subtotal before taxes.",
}

CONFIDENCE_DAMPEN = {
    "ZERO_SHOT": 0.88,
    "LEARNING":  0.92,
    "LEARNED":   1.00,
    "OPTIMISED": 1.00,
}


class ExtractionAgent(BaseAgent):
    """
    AI-first extraction agent with a learning lifecycle.

    When ANTHROPIC_API_KEY is set:
    ─────────────────────────────
      ZERO_SHOT        AI reads blind.  Any result passes (system is learning).
      LEARNING         AI uses prior examples as hints.  Lenient pass threshold.
      LEARNED_PROPOSED Same as LEARNING — schema proposed, awaiting confirmation.
      LEARNED          AI extracts against confirmed schema.  Strict proofreading.
      OPTIMISED        Fast patterns first → AI fallback → strict proofreading.

    After every extraction, SchemaLearnerAgent records field statistics and
    advances the learning stage automatically when quality thresholds are met.

    Without ANTHROPIC_API_KEY:
    ──────────────────────────
      Falls back to the original regex-only pipeline (Tiers 1, 2, 3) so the
      system degrades gracefully rather than failing entirely.
    """

    name = "ExtractionAgent"

    def extract(
        self,
        doc: Document,
        pdf_bytes: bytes | None = None,
    ) -> list[ExtractedField] | None:
        if not doc.document_class_id:
            return []

        dc_id         = doc.document_class_id
        target_fields = FIELDS_BY_CLASS.get(dc_id, [])
        # NOTE: do NOT bail on empty target_fields here.
        # ZERO_SHOT discovery runs without a predefined field list — the AI finds
        # everything itself. target_fields is only needed for LEARNING+ stages.

        from app.pipeline.states import PipelineState
        from app.agents.schema_learner import SchemaLearnerAgent
        from app.models.extracted_field import ExtractedField as EF

        proofreader    = ProofreadingAgent(self.db)
        schema_learner = SchemaLearnerAgent(self.db)

        # ── Resolve doc class name + DocumentTypeProfile ──────────────────────
        dc_obj  = self.db.query(DocumentClass).filter(DocumentClass.id == dc_id).first()
        dc_name = dc_obj.name if dc_obj else dc_id
        dtp     = self._get_dtp(dc_id)

        # ── Resolve learning stage: variant-level takes priority ──────────────
        # Priority: variant.learning_stage → dtp.learning_stage → ZERO_SHOT
        variant = None
        if doc.variant_id:
            variant = self.db.query(DocumentVariant).filter(
                DocumentVariant.id == doc.variant_id
            ).first()
        stage = (
            variant.learning_stage if variant
            else (dtp.learning_stage if dtp else "ZERO_SHOT")
        )

        logger.info("doc %s: extraction start | class=%s stage=%s api=%s",
                    doc.id, dc_id, stage, "yes" if settings.anthropic_api_key else "no")

        # ══════════════════════════════════════════════════════════════════════
        #  AI-FIRST PATH  (when ANTHROPIC_API_KEY is set)
        # ══════════════════════════════════════════════════════════════════════
        if settings.anthropic_api_key:
            fields = self._extract_ai_first(
                doc, pdf_bytes, dc_id, dc_name, target_fields,
                stage, dtp, proofreader, schema_learner,
            )
        else:
            # ══════════════════════════════════════════════════════════════════
            #  REGEX FALLBACK  (no API key — original 3-tier behaviour)
            # ══════════════════════════════════════════════════════════════════
            fields = self._extract_regex_fallback(
                doc, pdf_bytes, dc_id, target_fields, stage, proofreader,
            )

        return fields

    # ── AI-first extraction ────────────────────────────────────────────────────

    def _extract_ai_first(
        self,
        doc: Document,
        pdf_bytes: bytes | None,
        dc_id: str,
        dc_name: str,
        target_fields: list[str],
        stage: str,
        dtp,
        proofreader: "ProofreadingAgent",
        schema_learner: "SchemaLearnerAgent",
    ):
        from app.pipeline.states import PipelineState
        from app.models.extracted_field import ExtractedField as EF

        # ── Extract text + determine document physicality ─────────────────────
        text     = _extract_text_layer(pdf_bytes)
        has_text = bool(text and len(text.strip()) > 200)

        # ── Pre-compute page sample (used by ALL extraction paths) ────────────
        # Check if this variant has a learned page skip list from PageProfileAgent.
        # If yes, pass it so confirmed dead pages are excluded from the sample.
        page_meta: PageSampleMeta | None = None
        learned_skip: list[int] = []
        if doc.variant_id and has_text:
            try:
                from app.agents.page_profile import PageProfileAgent as _PPA
                _ppa = _PPA(self.db)
                learned_skip = _ppa.get_confident_skip(doc.variant_id)
                if learned_skip:
                    logger.info("doc %s: PageProfile skip list: %s", doc.id, learned_skip)
            except Exception as _e:
                logger.debug("doc %s: PageProfile lookup failed: %s", doc.id, _e)

        if pdf_bytes and has_text:
            # No learned skip list yet → send the full document so the AI sees
            # every page and can build a complete schema.  Once PageProfileAgent
            # has confirmed which pages are empty boilerplate, we switch to the
            # 80 000-char budget that skips those pages.
            char_budget = 80_000 if learned_skip else 300_000
            page_meta = _smart_page_sample(
                pdf_bytes, char_budget=char_budget, force_skip=learned_skip or None
            )
            sampled_text = page_meta.text if page_meta.text else text[:char_budget]
        else:
            sampled_text = text[:80_000]

        raw: dict[str, tuple[str, float]] = {}
        method = ""

        # ── Schema resolution: variant → class → hardcoded ──────────────────────
        # 1. Variant-level schema (most specific — operator confirmed for THIS issuer+format)
        # 2. Class-level schema   (DocumentTypeProfile — applies to all variants of this class)
        # 3. Hardcoded FIELDS_BY_CLASS  (fallback when nothing confirmed yet)
        variant_schema = None
        if stage != "ZERO_SHOT":
            if doc.variant_id:
                v = self.db.query(DocumentVariant).filter(
                    DocumentVariant.id == doc.variant_id
                ).first()
                if v and v.field_schema_json:
                    variant_schema = json.loads(v.field_schema_json)

            if variant_schema:
                target_fields = list(variant_schema.keys())
                logger.debug("doc %s: using variant schema (%d fields)", doc.id, len(target_fields))
            elif dtp and dtp.field_schema_json:
                confirmed = list(json.loads(dtp.field_schema_json).keys())
                if confirmed:
                    target_fields = confirmed
                    logger.debug("doc %s: using class schema (%d fields)", doc.id, len(target_fields))

        # ── If past ZERO_SHOT but NO schema confirmed yet (neither variant nor class),
        # fall back to free-form discovery so the operator sees everything and can
        # pick the fields. FIELDS_BY_CLASS is a hardcoded default, not a confirmation.
        has_confirmed_schema = bool(
            variant_schema or (dtp and dtp.field_schema_json)
        )
        if stage != "ZERO_SHOT" and not has_confirmed_schema:
            logger.info(
                "doc %s: stage=%s but no confirmed schema → forcing ZERO_SHOT free-form discovery",
                doc.id, stage,
            )
            stage = "ZERO_SHOT"

        # ══════════════════════════════════════════════════════════════════════
        #  ZERO_SHOT — pure AI discovery: pass the whole document, find everything
        # ══════════════════════════════════════════════════════════════════════
        # discovery_split stores (core_raw, additional_raw, tables_raw) so the
        # persist block can write them with the right extraction_method and field_type.
        discovery_split: tuple[dict, dict, dict] | None = None

        if stage == "ZERO_SHOT":
            if has_text:
                # ERP-focused split discovery: core scalars + additional scalars + tables.
                # sampled_text was built by _smart_page_sample above — already page-sampled.
                core_raw, additional_raw, tables_raw = _extract_via_ai_free_form(
                    sampled_text, dc_name
                )
                discovery_split = (core_raw, additional_raw, tables_raw)
                # Flat merge for schema learner (scalar fields only — tables are separate)
                raw    = {**core_raw, **additional_raw, **tables_raw}
                method = "AI_DISCOVERY"
                logger.info(
                    "doc %s: ZERO_SHOT AI discovery → %d core + %d additional + %d tables (%s)",
                    doc.id, len(core_raw), len(additional_raw), len(tables_raw),
                    ", ".join(tables_raw.keys()) or "none",
                )

            elif pdf_bytes:
                # Scanned / image-only PDF: pass directly to vision API in free-form mode
                raw    = _extract_vision_api(pdf_bytes, dc_id, [])  # empty list = find everything
                method = "AI_VISION_DISCOVERY"
                logger.info("doc %s: ZERO_SHOT vision discovery → %d fields",
                            doc.id, len(raw))

        # ══════════════════════════════════════════════════════════════════════
        #  OPTIMISED — try fast generated patterns first
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "OPTIMISED" and dtp and dtp.generated_patterns_json:
            try:
                patterns = json.loads(dtp.generated_patterns_json)
                raw = _apply_generated_patterns(text or "", patterns, target_fields)
                if _sufficient_quality(raw, target_fields, threshold=0.85):
                    method = "FAST_PATTERNS"
                    logger.info("doc %s: OPTIMISED fast-path succeeded (%d fields)",
                                doc.id, len(raw))
                else:
                    raw    = {}
                    method = ""
            except Exception as exc:
                logger.debug("doc %s: fast-pattern parse error: %s", doc.id, exc)
                raw = {}

        # ══════════════════════════════════════════════════════════════════════
        #  LEARNING / LEARNED / LEARNED_PROPOSED — AI against known field list
        #  (or free-form discovery if no schema confirmed yet for this class)
        # ══════════════════════════════════════════════════════════════════════
        if stage != "ZERO_SHOT" and not raw:
            if not target_fields:
                # No confirmed schema yet despite being past ZERO_SHOT — fall back
                # to free-form discovery (same as ZERO_SHOT) so we always get something.
                logger.info("doc %s: %s stage but no target_fields → free-form fallback",
                            doc.id, stage)
                if has_text:
                    core_raw, additional_raw, tables_raw = _extract_via_ai_free_form(
                        sampled_text, dc_name
                    )
                    discovery_split = (core_raw, additional_raw, tables_raw)
                    raw    = {**core_raw, **additional_raw, **tables_raw}
                    method = f"AI_DISCOVERY_{stage}"
                elif pdf_bytes:
                    raw    = _extract_vision_api(pdf_bytes, dc_id, [])
                    method = f"AI_VISION_DISCOVERY_{stage}"
            else:
                hints = schema_learner.get_schema_hints(dc_id)
                if has_text:
                    raw    = _extract_via_ai_text(sampled_text, target_fields, dc_name, hints)
                    method = f"AI_TEXT_{stage}"
                    logger.info("doc %s: AI text extraction → %d fields", doc.id, len(raw))

                if not raw and pdf_bytes:
                    raw    = _extract_vision_api(pdf_bytes, dc_id, target_fields)
                    method = f"AI_VISION_{stage}"
                    logger.info("doc %s: AI vision → %d fields", doc.id, len(raw))

        # ── Always record field statistics (even on empty result) ─────────────
        schema_learner.record_extraction(doc, raw, text or "", target_fields)

        if not raw:
            logger.warning("doc %s: AI extraction returned no fields → NEEDS_REVIEW", doc.id)
            return None

        # ── Persist extracted fields ──────────────────────────────────────────
        # Tier 1 = text layer (fast patterns, AI-on-text, AI discovery from text)
        # Tier 2 = OCR       (only when method explicitly contains "OCR")
        # Tier 3 = AI Vision  (image input to Claude)
        tier = (
            1 if "FAST" in method
            else (3 if "VISION" in method
                  else (2 if "OCR" in method
                        else 1))
        )
        # ── Normalise field names against alias map ───────────────────────────
        # For LEARNING/LEARNED/OPTIMISED stages: if the schema has a canonical
        # name (e.g. "po_number") with aliases (e.g. "order_no", "order_number"),
        # remap whatever Claude extracted so the DB and API always return the
        # canonical name regardless of how each supplier labels the field.
        # Skipped for ZERO_SHOT discovery (no schema exists yet).
        if not discovery_split:
            active_schema = variant_schema or (
                json.loads(dtp.field_schema_json) if dtp and dtp.field_schema_json else None
            )
            if active_schema:
                raw = self._normalise_fields(raw, active_schema)
                logger.debug(
                    "doc %s: alias normalisation applied → %d fields after remap",
                    doc.id, len(raw),
                )

        # ── Enrich with spatial locations (page + bbox) via PyMuPDF ─────────────
        # Run a quick text search for each extracted scalar value to find which
        # page and pixel region it came from.  Stored on ExtractedField so the
        # viewer can jump directly to the right page and draw an exact highlight.
        all_scalars = dict(raw) if not discovery_split else {}
        if discovery_split:
            core_raw, additional_raw, tables_raw = discovery_split
            all_scalars = {**core_raw, **additional_raw}   # tables excluded (JSON values)
        _sampled_pages = page_meta.pages_used if page_meta else None
        _locations = self._enrich_with_locations(all_scalars, pdf_bytes, pages_used=_sampled_pages)

        if discovery_split:
            # ZERO_SHOT split: persist core scalars (pre-checked in UI),
            # then additional scalars (unchecked, collapsible),
            # then table fields (field_type="table", displayed as expandable tables).
            table_names = set(tables_raw.keys())

            fields = self._persist_fields(
                doc, core_raw,
                tier=tier,
                model=f"{settings.anthropic_model}-{method.lower()}",
                method="AI_DISCOVERY",
                commit=False,
                delete_first=True,
                locations=_locations,
            )
            if additional_raw:
                extra = self._persist_fields(
                    doc, additional_raw,
                    tier=tier,
                    model=f"{settings.anthropic_model}-{method.lower()}",
                    method="AI_DISCOVERY_ADDITIONAL",
                    commit=False,
                    delete_first=False,
                    locations=_locations,
                )
                fields = (fields or []) + (extra or [])
            if tables_raw:
                table_fields = self._persist_fields(
                    doc, tables_raw,
                    tier=tier,
                    model=f"{settings.anthropic_model}-{method.lower()}",
                    method="AI_DISCOVERY_TABLE",
                    commit=False,
                    delete_first=False,
                    table_field_names=table_names,
                    # no locations for table fields — value is JSON array
                )
                fields = (fields or []) + (table_fields or [])
        else:
            fields = self._persist_fields(
                doc, raw,
                tier=tier,
                model=f"{settings.anthropic_model}-{method.lower()}",
                method=method,
                commit=False,
                locations=_locations,
            )

        # ── Proofreading gate ─────────────────────────────────────────────────
        # ZERO_SHOT / LEARNING / LEARNED_PROPOSED: lenient — always pass as long
        # as we extracted at least one field. System is still building confidence.
        # LEARNED / OPTIMISED: strict — confirmed schema is the truth.
        pr = proofreader.check(doc, fields, tier=tier)
        early_stage = stage in ("ZERO_SHOT", "LEARNING", "LEARNED_PROPOSED")
        passes = pr.passed or (early_stage and len(raw) >= 1)

        if passes:
            self.db.commit()

            # ── Store page metadata on the document ───────────────────────────
            if page_meta:
                doc.pages_total        = page_meta.pages_total
                doc.pages_sampled_json = json.dumps(page_meta.pages_used)
                doc.pages_skipped_count = page_meta.pages_skipped
                self.db.commit()

            self.transition(
                doc, PipelineState.PROOFREADING,
                pr.as_event_detail(tier)
                + (f" [learning:{stage}]" if early_stage else "")
                + (f" [pages:{doc.pages_total or '?'}|used:{len(page_meta.pages_used) if page_meta else '?'}|skipped:{doc.pages_skipped_count or 0}]" if page_meta else ""),
            )

            # ── Update PageProfileAgent with this document's page attribution ─
            # Runs for any document with a variant — stage-agnostic so the profile
            # starts building from the very first document seen for each variant.
            if doc.variant_id and page_meta:
                try:
                    from app.agents.page_profile import PageProfileAgent as _PPA
                    fresh_fields = self.db.query(EF).filter(EF.document_id == doc.id).all()
                    _PPA(self.db).record_and_update(doc, fresh_fields, pdf_bytes, page_meta)
                except Exception as _pp_exc:
                    logger.warning("doc %s: PageProfileAgent failed (non-fatal): %s",
                                   doc.id, _pp_exc)

            logger.info("doc %s: AI extraction PASS (stage=%s, %d fields, method=%s, "
                        "pages=%s used=%d skipped=%d)",
                        doc.id, stage, len(fields), method,
                        doc.pages_total or "?",
                        len(page_meta.pages_used) if page_meta else 0,
                        doc.pages_skipped_count or 0)
            return self.db.query(EF).filter(EF.document_id == doc.id).all()

        # Strict stage failed proofreading → NEEDS_REVIEW
        self.db.rollback()
        self.transition(doc, PipelineState.PROOFREADING, pr.as_event_detail(tier))
        logger.warning("doc %s: AI extraction FAIL proofreading (stage=%s) → NEEDS_REVIEW",
                       doc.id, stage)
        return None

    # ── Regex fallback (no API key) ────────────────────────────────────────────

    def _extract_regex_fallback(
        self,
        doc: Document,
        pdf_bytes: bytes | None,
        dc_id: str,
        target_fields: list[str],
        stage: str,
        proofreader: "ProofreadingAgent",
    ):
        """Original 3-tier regex pipeline — used when no API key is configured."""
        from app.pipeline.states import PipelineState
        from app.models.extracted_field import ExtractedField as EF

        dampen = CONFIDENCE_DAMPEN.get(stage, 0.88)

        # Strip fields that are known to produce false-positives via regex for
        # this document class (AI handles them properly via FIELD_DESCRIPTIONS).
        skip = REGEX_SKIP_FIELDS.get(dc_id, set())
        if skip:
            target_fields = [f for f in target_fields if f not in skip]

        # Tier 1: text layer + regex
        text1 = _extract_text_layer(pdf_bytes)
        if text1:
            raw1 = _apply_patterns(text1, target_fields, dampen)
            if raw1:
                fields1 = self._persist_fields(doc, raw1, tier=1,
                                               model="regex-tier1",
                                               method="TEXT_LAYER", commit=False)
                pr1 = proofreader.check(doc, fields1, tier=1)
                if pr1.passed:
                    self.db.commit()
                    self.transition(doc, PipelineState.PROOFREADING, pr1.as_event_detail(1))
                    return self.db.query(EF).filter(EF.document_id == doc.id).all()
                self.db.rollback()
                self.transition(doc, PipelineState.PROOFREADING, pr1.as_event_detail(1))
                self.transition(doc, PipelineState.EXTRACTING,
                                "Tier 1 failed proofreading. Escalating to Vision.")

        logger.warning("doc %s: regex fallback exhausted → NEEDS_REVIEW", doc.id)
        return None

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _get_dtp(self, doc_class_id: str):
        """Find the first active DocumentTypeProfile for this doc class."""
        try:
            from app.models.client import DocumentTypeProfile
            return (
                self.db.query(DocumentTypeProfile)
                .filter(
                    DocumentTypeProfile.document_class_id == doc_class_id,
                    DocumentTypeProfile.active.is_(True),
                )
                .first()
            )
        except Exception:
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _alias_map(schema: dict) -> dict[str, str]:
        """
        Build a reverse lookup: raw_alias → canonical_name.
        New schema format:  {canonical: {"aliases": [...], ...}}
        Old schema format:  {canonical: {"source_field": "...", ...}}  (backward compat)
        """
        mapping: dict[str, str] = {}
        for canonical, meta in schema.items():
            mapping[canonical] = canonical  # canonical always maps to itself
            if isinstance(meta, dict):
                for alias in meta.get("aliases", []):
                    mapping[alias] = canonical
                src = meta.get("source_field")
                if src:
                    mapping[src] = canonical
        return mapping

    @staticmethod
    def _normalise_fields(
        raw: dict[str, tuple[str, float]],
        schema: dict,
    ) -> dict[str, tuple[str, float]]:
        """
        Remap extracted field names to their canonical names using the schema alias map.
        If a raw name appears as an alias for a canonical, the output uses the canonical.
        Unknown raw names are passed through unchanged (schema may grow over time).
        """
        alias_map = ExtractionAgent._alias_map(schema)
        result: dict[str, tuple[str, float]] = {}
        for raw_name, val_conf in raw.items():
            canonical = alias_map.get(raw_name, raw_name)
            # Keep highest-confidence value if two aliases map to the same canonical
            if canonical not in result or val_conf[1] > result[canonical][1]:
                result[canonical] = val_conf
        return result

    def _persist_fields(
        self,
        doc: Document,
        raw_fields: dict[str, tuple[str, float]],
        tier: int,
        model: str,
        method: str,
        commit: bool = True,
        delete_first: bool = True,
        table_field_names: set[str] | None = None,
        locations: dict[str, dict] | None = None,
    ) -> list[ExtractedField]:
        """Write extracted fields to the DB.

        By default deletes prior fields first.
        Set delete_first=False when appending a second batch (e.g. AI_DISCOVERY_ADDITIONAL).

        table_field_names — if provided, any field whose name is in this set is stored with
        field_type="table".

        locations — optional {field_name: {page: int, bbox: [x0,y0,x1,y1]}} from
        _enrich_with_locations().  When provided, extraction_page and extraction_bbox_json
        are stored on each ExtractedField for instant viewer highlighting.
        """
        from sqlalchemy import delete as sa_delete
        if delete_first:
            self.db.execute(
                sa_delete(ExtractedField).where(ExtractedField.document_id == doc.id)
            )
        _table_names = table_field_names or set()
        _locs = locations or {}
        fields = []
        for fname, (value, confidence) in raw_fields.items():
            loc = _locs.get(fname, {})
            ef = ExtractedField(
                document_id=doc.id,
                field_name=fname,
                field_value=value,
                field_type="table" if fname in _table_names else "scalar",
                confidence=round(min(confidence, 1.0), 4),
                extraction_model=model,
                extraction_method=method,
                extraction_tier=tier,
                human_corrected=False,
                extraction_page=loc.get("page"),
                extraction_bbox_json=(
                    json.dumps(loc["bbox"]) if loc.get("bbox") else None
                ),
            )
            self.db.add(ef)
            fields.append(ef)
        if commit:
            self.db.commit()
        return fields

    # ── Spatial location enrichment ───────────────────────────────────────────

    def _enrich_with_locations(
        self,
        raw_fields: dict[str, tuple[str, float]],
        pdf_bytes: bytes | None,
        pages_used: list[int] | None = None,
    ) -> dict[str, dict]:
        """
        For each extracted scalar field, use PyMuPDF to find its text location
        in the PDF.  Returns {field_name: {page: int, bbox: [x0,y0,x1,y1]}}.

        bbox values are fractions of page dimensions (0.0–1.0) so they are
        DPI-independent and can be scaled by the viewer at render time.

        pages_used — the page indices that were actually sent to the AI.  When
        provided, only those pages are searched (in document order).  This
        prevents false matches: if "PO-1234" appears in a page-0 header AND
        on page 5, and the AI only saw page 5, we must not record page 0.
        Falls back to searching all pages if pages_used is None or empty.

        Multi-line values (addresses) are searched by their first non-empty
        line only, because PyMuPDF cannot match text that spans line breaks.

        Table fields are skipped (their value is a JSON array, not searchable).
        Called immediately after extraction before _persist_fields.
        """
        if not pdf_bytes:
            return {}
        try:
            import fitz
        except ImportError:
            return {}

        results: dict[str, dict] = {}
        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages = len(fitz_doc)

            # Restrict to pages the AI actually saw; fall back to all pages.
            search_order = sorted(pages_used) if pages_used else list(range(n_pages))

            for fname, (value, _conf) in raw_fields.items():
                if not value or len(value.strip()) < 2:
                    continue
                # Skip table values (JSON arrays)
                v = value.strip()
                if v.startswith("[") or len(v) > 1000:
                    continue

                # Choose the best search term from the extracted value.
                #
                # Multi-line values (addresses): search the first non-empty line
                # because PyMuPDF cannot match text that spans line breaks.
                #
                # AI-joined values (e.g. "CODE / 30 Days from Shipment Date"):
                # when extraction accidentally grabbed surrounding text and joined
                # it with " / " or " — ", the LAST meaningful segment is usually
                # the actual field value, so prefer it over the first segment.
                search_v = v
                if "\n" in v:
                    first_line = next(
                        (ln.strip() for ln in v.splitlines() if ln.strip()), v
                    )
                    if len(first_line) >= 4:
                        search_v = first_line
                else:
                    # Try right-hand side of common AI join separators
                    for sep in (" / ", " — ", " | "):
                        if sep in v:
                            parts = v.split(sep, 1)
                            right = parts[1].strip() if len(parts) > 1 else ""
                            if len(right) >= 4:
                                search_v = right
                            break

                # ── Collect ALL matches across sampled pages ───────────────────
                # Body region: y between 12% and 88% of page height.
                # When the same text appears on multiple pages (e.g. a repeated
                # header), we prefer the body-region match over the header/footer.
                # If every match is in a header/footer, we take the first one.
                # If there's only one match, we use it directly.
                body_match:   tuple | None = None   # (pg_num, r, pw, ph)
                first_match:  tuple | None = None   # first match regardless of region

                for pg_num in search_order:
                    if pg_num >= n_pages:
                        continue
                    page = fitz_doc[pg_num]
                    pw = page.rect.width
                    ph = page.rect.height
                    if pw <= 0 or ph <= 0:
                        continue
                    rects = page.search_for(search_v, quads=False)
                    if not rects:
                        continue
                    r = rects[0]
                    if first_match is None:
                        first_match = (pg_num, r, pw, ph)
                    # Classify as body if y0 is between 12% and 88% of page height
                    y_frac = r.y0 / ph
                    if body_match is None and 0.12 <= y_frac <= 0.88:
                        body_match = (pg_num, r, pw, ph)
                    # Once we have both a first match and a body match we can stop
                    if first_match is not None and body_match is not None:
                        break

                chosen = body_match if body_match is not None else first_match
                if chosen is None:
                    continue
                pg_num, r, pw, ph = chosen
                results[fname] = {
                    "page": pg_num,
                    "bbox": [
                        round(r.x0 / pw, 4),
                        round(r.y0 / ph, 4),
                        round(r.x1 / pw, 4),
                        round(r.y1 / ph, 4),
                    ],
                }

            fitz_doc.close()
        except Exception as exc:
            logger.debug("_enrich_with_locations failed: %s", exc)

        return results

    def second_opinion(
        self,
        doc: Document,
        original_fields: list[ExtractedField],
    ) -> list[ExtractedField]:
        """
        AI_REVIEWING pass — apply heuristic confidence adjustment based on
        whether each field's value matches its expected format.

        In production: cross-run with a secondary model and compare values.
        """
        for ef in original_fields:
            if not ef.field_value:
                continue
            val = ef.field_value.strip()
            validator = FIELD_FORMAT_VALIDATORS.get(ef.field_name)
            if validator:
                pattern, _ = validator
                if re.match(pattern, val, re.IGNORECASE):
                    ef.confidence = round(min(ef.confidence + 0.04, 0.97), 4)
                else:
                    ef.confidence = round(max(ef.confidence - 0.06, 0.50), 4)
        self.db.commit()
        return original_fields


# ── Low-level extraction functions ────────────────────────────────────────────

# Labels that appear commonly in document boilerplate and are NOT useful fields.
_COLON_SCAN_SKIP = {
    'note', 'page', 'print', 'for', 'if', 'in', 'the', 'a', 'an', 'and', 'or',
    'etc', 'e', 'i', 're', 'to', 'of', 'at', 'by', 'this', 'that', 'we', 'our',
    'your', 'their', 'is', 'be', 'no', 'yes', 'sir', 'dear', 'date', 'from',
    'subject', 'regards', 'attention', 'ref', 'cc',
}

def _colon_scan(text: str) -> dict[str, tuple[str, float]]:
    """
    Fast structural pre-pass: find 'Label : Value' or 'Label :- Value' patterns
    that appear in trade documents (RFQ, PO, invoice item lines, etc.).

    Returns {snake_case_name: (value, confidence=0.65)}.
    Confidence is lower than AI because the label normalisation is mechanical.
    """
    results: dict[str, tuple[str, float]] = {}
    seen_names: set[str] = set()

    # Matches:  "Material Number    : 5989A1059"  or  "RFQ Date :- 02.06.2026"
    # Anchored to line start (the text has been compacted to single spaces, but
    # we also handle the raw multi-line form by treating \n as a line boundary).
    pattern = re.compile(
        r'(?:^|(?<=\n))\s*'
        r'([A-Za-z][A-Za-z0-9 \./\(\)\-]{1,40}?)'   # label (not starting with digit)
        r'\s*:-?\s*'                                   # colon separator (with optional dash)
        r'([^\n]{1,200})',                             # value (rest of line)
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        label = m.group(1).strip()
        value = m.group(2).strip()

        if not value or len(value) < 2:
            continue
        label_lower = label.lower()
        if label_lower in _COLON_SCAN_SKIP or len(label) < 3:
            continue
        # Reject values that look like sentence continuations (start with lowercase)
        if value and value[0].islower() and len(value) > 30:
            continue

        fname = re.sub(r'[^a-z0-9]+', '_', label_lower).strip('_')
        if not fname or fname in seen_names:
            continue
        seen_names.add(fname)
        results[fname] = (value[:200], 0.65)

    return results


@dataclass
class PageSampleMeta:
    """
    Returned by _smart_page_sample. Carries both the assembled text and the
    metadata needed to record what was and wasn't passed to the AI.
    """
    text: str
    pages_total: int
    pages_used: list[int]    # 0-indexed list of page numbers included in `text`
    pages_skipped: int       # pages_total - len(pages_used)
    used_learned_profile: bool = False   # True when confident_skip list was applied


def _extract_page_text_layout(page) -> str:
    """
    Word-level layout-aware text extraction from a single fitz page.

    Block-level column detection fails when a PDF author puts both left- and
    right-column labels inside a single wide text block (common in Tata Steel
    purchase orders and many other two-column forms).  Word-level extraction
    (`get_text("words")`) gives each word's exact bounding box, so we can split
    "Import is covered by Licence No:" (x≈28) from "FINAL DESTINATION :" (x≈300)
    even when they share the same PDF block.

    Algorithm:
      1. Extract all words with bboxes.
      2. Cluster into visual lines (words within 6 px of each other in y).
      3. For each line, check if there are clearly-left words AND clearly-right
         words separated by a gap of ≥10 % of page width → two-column line.
      4. If any two-column lines are found: output left column first, then right.
         Otherwise: fall back to fitz's standard reading-order text.

    Whitespace:
      • Horizontal whitespace (spaces/tabs) collapsed to single space.
      • Newlines are PRESERVED — column boundaries survive into the AI prompt.
    """
    try:
        page_width = page.rect.width
        if page_width <= 0:
            t = page.get_text("text")
            return re.sub(r"[^\S\n]+", " ", t).strip()

        mid    = page_width / 2
        # MARGIN: half-width of the "gap zone" around the page midpoint.
        # Words whose right edge (x1) is < mid-MARGIN are clearly left-column;
        # words whose left edge (x0) is > mid+MARGIN are clearly right-column.
        # A smaller margin catches borderline right-column content like stamp-box
        # text that sits just past centre (e.g. x0=324 on a 595pt-wide page).
        MARGIN  = max(8.0, page_width * 0.02)   # was max(15.0, 0.05) — too conservative
        GAP_MIN = page_width * 0.10             # min gap between left and right content

        # ── Step 1: word bboxes ────────────────────────────────────────────────
        # words: (x0, y0, x1, y1, word_text, block_no, line_no, word_no)
        raw_words = page.get_text("words")
        if not raw_words:
            return ""

        # ── Step 2: cluster into visual lines (3 px y-tolerance) ─────────────
        # A tight 3 px window keeps adjacent rows in two-column forms from
        # merging.  On this TML purchase order, "TML/MUM/UTTARAKHAND" (y=544)
        # and "TERMS OF PAYMENT" (y=548) share the same visual row but are in
        # different columns; the old 6 px window collapsed them into one line,
        # causing the AI to associate the stamp text with the payment label.
        raw_words = sorted(raw_words, key=lambda w: (w[1], w[0]))
        vlines: list[list] = []    # each entry: [running_y, [word_tuples]]
        for w in raw_words:
            y = w[1]
            placed = False
            for vl in reversed(vlines):   # check recent lines first (fast)
                if abs(vl[0] - y) <= 3:
                    vl[1].append(w)
                    vl[0] = sum(ww[1] for ww in vl[1]) / len(vl[1])
                    placed = True
                    break
            if not placed:
                vlines.append([y, [w]])
        vlines.sort(key=lambda vl: vl[0])

        # ── Step 3: classify each line ────────────────────────────────────────
        left_col:  list[tuple[float, str]] = []
        right_col: list[tuple[float, str]] = []
        is_two_col = False

        for avg_y, lwords in vlines:
            lwords.sort(key=lambda w: w[0])

            c_left  = [w for w in lwords if w[2] < mid - MARGIN]   # x1 left of mid
            c_right = [w for w in lwords if w[0] > mid + MARGIN]   # x0 right of mid

            if c_left and c_right:
                # Require a real gap — rules out continuous text that straddles mid
                rightmost_left  = max(w[2] for w in c_left)
                leftmost_right  = min(w[0] for w in c_right)
                gap = leftmost_right - rightmost_left

                if gap >= GAP_MIN:
                    is_two_col = True
                    # Assign ALL words on this line to left/right by centroid x.
                    # Using simple midpoint (no margin) so words in the gap zone
                    # (like "FINAL" in "FINAL DESTINATION :") are not dropped.
                    l_text = " ".join(
                        w[4] for w in lwords if (w[0] + w[2]) / 2 < mid
                    ).strip()
                    r_text = " ".join(
                        w[4] for w in lwords if (w[0] + w[2]) / 2 >= mid
                    ).strip()
                    if l_text:
                        left_col.append((avg_y, l_text))
                    if r_text:
                        right_col.append((avg_y, r_text))
                    continue   # handled as two-column

            # Single-column line — route by centroid x
            text = " ".join(w[4] for w in lwords).strip()
            if not text:
                continue
            centroid_x = sum(w[0] for w in lwords) / len(lwords)
            if centroid_x < mid:
                left_col.append((avg_y, text))
            else:
                right_col.append((avg_y, text))

        # ── Step 4: assemble output ───────────────────────────────────────────
        if is_two_col:
            parts: list[str] = []
            if left_col:
                parts.append("\n".join(t for _, t in left_col))
            if right_col:
                parts.append("\n".join(t for _, t in right_col))
            result = "\n\n".join(parts)
        else:
            result = page.get_text("text")

        result = re.sub(r"[^\S\n]+", " ", result)    # spaces/tabs → single space
        result = re.sub(r"\n{3,}", "\n\n", result)   # max 2 blank lines
        return result.strip()

    except Exception:
        try:
            t = page.get_text("text")
        except Exception:
            t = ""
        return re.sub(r"[^\S\n]+", " ", t).strip()


def _smart_page_sample(
    pdf_bytes: bytes,
    char_budget: int = 30_000,
    force_skip: list[int] | None = None,
) -> PageSampleMeta:
    """
    Build a representative text sample from a multi-page PDF that fits within char_budget.

    Returns a PageSampleMeta that contains:
      - text:          the assembled sample string with [Page N] markers
      - pages_total:   total pages in the PDF
      - pages_used:    0-indexed list of page numbers included
      - pages_skipped: count of pages excluded

    For short docs (total chars ≤ char_budget) the full text is returned unchanged
    and pages_used = list(range(n)).

    For long docs:
      1. Always includes the first HEADER_PAGES and last TAIL_PAGES pages.
      2. Any page listed in force_skip is excluded regardless.
      3. Scores remaining pages by data density and fills remaining budget with
         the highest-scoring ones (skipping zero-score boilerplate).

    force_skip — list of 0-indexed page numbers to exclude unconditionally.
    These come from PageProfileAgent's confident_skip list for this variant.
    """
    try:
        import fitz  # PyMuPDF — layout-aware, far superior to pypdf for multi-column forms

        fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_raw: list[str] = []
        for page in fitz_doc:
            pages_raw.append(_extract_page_text_layout(page))

        n = len(pages_raw)
        if n == 0:
            return PageSampleMeta(text="", pages_total=0, pages_used=[], pages_skipped=0)

        learned_skip: set[int] = set(force_skip or [])

        # ── Quick check: total chars small enough to use as-is ─────────────────
        # (Only applies when there are no forced skips.)
        if not learned_skip:
            flat = "\n\n".join(pages_raw)
            if len(flat) <= char_budget:
                return PageSampleMeta(
                    text=flat,
                    pages_total=n,
                    pages_used=list(range(n)),
                    pages_skipped=0,
                )

        # ── Score every page by "data density" ─────────────────────────────────
        _DATA_RE = re.compile(
            r"(\b\d{4,}\b"                         # 4+ digit numbers (amounts, codes)
            r"|\b\d{1,3}[,\.]\d{3}\b"              # formatted numbers like 1,000 or 1.000
            r"|\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}"   # dates DD/MM/YYYY
            r"|material\s*(?:no|number)"
            r"|wbs\s*no"
            r"|item\s*(?:no|price|charges)"
            r"|delivery\s*(?:date|requirements?)"
            r"|(?:usd|eur|gbp|inr|chf)\b"
            r"|\btotal\s*(?:qty|quantity|value|amount|price)\b"
            r"|\bschedule\s*\d"
            r"|\bper\s+\d+\s+(?:lot|set|nos?|pcs?|kg)\b"
            r"|\bquantity\b|\bpayment\b|\binvoice\b|\bpo\s*no\b"
            r"|\b@\s*[\d,]+\.\d{2}\b"              # price @8,000,000.00
            r")",
            re.IGNORECASE,
        )
        scores: list[int] = [len(_DATA_RE.findall(p)) for p in pages_raw]

        # ── Always-include sets (minus learned_skip) ──────────────────────────
        HEADER_PAGES = min(3, n)
        TAIL_PAGES   = min(10, n)
        always_idx: set[int] = (
            set(range(HEADER_PAGES)) | set(range(n - TAIL_PAGES, n))
        ) - learned_skip

        # ── Fill remaining budget with highest-scoring "middle" pages ───────────
        middle_scored = sorted(
            [
                (scores[i], i)
                for i in range(n)
                if i not in always_idx and i not in learned_skip
            ],
            reverse=True,
        )

        selected_idx: set[int] = set(always_idx)
        used_chars = sum(len(pages_raw[i]) for i in always_idx)

        for score, idx in middle_scored:
            if score == 0:
                break   # stop at zero-score (pure boilerplate)
            page_chars = len(pages_raw[idx])
            if used_chars + page_chars > char_budget:
                continue
            selected_idx.add(idx)
            used_chars += page_chars

        # ── Assemble in page order with markers ────────────────────────────────
        parts: list[str] = []
        prev_included = -1
        for i in range(n):
            if i in selected_idx:
                if i > prev_included + 1:
                    skipped = i - (prev_included + 1)
                    parts.append(f"[... {skipped} page(s) skipped ...]")
                if pages_raw[i]:
                    parts.append(f"[Page {i + 1}]\n{pages_raw[i]}")
                prev_included = i

        if prev_included < n - 1:
            tail_skipped = n - 1 - prev_included
            parts.append(f"[... {tail_skipped} page(s) skipped ...]")

        result_text = "\n\n".join(parts)
        pages_used_sorted = sorted(selected_idx)
        logger.info(
            "smart_page_sample: %d/%d pages used, %d learned-skip, %d chars (budget %d)",
            len(selected_idx), n, len(learned_skip), len(result_text), char_budget,
        )
        return PageSampleMeta(
            text=result_text,
            pages_total=n,
            pages_used=pages_used_sorted,
            pages_skipped=n - len(selected_idx),
            used_learned_profile=bool(learned_skip),
        )

    except Exception as exc:
        logger.debug("smart_page_sample failed (%s) — falling back to flat text", exc)
        return PageSampleMeta(text="", pages_total=0, pages_used=[], pages_skipped=0)


def _extract_via_ai_free_form(
    sampled_text: str,
    doc_class_name: str,
) -> tuple[
    dict[str, tuple[str, float]],   # core scalar fields
    dict[str, tuple[str, float]],   # additional scalar fields
    dict[str, tuple[str, float]],   # tables: {table_name: (json_array_str, confidence)}
]:
    """
    Zero-shot ERP-focused discovery extraction.

    sampled_text — pre-sampled text from _smart_page_sample (may contain [Page N] markers
                   and [... X page(s) skipped ...] separators). Callers are responsible
                   for calling _smart_page_sample before passing text here.

    Asks Claude to split output into three sections:
      core       — scalar fields needed for ERP import (identifiers, dates, parties, amounts)
      additional — supplementary scalar fields (contact details, compliance refs, etc.)
      tables     — any repeating structured data; AI names each table and returns it as
                   an array of row objects. Examples: line_items, freight_charges, packages.
                   Multiple tables supported — one per distinct repeating structure.

    Returns three dicts of {snake_case_name: (value_str, confidence)}.
    Table values are JSON-serialised arrays; scalars are plain strings.
    """
    if not settings.anthropic_api_key:
        return {}, {}, {}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        ai_text = sampled_text or ""

        # ── Call 1: scalar fields only (core + additional) ─────────────────────
        # Keeping scalars separate means the AI has its full token budget just
        # for field values — no competition from large line-item arrays.
        scalar_prompt = (
            f"You are extracting structured data from a **{doc_class_name}** document for ERP data entry.\n\n"
            "Return ONLY this JSON (no markdown, no explanation):\n"
            "{\n"
            '  "core":       { "field_name": "value", ... },\n'
            '  "additional": { "field_name": "value", ... }\n'
            "}\n\n"
            "CORE — the scalar fields an ERP operator must have to process this document (aim 10–20).\n"
            f"  Use your knowledge of what a {doc_class_name} contains.\n"
            "  Capture: key identifiers, dates, counterparties, totals — anything needed to post or match.\n\n"
            "ADDITIONAL — scalar fields useful but not required to post (up to 15):\n"
            "  Secondary references, compliance codes, contact details, supplementary charges.\n\n"
            "DO NOT include any tables or repeating row data — that will be extracted separately.\n\n"
            "Rules:\n"
            "  • Field names: copy the EXACT label text as printed on the document — do NOT\n"
            "    normalise, lowercase, or add underscores. e.g. use \"Payment Terms\" not \"payment_terms\".\n"
            "  • Include scalar fields even if value is blank — use empty string \"\"\n"
            "  • Raw JSON only — no markdown fences\n\n"
            "VERBATIM RULE — strictly enforced:\n"
            "  • Values MUST be copied verbatim from the document. Never paraphrase or explain.\n"
            "  • You MAY extract a single relevant token (e.g. \"30\" from \"Net 30 days\") but it\n"
            "    must be a literal substring of the source text — never invented or rephrased.\n"
            "  • NEVER paraphrase, summarise, or compose a value yourself.\n"
            "  • BAD:  payment_terms: \"Invoice payable within 30 days of receipt\"  (paraphrase)\n"
            "  • GOOD: payment_terms: \"Net 30\"  or  payment_terms: \"30\"\n"
            "  • BAD:  incoterms: \"Freight paid by seller to named destination\"    (paraphrase)\n"
            "  • GOOD: incoterms: \"FCA\"  or  incoterms: \"CIF Mumbai\"\n\n"
            "MULTI-LINE VALUES — important:\n"
            "  • If a field value naturally spans multiple lines in the document\n"
            "    (address blocks, descriptions, remarks, header details), capture ALL lines\n"
            "    joined with \\n. Do NOT truncate to the first line.\n"
            "  • Address fields (To, From, Ship To, Deliver To, Consignee, Buyer Address,\n"
            "    Seller Address, Vendor, Supplier, Customer, etc.): capture the FULL block —\n"
            "    company/person name AND every address line below it.\n"
            "  • BAD:  \"To\": \"Refteck Solutions Limited\"  (first line only — truncated)\n"
            "  • GOOD: \"To\": \"Refteck Solutions Limited\\n43\\nCoulsdon, CR51NL\\nSurrey\\nUnited Kingdom\"\n\n"
            "BLANK FIELD RULE — critical for multi-column documents:\n"
            "  • Many forms are two-column. The PDF text stream interleaves both columns,\n"
            "    so text that appears AFTER a label may belong to the OTHER column entirely.\n"
            "  • A field is BLANK (\"\") when the text following its label is:\n"
            "      (a) clearly a legal clause, policy text, or licence condition\n"
            "      (b) obviously a value for a completely different field type\n"
            "      (c) another field's label\n"
            "  • SEMANTIC CHECK: the value must make sense for the field.\n"
            "      A 'Final Destination' must be a place name, not policy text.\n"
            "      A 'Payment Terms' must be a payment condition, not an address.\n"
            "  • BAD:  \"Final Destination\": \"FREELY IMPORTABLE AS PER CHAPTER 2 PARA 2.01\"\n"
            "          — this is an import licence clause from an adjacent column; the field is blank\n"
            "  • GOOD: \"Final Destination\": \"\"\n"
            "  • BAD:  \"Delivery Date\": \"Tata Steel UK Limited, 18 Grosvenor Place\"\n"
            "          — that is an address, not a date; the field is blank\n"
            "  • GOOD: \"Delivery Date\": \"\"\n\n"
            "Document text:\n---\n"
            + ai_text
            + "\n---"
        )
        scalar_msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": scalar_prompt}],
        )
        scalar_raw = scalar_msg.content[0].text.strip()
        scalar_raw = re.sub(r"^```(?:json)?\s*", "", scalar_raw)
        scalar_raw = re.sub(r"\s*```$", "", scalar_raw)
        try:
            scalar_data = json.loads(scalar_raw)
        except Exception:
            scalar_data = {}

        # ── Call 2: tables only (line items, packages, etc.) ───────────────────
        # Dedicated call so the full 8 192-token output budget goes to row data.
        table_prompt = (
            f"You are extracting structured data from a **{doc_class_name}** document for ERP data entry.\n\n"
            "Return ONLY this JSON (no markdown, no explanation):\n"
            "{\n"
            '  "tables": { "table_name": [ {"col": "val", ...}, ... ], ... }\n'
            "}\n\n"
            "TABLES — every repeating structured data set in the document:\n"
            "  Look for: line items, order positions, package rows, charge breakdowns, test results,\n"
            "  payment schedules, dangerous goods entries — anything that forms a natural table.\n"
            "  Give each table a short descriptive snake_case name (e.g. line_items, freight_charges,\n"
            "  packages, hazmat_entries). Each table is an array of objects — one object per row.\n"
            "  Column names: use the EXACT header text as printed — do NOT normalise or add underscores.\n"
            "  IMPORTANT: include EVERY row from EVERY page — do not stop early or summarise.\n"
            "  If no tables are present, return {\"tables\": {}}.\n\n"
            "DO NOT include scalar fields — only tables.\n\n"
            "Rules:\n"
            "  • Raw JSON only — no markdown fences\n"
            "  • No page numbers or boilerplate rows\n"
            "  • Cell values must be verbatim from the document — never paraphrase or summarise.\n\n"
            "Document text:\n---\n"
            + ai_text
            + "\n---"
        )
        table_msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": table_prompt}],
        )
        if table_msg.stop_reason == "max_tokens":
            logger.warning(
                "free_form table extraction hit max_tokens (8192) for %s — "
                "document has more line items than can fit in one response. "
                "Confirm the schema so future runs use targeted per-table extraction.",
                doc_class_name,
            )
        table_raw = table_msg.content[0].text.strip()
        table_raw = re.sub(r"^```(?:json)?\s*", "", table_raw)
        table_raw = re.sub(r"\s*```$", "", table_raw)
        try:
            table_data = json.loads(table_raw)
        except Exception:
            table_data = {}

        data = {
            "core":       scalar_data.get("core", {}),
            "additional": scalar_data.get("additional", {}),
            "tables":     table_data.get("tables", {}),
        }

        def _parse_scalar_section(section: dict) -> dict[str, tuple[str, float]]:
            out: dict[str, tuple[str, float]] = {}
            for fname, value in (section or {}).items():
                # Keep field names VERBATIM — strip whitespace only
                fname_clean = str(fname).strip()
                if not fname_clean:
                    continue
                str_val = str(value).strip() if value is not None else ""
                if str_val in ("null", "None"):
                    str_val = ""
                out[fname_clean] = (str_val[:500], 0.80)
            return out

        def _parse_tables_section(section: dict) -> dict[str, tuple[str, float]]:
            """
            Convert the AI's tables dict into {table_name: (json_array_str, confidence)}.
            Table names stay snake_case (internal key); column names are kept verbatim.
            """
            out: dict[str, tuple[str, float]] = {}
            for tname, rows in (section or {}).items():
                # Table name: snake_case for internal key
                tname_clean = re.sub(r"[^a-z0-9_]+", "_", str(tname).lower()).strip("_")
                if not tname_clean:
                    continue
                if not isinstance(rows, list) or not rows:
                    continue
                # Column names kept VERBATIM; values to strings
                normalised_rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    norm_row = {}
                    for k, v in row.items():
                        col = str(k).strip()  # verbatim
                        if col:
                            norm_row[col] = str(v).strip() if v is not None else ""
                    if norm_row:
                        normalised_rows.append(norm_row)
                if normalised_rows:
                    out[tname_clean] = (json.dumps(normalised_rows), 0.80)
            return out

        core       = _parse_scalar_section(data.get("core", {}))
        additional = _parse_scalar_section(data.get("additional", {}))
        tables     = _parse_tables_section(data.get("tables", {}))

        logger.info(
            "AI free-form discovery: %d core + %d additional scalars + %d tables (%s)",
            len(core), len(additional), len(tables), ", ".join(tables.keys()) or "none",
        )
        return core, additional, tables

    except Exception as exc:
        logger.warning("AI free-form extraction failed: %s", exc)
        return {}, {}, {}


def _extract_text_layer(pdf_bytes: bytes | None) -> str:
    """Extract text from the PDF text layer via pypdf."""
    if not pdf_bytes:
        return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = [t for page in reader.pages if (t := page.extract_text())]
        text = " ".join(parts).strip()
        return re.sub(r"\s+", " ", text) if text else ""
    except Exception as exc:
        logger.debug("text layer extraction failed: %s", exc)
        return ""


# ── Vision preprocessing helper ───────────────────────────────────────────────

def _preprocess_for_vision(pil_img):
    """
    Lighter preprocessing before sending to Claude Vision:
    deskew + CLAHE on the L channel (preserves colour — Claude reads
    layout context from the full image, not just isolated text pixels).
    Falls back to the original image if OpenCV is unavailable.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image as _PILImg

        img_np = np.array(pil_img.convert("RGB"))

        # Deskew on grayscale, then apply rotation to the colour image ───
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, binary_inv = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            coords = np.column_stack(np.where(binary_inv > 0))
            if len(coords) > 100:
                angle = cv2.minAreaRect(coords)[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                if abs(angle) > 0.3:
                    h, w = img_np.shape[:2]
                    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                    img_np = cv2.warpAffine(
                        img_np, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
        except Exception:
            pass

        # CLAHE on the L channel of LAB (preserves hue/saturation) ───────
        lab          = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_ch, a, b   = cv2.split(lab)
        clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab_enhanced = cv2.merge([clahe.apply(l_ch), a, b])
        img_np       = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        return _PILImg.fromarray(img_np)

    except ImportError:
        return pil_img   # OpenCV absent — return original unchanged


def _apply_patterns(
    text: str,
    target_fields: list[str],
    dampen: float,
) -> dict[str, tuple[str, float]]:
    """
    Run EXTRACTION_PATTERNS against `text` for the target fields.
    Returns {field_name: (value, confidence)}.
    """
    text_lower = text.lower()
    results: dict[str, tuple[str, float]] = {}

    for fname in target_fields:
        patterns = EXTRACTION_PATTERNS.get(fname, [])
        for pattern, group_idx, base_conf in patterns:
            try:
                m = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if m:
                    # Re-match original-case text to preserve casing in stored value
                    m_orig = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    raw_val = (m_orig.group(group_idx) if m_orig else m.group(group_idx)).strip()
                    # Strip trailing punctuation from numeric/amount fields to avoid
                    # values like "5,06,053.58." (sentence-end period breaks validation)
                    if fname in _NUMERIC_FIELD_NAMES:
                        raw_val = raw_val.rstrip(".,;:")
                        # Normalise thousands-separator spaces (Nordic/French format).
                        # "166 825,00" → "166825,00" so format validator ^[\d,\.]+$ passes.
                        raw_val = re.sub(r'(?<=\d)\s+(?=\d)', '', raw_val)
                    if raw_val:
                        results[fname] = (raw_val, round(min(base_conf * dampen, 1.0), 4))
                        break
            except Exception as exc:
                logger.debug("pattern error for field %s: %s", fname, exc)

    return results


def _extract_vision_api(
    pdf_bytes: bytes,
    doc_class_id: str,
    target_fields: list[str],
) -> dict[str, tuple[str, float]]:
    """
    Tier 3: Send the first 2 PDF pages as JPEG images to Claude claude-sonnet-4-6
    and parse the JSON response for the target field list.
    """
    try:
        import anthropic
        import base64

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # ── Render PDF pages to JPEG with preprocessing ──────────────────────────
        # Up to 4 pages at 250 DPI; apply deskew + CLAHE so faded / rotated scans
        # are readable by Claude (same _preprocess_for_vision used for OCR path).
        jpeg_pages: list[bytes] = []
        MAX_PAGES = 4
        try:
            import fitz  # PyMuPDF — no poppler dependency
            from PIL import Image as _VPil
            doc_fitz = fitz.open(stream=pdf_bytes, filetype="pdf")
            mat      = fitz.Matrix(250 / 72, 250 / 72)   # 250 DPI (up from 200)
            for page_num in range(min(MAX_PAGES, len(doc_fitz))):
                pix      = doc_fitz[page_num].get_pixmap(matrix=mat)
                pil_img  = _VPil.open(io.BytesIO(pix.tobytes("png")))
                enhanced = _preprocess_for_vision(pil_img)
                buf      = io.BytesIO()
                enhanced.save(buf, format="JPEG", quality=88)
                jpeg_pages.append(buf.getvalue())
        except ImportError:
            from pdf2image import convert_from_bytes
            from PIL import Image as _VPil
            pil_imgs = convert_from_bytes(pdf_bytes, dpi=250,
                                          first_page=1, last_page=MAX_PAGES)
            for img in pil_imgs[:MAX_PAGES]:
                enhanced = _preprocess_for_vision(img)
                buf      = io.BytesIO()
                enhanced.save(buf, format="JPEG", quality=88)
                jpeg_pages.append(buf.getvalue())

        if not jpeg_pages:
            return {}

        image_blocks = []
        for jpeg_bytes in jpeg_pages:
            b64 = base64.standard_b64encode(jpeg_bytes).decode()
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

        # ── Prompt: free-form discovery (empty list) or targeted extraction ──────
        if not target_fields:
            # ZERO_SHOT discovery mode: find EVERYTHING, return structured output
            prompt = (
                "You are reading a document image for a logistics and procurement company.\n"
                "Some pages may contain two documents overlaid on the same scan (e.g. a\n"
                "courier waybill printed on top of a commercial invoice). Extract data from\n"
                "BOTH documents — label fields from each document separately if they conflict.\n"
                "Extract ALL data visible in the document and return it in this JSON structure:\n\n"
                "{\n"
                '  "fields": { "field_name": "value", ... },\n'
                '  "tables": { "table_name": [ {"col": "val", ...}, ... ], ... }\n'
                "}\n\n"
                "FIELDS — every individual labelled value: reference numbers, dates, addresses,\n"
                "  amounts, contacts, terms, party names.\n"
                "  Field names: copy the EXACT label text as printed on the document.\n"
                "  e.g. use \"Payment Terms\" not \"payment_terms\", \"P.O. No.\" not \"po_number\".\n\n"
                "TABLES — any repeating structured rows (line items, packages, charges, test results,\n"
                "  dangerous goods entries, etc.). Give each table a clear snake_case name and return\n"
                "  it as an array of row objects.\n"
                "  Column names: use the EXACT header text as printed — do NOT normalise.\n"
                "  If no tables exist, return an empty object {}.\n\n"
                "Rules:\n"
                "  - Do NOT flatten table rows into field_1_x, field_2_x — use the tables section\n"
                "  - Skip blank fields and pure boilerplate paragraphs\n"
                "  - MULTI-LINE VALUES: if a field value spans multiple lines (address blocks,\n"
                "    descriptions, remarks), capture ALL lines joined with \\n — do NOT truncate\n"
                "  - Return ONLY the JSON object, no markdown fences"
            )
        else:
            # Targeted extraction: specific fields with descriptions
            field_lines: list[str] = []
            for f in target_fields:
                desc = FIELD_DESCRIPTIONS.get(f)
                if desc:
                    field_lines.append(f'  "{f}": "{desc}"')
                else:
                    field_lines.append(f'  "{f}": "{f.replace("_", " ").title()}"')
            fields_schema = "{\n" + "\n".join(field_lines) + "\n}"
            prompt = (
                "You are a precise document field extractor for a logistics and procurement company.\n"
                "Extract ONLY the following fields from the attached document image(s).\n"
                "Return a single valid JSON object. Use null for any field not found or uncertain.\n"
                "The field descriptions are authoritative — follow them precisely.\n\n"
                "VERBATIM RULE — strictly enforced:\n"
                "  • Copy values exactly as they appear in the document. Never paraphrase or explain.\n"
                "  • You MAY extract a single relevant token (e.g. \"30\" from \"Net 30 days\") but it\n"
                "    must be a literal substring of the visible text — never invented or rephrased.\n"
                "  • NEVER paraphrase, summarise, or compose a value yourself.\n"
                "  • BAD:  payment_terms: \"Invoice payable within 30 days\"  →  GOOD: \"Net 30\" or \"30\"\n"
                "  • BAD:  incoterms: \"Freight paid to destination\"         →  GOOD: \"FCA\" or \"CIF\"\n\n"
                "MULTI-LINE VALUES:\n"
                "  • If a field spans multiple lines (address, description, remarks), capture ALL lines\n"
                "    joined with \\n. Do NOT stop at the first line.\n"
                "  • BAD:  \"To\": \"Refteck Solutions Limited\"  →  GOOD: \"To\": \"Refteck Solutions Limited\\n43\\nCoulsdon, CR51NL\\nSurrey\\nUnited Kingdom\"\n\n"
                "BLANK FIELD / MULTI-COLUMN RULE:\n"
                "  • The value must make SEMANTIC SENSE for the field type.\n"
                "  • If a field is blank on the document, return null — never fill it with\n"
                "    text from an adjacent column or a nearby unrelated section.\n"
                "  • BAD:  \"Final Destination\": \"FREELY IMPORTABLE AS PER CHAPTER 2 PARA 2.01\"\n"
                "          — this is an import licence clause, not a destination; return null\n"
                "  • GOOD: \"Final Destination\": null  (field is blank on this document)\n\n"
                f"Fields:\n{fields_schema}\n\n"
                "Return ONLY the JSON object, no explanation or markdown fences."
            )

        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [*image_blocks, {"type": "text", "text": prompt}],
            }],
        )

        raw_text = response.content[0].text.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            logger.warning("Tier 3 response contained no JSON: %r", raw_text[:200])
            return {}

        extracted: dict = json.loads(json_match.group(0))
        results: dict[str, tuple[str, float]] = {}
        null_vals = {"null", "none", "n/a", "n.a.", "-", "–", "tbd"}

        if not target_fields:
            # ZERO_SHOT vision discovery — response uses {"fields": {...}, "tables": {...}}
            # Flatten fields section — keep names VERBATIM (strip only)
            for fname, val in (extracted.get("fields") or extracted).items():
                if not isinstance(val, (str, int, float)):
                    continue
                val_str = str(val).strip()
                fname_clean = str(fname).strip()
                if val_str and val_str.lower() not in null_vals and fname_clean:
                    results[fname_clean] = (val_str, 0.82)
            # Flatten tables section — table names stay snake_case; column names verbatim
            for tname, rows in (extracted.get("tables") or {}).items():
                if not isinstance(rows, list) or not rows:
                    continue
                # Table name: snake_case for internal key (it's not shown to the user directly)
                tname_clean = re.sub(r"[^a-z0-9_]+", "_", str(tname).lower()).strip("_")
                if not tname_clean:
                    continue
                norm_rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    # Column names verbatim — just strip whitespace
                    norm_row = {
                        str(k).strip(): str(v).strip()
                        for k, v in row.items()
                        if str(k).strip()
                    }
                    if norm_row:
                        norm_rows.append(norm_row)
                if norm_rows:
                    results[tname_clean] = (json.dumps(norm_rows), 0.80)
        else:
            # Targeted extraction against known field list (LEARNING/LEARNED)
            for fname in target_fields:
                val = extracted.get(fname)
                if val and str(val).strip() and str(val).strip().lower() not in null_vals:
                    conf = 0.88 if fname in FIELD_FORMAT_VALIDATORS else 0.82
                    results[fname] = (str(val).strip(), conf)

        return results

    except Exception as exc:
        logger.warning("Tier 3 AI vision extraction failed: %s", exc)
        return {}


def _apply_generated_patterns(
    text: str,
    patterns: dict,
    target_fields: list[str],
) -> dict[str, tuple[str, float]]:
    """
    Apply AI-generated regex patterns (from SchemaLearnerAgent.generate_fast_patterns)
    to extracted text. Returns {field_name: (value, confidence)}.
    """
    results: dict[str, tuple[str, float]] = {}
    if not text:
        return results
    for fname in target_fields:
        entry = patterns.get(fname)
        if not entry or not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        group   = entry.get("group", 1)
        conf    = float(entry.get("confidence", 0.88))
        if not pattern:
            continue
        try:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = m.group(group).strip()
                val = re.sub(r"\s+", " ", val)
                if val:
                    results[fname] = (val, conf)
        except Exception as exc:
            logger.debug("Generated pattern error for %s: %s", fname, exc)
    return results


def _sufficient_quality(
    raw: dict[str, tuple[str, float]],
    target_fields: list[str],
    threshold: float = 0.85,
) -> bool:
    """
    Return True if the extraction result is good enough to skip AI fallback.
    Requires ≥40% of target fields extracted at ≥threshold confidence.
    """
    if not raw or not target_fields:
        return False
    good = sum(1 for f in target_fields if f in raw and raw[f][1] >= threshold)
    return good / len(target_fields) >= 0.40


def _extract_via_ai_text(
    text: str,
    target_fields: list[str],
    doc_class_name: str,
    schema_hints: dict | None = None,
) -> dict[str, tuple[str, float]]:
    """
    Tier 1.5: Send extracted text to Claude and get structured field JSON back.

    This catches cases where regex grabs the wrong occurrence of a pattern
    (e.g., picks up the buyer's name when looking for supplier_name, or
    captures a PO number with embedded line-breaks from raw PDF text).

    Called after Tier 1 regex so AI can SUPPLEMENT and CORRECT low-confidence
    regex extractions without re-running the full OCR pipeline.

    Returns {field_name: (value, confidence)} — same shape as _apply_patterns().
    """
    if not settings.anthropic_api_key or not text.strip():
        return {}

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # ── Split fields into scalar vs table ────────────────────────────────
        # Table fields (field_type="table") need JSON array output, not strings.
        # Detect via hints for LEARNED/OPTIMISED stages; fall back to name heuristic.
        hints = schema_hints or {}
        scalar_fields: list[str] = []
        table_fields: list[str] = []
        for f in target_fields:
            hint_data = hints.get(f, {})
            if isinstance(hint_data, dict) and hint_data.get("field_type") == "table":
                table_fields.append(f)
            else:
                scalar_fields.append(f)

        # ── Build scalar field schema ─────────────────────────────────────────
        scalar_lines: list[str] = []
        for f in scalar_fields:
            desc = FIELD_DESCRIPTIONS.get(f, f.replace("_", " ").title())
            hint_data = hints.get(f, {})
            hint_parts: list[str] = []
            if hint_data.get("description"):
                desc = hint_data["description"]
            if hint_data.get("format_hint"):
                hint_parts.append(f"format: {hint_data['format_hint']}")
            if hint_data.get("example_values"):
                exs = ", ".join(str(v) for v in hint_data["example_values"][:3])
                hint_parts.append(f"e.g. {exs}")
            if hint_data.get("occurrence_note"):
                hint_parts.append(hint_data["occurrence_note"])
            if hint_parts:
                desc = desc + " [" + "; ".join(hint_parts) + "]"
            scalar_lines.append(f'  "{f}": "{desc}"')

        fields_schema = "{\n" + ",\n".join(scalar_lines) + "\n}" if scalar_lines else "{}"

        # ── Build table field schema ──────────────────────────────────────────
        table_lines: list[str] = []
        for f in table_fields:
            hint_data = hints.get(f, {})
            desc = hint_data.get("description") or FIELD_DESCRIPTIONS.get(f, f.replace("_", " ").title())
            table_lines.append(f'  "{f}": "{desc} — return as a JSON ARRAY of row objects"')
        tables_schema = "{\n" + ",\n".join(table_lines) + "\n}" if table_lines else ""

        # ── Assemble prompt ───────────────────────────────────────────────────
        # Keep text to 80 000 chars — callers use _smart_page_sample so this is a safety cap only
        text_excerpt = text[:80_000]
        if len(text) > 80_000:
            text_excerpt += "\n[... text truncated ...]"

        # Build hint block if we have confirmed schema examples
        hint_block = ""
        if hints and any(h.get("required") for h in hints.values() if isinstance(h, dict)):
            hint_block = (
                "\nThis schema has been CONFIRMED by an operator — all 'required' fields "
                "MUST be present. If a required field is genuinely absent, return null "
                "but flag it. Do not fill required fields with guesses.\n"
            )

        tables_section = ""
        if tables_schema:
            tables_section = (
                "\n\nTABLE FIELDS (return as JSON arrays — one object per row, "
                "all rows from ALL pages):\n"
                f"{tables_schema}"
            )

        prompt = (
            f"You are extracting structured data from a '{doc_class_name}' document "
            "for a logistics and procurement company.\n\n"
            f"Document text (raw PDF extract):\n---\n{text_excerpt}\n---\n\n"
            "Extract ONLY the fields listed below. Return a single valid JSON object.\n"
            "Use null for any scalar field absent or uncertain.\n"
            f"{hint_block}"
            "SCALAR FIELD rules:\n"
            "- For amounts: include currency symbol only if it appears with the number\n"
            "- For dates: use the format as written in the document\n"
            "- Do NOT guess — return null if uncertain\n"
            "- VERBATIM RULE: copy values exactly as they appear in the document text above.\n"
            "  Never paraphrase, explain, or compose a value yourself.\n"
            "  You MAY shorten to a single relevant token (e.g. '30' from 'Net 30 days') but\n"
            "  that token must literally appear in the source text — never invented.\n"
            "  BAD:  payment_terms: 'Invoice payable within 30 days of receipt'\n"
            "  GOOD: payment_terms: 'Net 30'  or  payment_terms: '30'\n"
            "- MULTI-LINE RULE: if a field value spans multiple lines (address, description,\n"
            "  remarks), capture ALL lines joined with \\n. Do NOT truncate to line 1.\n"
            "  Address fields (To, From, Ship To, Deliver To, Consignee, Buyer Address, etc.):\n"
            "  capture the full block — name AND all address lines below it.\n"
            "  BAD:  'To': 'Refteck Solutions Limited'\n"
            "  GOOD: 'To': 'Refteck Solutions Limited\\n43\\nCoulsdon, CR51NL\\nSurrey\\nUnited Kingdom'\n\n"
            "TABLE FIELD rules:\n"
            "- Return a JSON array of row objects (one object per line item/row)\n"
            "- Include EVERY row from EVERY page — do not truncate or summarise\n"
            "- Column names must be consistent snake_case across all rows\n"
            "- If a table is absent, return an empty array []\n\n"
            f"Scalar fields:\n{fields_schema}"
            f"{tables_section}\n\n"
            "Return ONLY the JSON object, no explanation or markdown fences."
        )

        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"\s*```$", "", raw_text, flags=re.MULTILINE)

        # Warn if response was cut by token limit — JSON will be incomplete
        if response.stop_reason == "max_tokens":
            logger.warning(
                "AI text extraction hit max_tokens (%d) for %s — response may be incomplete",
                8192, doc_class_name,
            )

        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            logger.warning("AI text extraction: no JSON in response: %r", raw_text[:200])
            return {}

        extracted: dict = json.loads(json_match.group(0))
        results: dict[str, tuple[str, float]] = {}
        null_vals = {"null", "none", "n/a", "n.a.", "-", "–", "tbd", "not found",
                     "unknown", "not specified", "not available", "not present"}

        for fname in target_fields:
            val = extracted.get(fname)
            if val is None:
                continue

            # Table fields: serialize as JSON, not Python repr
            if fname in table_fields:
                if isinstance(val, (list, dict)):
                    if isinstance(val, list) and len(val) == 0:
                        continue  # empty table — skip
                    val_str = json.dumps(val)
                else:
                    val_str = str(val).strip()
                if not val_str or val_str in ("[]", "{}"):
                    continue
                results[fname] = (val_str, 0.80)
                continue

            # Scalar fields: clean string normalisation
            val_str = str(val).strip()
            val_str = re.sub(r"[\r\n\t]+", " ", val_str)
            val_str = re.sub(r"\s{2,}", " ", val_str).strip()
            if not val_str or val_str.lower() in null_vals:
                continue
            # AI-text confidence: slightly below Tier 3 vision since text can be garbled
            conf = 0.84 if fname in FIELD_FORMAT_VALIDATORS else 0.80
            results[fname] = (val_str, conf)

        logger.debug("AI text extraction returned %d/%d fields (%d tables) for %s",
                     len(results), len(target_fields), len(table_fields), doc_class_name)
        return results

    except Exception as exc:
        logger.warning("AI text extraction failed: %s", exc)
        return {}
