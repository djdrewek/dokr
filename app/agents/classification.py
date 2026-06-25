"""
Classification Agent — assigns a Document Class to an incoming document.

Keyword scoring: count how many of a class's keywords appear in the
normalised PDF text. Highest scorer above MIN_MATCH_THRESHOLD wins.

Keywords derived from real PDF text extracted from Tata Limited's sample
document corpus (June 2026). Each list is ordered with the most
discriminating terms first.

Text extraction strategy (two-pass):
  1. pypdf  — fast, zero-cost, handles text-layer PDFs.
  2. OCR fallback — if pypdf returns no text the PDF is likely a scanned
     image. pdf2image converts each page to a PIL image; pytesseract (backed
     by Tesseract 4.x LSTM) extracts text. OCR is slower (~1-2 s/page) but
     fully transparent to the rest of the pipeline.
"""

from __future__ import annotations

import io
import logging
import tempfile

import pypdf
from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.pipeline.states import PipelineState

logger = logging.getLogger(__name__)

# Minimum keyword matches required to classify
MIN_MATCH_THRESHOLD = 2


# ── Classification rules ──────────────────────────────────────────────────────
# Keywords are lowercased; scored by substring match in normalised PDF text.

CLASSIFICATION_RULES: dict[str, list[str]] = {

    # dc_001: TML Import Contract PO
    "dc_001": [
        "tata motors ltd",
        "tata motors limited",
        "please supply and deliver",
        "purchase order no",
        "tml/",
        "imports contract",
        "buying agent",
        "orders@tata.co.uk",
        "tata limited",
        "grosvenor place",
    ],

    # dc_002: TMPVL Purchase Order
    "dc_002": [
        "tata motors passenger vehicles",
        "tmpvl/",
        "please supply and deliver",
        "purchase order no",
        "tata limited",
        "orders@tata.co.uk",
        "grosvenor place",
    ],

    # dc_003: Tata Steel Purchase Order
    "dc_003": [
        "tata steel limited",
        "please supply and deliver",
        "purchase order no",
        "tsl/",
        "tata steel",
        "orders@tata.co.uk",
        "grosvenor place",
        "tata limited",
    ],

    # dc_004: Airway Bill / House AWB
    "dc_004": [
        "airway bill",
        "house air waybill",
        "hawb",
        "mawb",
        "air waybill",
        "awb no",
        "shipper",
        "consignee",
        "departure",
        "destination",
        "pieces",
    ],

    # dc_005: Dispatch Clearance Certificate
    "dc_005": [
        "dispatch clearance certificate",
        "despatch clearance certificate",
        "dcc",
        "complete shipment",
        "partial shipment",
        "import licence",
        "import license",
        "tata steel",
        "supplier",
        # Additional discriminators present in real DCCs
        "dispatch clearance",
        "despatch clearance",
        "without prejudice",
        "sap reference",
    ],

    # dc_006: Supplier Invoice (non-TLL issuer)
    "dc_006": [
        "invoice",
        "purchaser's order",
        "purchaser",
        "tata steel limited",
        "invoice n.",
        "invoice no.",
        "payment terms",
        "bank",
        "iban",
        "vat",
    ],

    # dc_007: Packing List
    "dc_007": [
        "packing list",
        "gross wt",
        "nett wt",
        "net wt",
        "gross weight",
        "net weight",
        "dimensions",
        "packages",
        "marks & nos",
        "marks and nos",
        "qty",
        "shipping mark",
        "shipping",
        "consignee",
        "carton",
        "crate",
        "pcs",
        "pieces",
    ],

    # dc_008: Inspection Certificate
    "dc_008": [
        "inspection certificate",
        "ic number",
        "ic release date",
        "e/ic/",
        "category of inspection",
        "call no",
        "inspection quantity",
        "inspection",
    ],

    # dc_009: Dangerous Goods Declaration
    "dc_009": [
        "dangerous goods",
        "shipper's declaration",
        "shippers declaration",
        "un number",
        "hazard class",
        "packing group",
        "iata",
        "un3",
        "emergency contact",
    ],

    # dc_010: Order Acknowledgement
    "dc_010": [
        "order confirmation",
        "orderconfirmation",        # German SAP-style no-space variant
        "sales confirmation",
        "order acknowledgement",
        "order acknowledgment",
        "orderacknowledgment",      # concatenated no-space OCR variant
        "thank you for your order",
        "we confirm",
        "confirm with this document",
        "estimated delivery",
        "your p/o number",
        "your order",
        "sc number",                # sales confirmation number label
        # OCR-derived terms from scanned order confirmations
        "order date",
        "shipment date",
        "salesperson",
        "delivery date",
        "note of goods",
    ],

    # dc_011: TLL Sales Invoice (CLIENT INVOICE, not -A2)
    "dc_011": [
        "client invoice",
        "tata limited invoice no",
        "tata limited",
        "buying commission",
        "to freight",
        "to insurance",
        "indian import license",
        "grosvenor place",
        "invoice date",
    ],

    # dc_012: TLL A2 Commission Invoice
    "dc_012": [
        "client invoice -a2",
        "a2 invoice",
        "buying commission",
        "tata limited invoice no",
        "commission",
        "tata limited",
        "grosvenor place",
    ],

    # dc_013: Freight Agent Invoice (invoice with AWB/MAWB reference)
    "dc_013": [
        "mawb-no",
        "hawb-no",
        "mawb",
        "hawb",
        "freight",
        "shipper",
        "consignee",
        "tata limited",
        "inv-no",
        "acct-no",
        "invoice",
    ],

    # dc_014: Insurance Certificate
    "dc_014": [
        "certificate of insurance",
        "tata aig",
        "policy no",
        "taigd",
        "open policy",
        "insured",
        "certificate dated",
        "declared by",
        "tata ltd",
    ],

    # dc_015: RFQ
    "dc_015": [
        "request for quotation",
        "rfq",
        "rfq no",
        "submit your best offer",
        "please quote",
        "rfq due date",
        "best offer",
        "tata steel",
    ],

    # dc_016: Customs Release / Bill of Entry
    "dc_016": [
        "bill of entry",
        "home consumption",
        "indian customs",
        "icegate",
        "be no",
        "port code",
        "iec",
        "assessed",
        "customs",
    ],

    # dc_017: Quality / Test Certificate
    "dc_017": [
        "test certificate",
        "certificate of conformance",
        "material test",
        "test report",
        "quality certificate",
        "heat number",
        "heat no",
        "chemical composition",
        "mechanical properties",
        # OCR-derived terms from scanned test reports/certs
        "we hereby certify",
        "in compliance with",
        "tests executed",
        "examination of",
        "acc. to en",
        "manufacturer",
        # Heat treatment / IPPC / phytosanitary certificates
        "heat treatment certificate",
        "heat treatment",
        "phytosanitary",
        "ippc",
        "certificate of treatment",
        # CE marks / declarations of conformity
        "declaration of conformity",
        "ce marking",
        "certificate of compliance",
    ],

    # dc_018: FTA Certificate / Form I
    "dc_018": [
        "form-i",
        "form i",
        "free trade agreement",
        "certificate of origin",
        "fta",
        "preferential tariff",
        "origin criteria",
        # OCR-derived: Asia-Pacific Trade Agreement / APTA certs
        "asia-pacific trade agreement",
        "goods consigned from",
        "goods consigned to",
        "combined declaration and certificate",
        "country of origin",
        "issued in",
    ],

    # dc_019: Quotation / RFQ Response
    "dc_019": [
        "quotation",
        "quotation no",
        "quotation no.",
        "quotation number",
        "quotation date",
        "validity end date",
        "unit price",
        "list price",
        "list price",
        "price each",
        "price per",
        "discount",
        "incoterms",
        "part number",
        "we thank you for the inquiry",
        "thank you for your inquiry",
        "following quotation",
        "offer validity",
        "quotation ref",
    ],

    # dc_020: Customer Remittance Advice
    "dc_020": [
        "remittance advice",
        "remittance",
        "payment advice",
        "amount paid",
        "bank transfer",
        "payment date",
    ],
}


class ClassificationAgent(BaseAgent):
    """
    Scores each Document Class by keyword frequency in the PDF text.
    Returns the winning class ID, or None if no class exceeds the threshold.
    """

    name = "ClassificationAgent"

    def classify(self, doc: Document, pdf_bytes: bytes) -> tuple[str | None, float]:
        """
        Returns (class_id, confidence) where confidence is in [0.0, 1.0].

        Three-tier strategy:
          1. Manual override (human pinned the class) → instant return.
          2. Keyword scoring against classes that exist in the DB AND have rules.
          3. AI classification — used when no classes exist yet (fresh install) OR
             no keyword rule fires. AI proposes a class name; if it's new, creates
             a DocumentClass row in the DB on the fly.

        Returns (None, 0.0) only if all three tiers fail.
        """
        # Manual override bypasses auto-classification (confidence = 1.0 — human decided)
        if doc.document_class_override:
            return doc.document_class_override, 1.0

        text = _extract_pdf_text(pdf_bytes)
        if not text:
            return None, 0.0

        # ── Load existing DB classes ──────────────────────────────────────────────
        from app.models.document import DocumentClass
        db_classes = {dc.id: dc for dc in self.db.query(DocumentClass).all()}

        # If there are no document classes at all, skip straight to AI creation
        if not db_classes:
            logger.info("doc %s: no document classes in DB → AI classification", doc.id)
            return self._ai_classify(text, [])

        # ── High-priority pre-checks for classes that share many generic keywords ──

        # TLL Sales Invoice (dc_011) / A2 Commission Invoice (dc_012):
        # Both contain "client invoice" + "tata limited invoice no" — phrases unique
        # to Tata Limited's own invoice format. Checked before general scoring to
        # prevent dc_006 (Supplier Invoice) winning on raw keyword count.
        if "client invoice" in text and "tata limited invoice no" in text:
            if "client invoice -a2" in text or "a2 invoice" in text:
                return "dc_012", 1.0
            return "dc_011", 1.0

        # ── General keyword scoring ───────────────────────────────────────────────
        # Keywords come from the DB profile (classifier_profile_json) when present;
        # fall back to the static CLASSIFICATION_RULES dict if not yet seeded.
        # This lets operators edit keywords in the dashboard without code changes.
        import json as _json
        raw_scores: dict[str, int] = {}
        for class_id, dc_obj in db_classes.items():
            # Resolve keyword list: DB profile takes precedence over static dict
            db_keywords: list[str] | None = None
            neg_keywords: list[str] = []
            if dc_obj.classifier_profile_json:
                try:
                    profile = _json.loads(dc_obj.classifier_profile_json)
                    db_keywords   = profile.get("keywords") or []
                    neg_keywords  = profile.get("negative_keywords") or []
                except Exception:
                    pass
            keywords = db_keywords if db_keywords is not None else CLASSIFICATION_RULES.get(class_id, [])
            if not keywords:
                continue
            score = sum(1 for kw in keywords if kw in text)
            # Subtract for negative keyword hits
            score -= sum(1 for kw in neg_keywords if kw in text)
            raw_scores[class_id] = max(score, 0)

        scores = {dc: s for dc, s in raw_scores.items() if s >= MIN_MATCH_THRESHOLD}

        if not scores:
            # No keyword rule fired — fall back to AI classification.
            # AI will match against existing DB classes OR propose + create a new one.
            logger.info("doc %s: no keyword match → AI classification fallback", doc.id)
            return self._ai_classify(text, list(db_classes.values()))

        # Return highest scorer; prefer more specific classes in ties.
        # Use DB profile priority if available, otherwise fall back to static map.
        def _db_priority(class_id: str) -> int:
            dc_obj = db_classes.get(class_id)
            if dc_obj and dc_obj.classifier_profile_json:
                try:
                    p = _json.loads(dc_obj.classifier_profile_json)
                    return int(p.get("priority", _specificity(class_id)))
                except Exception:
                    pass
            return _specificity(class_id)

        winner = max(scores, key=lambda c: (scores[c], _db_priority(c)))

        # Confidence = fraction of the winning class's keywords that matched.
        dc_winner = db_classes.get(winner)
        if dc_winner and dc_winner.classifier_profile_json:
            try:
                _wp = _json.loads(dc_winner.classifier_profile_json)
                winner_kw_count = len(_wp.get("keywords") or []) or 1
            except Exception:
                winner_kw_count = len(CLASSIFICATION_RULES.get(winner, [])) or 1
        else:
            winner_kw_count = len(CLASSIFICATION_RULES.get(winner, [])) or 1
        raw_confidence = scores[winner] / winner_kw_count

        # ── Post-processing overrides ─────────────────────────────────────────────
        # Each early return uses raw_confidence from the original winner.
        # When we redirect to a different class, compute that class's confidence
        # so the stored value reflects the actual assigned class.

        def _conf(class_id: str) -> float:
            kw_count = len(CLASSIFICATION_RULES.get(class_id, [])) or 1
            return round(min(raw_scores.get(class_id, 0) / kw_count, 1.0), 4)

        # dc_011 vs dc_012: if -A2 marker present, use dc_012
        if winner == "dc_011" and "client invoice -a2" in text:
            return "dc_012", _conf("dc_012")

        # dc_006 vs dc_013: MAWB-NO / HAWB-NO labels are unique to TKM-style
        # freight agent invoices; "shipment no" is too generic (appears in
        # supplier invoice metadata) so is deliberately excluded here.
        if winner == "dc_006" and ("mawb-no" in text or "hawb-no" in text):
            return "dc_013", _conf("dc_013")

        # dc_006 vs dc_019: A document that explicitly says "quotation" (or
        # "quotation date") but won on dc_006 keywords (bank/iban/vat from
        # letterhead) is a quotation, not an invoice.
        # Guard: block only when an explicit invoice HEADER field appears —
        # i.e. "Invoice No." / "Invoice Date" followed by a value (digit or
        # colon), NOT when "invoice" merely appears in T&C boilerplate like
        # "should an invoice not be paid" (which contains "invoice no" as a
        # substring of "invoice not") or "state the invoice number in your
        # payment" (no digit follows directly).
        import re as _re
        _has_invoice_header = bool(_re.search(
            r"invoice\s*(?:no\.?|number|date|ref\.?|#)\s*[:\-\.\s]*\d",
            text, _re.IGNORECASE
        )) or any(k in text for k in ("tax invoice", "proforma invoice", "pro-forma invoice"))
        if winner == "dc_006" and (
            "quotation" in text or "quotation date" in text or "quotation no" in text
        ) and not _has_invoice_header:
            return "dc_019", _conf("dc_019")

        # dc_006 vs dc_017: Heat treatment certs, phytosanitary certs, and
        # CE declarations score dc_006 due to company bank details on letterhead,
        # but explicit certificate phrases override.
        if winner == "dc_006" and any(k in text for k in (
            "heat treatment certificate", "heat treatment", "phytosanitary",
            "certificate of treatment", "declaration of conformity", "ce marking",
            "we hereby certify", "certificate of compliance",
        )) and "invoice" not in text:
            return "dc_017", _conf("dc_017")

        # dc_006 vs dc_010: Supplier invoices and order confirmations share many
        # keywords. If the text contains unambiguous order-confirmation phrases,
        # prefer dc_010 over dc_006 even when invoice keywords win on count.
        # GUARD: supplier quotations include T&C sections ("present offer/order
        # confirmation", "general conditions of order acceptance") — if the document
        # also has "quotation" in text, the OC phrase is incidental boilerplate, not
        # the document type.
        if winner == "dc_006" and any(k in text for k in (
            "order confirmation", "orderconfirmation", "sales confirmation",
            "order acknowledgement", "order acknowledgment", "orderacknowledgment",
        )) and "quotation" not in text:
            return "dc_010", _conf("dc_010")

        # dc_017 vs TSL PO (dc_003): Large multi-page TSL POs attach quality/conformity
        # requirements in their T&C annexure (e.g. "PLEASE PROVIDE AN INSPECTION/TEST
        # CERTIFICATE", phytosanitary clauses) that score heavily for dc_017. The phrase
        # "please supply and deliver" + "tsl/" is the unambiguous TSL PO header pair.
        if winner == "dc_017" and "please supply and deliver" in text and (
            "tsl/" in text or "purchase order no" in text
        ):
            return "dc_003", _conf("dc_003")

        # dc_017 vs Quotation (dc_019): Supplier quotations that describe certified/compliant
        # products mention "manufacturer", "in compliance with" etc. (→ dc_017 keywords) while
        # clearly being price quotations. Two explicit quotation keywords override.
        if (winner == "dc_017"
                and scores.get("dc_019", 0) >= 2
                and "quotation" in text):
            return "dc_019", _conf("dc_019")

        # dc_010 vs Quotation (dc_019): Multi-page supplier quotations include T&C boilerplate
        # sections titled "Order Confirmation" or "General Conditions of Order Acceptance"
        # that heavily inflate dc_010 scores, even though the document is clearly a price
        # quotation with no confirmed order number. Explicit "quotation" keyword wins.
        if (winner == "dc_010"
                and scores.get("dc_019", 0) >= 2
                and "quotation" in text):
            return "dc_019", _conf("dc_019")

        # dc_007 vs TML Import Contract (dc_001): "imports contract" is the unique
        # header phrase in all TML/Tata Motors import POs. Their Annexure A T&C section
        # is dense with packing/weight/shipping keywords, causing dc_007 to win on count.
        if winner == "dc_007" and "imports contract" in text and "tata motors" in text:
            return "dc_001", _conf("dc_001")

        # dc_007 vs TSL PO / rate contract (dc_003): "please supply and deliver" is the
        # TSL PO header boilerplate; "rate contract" flags blanket supply agreements.
        # Both types mention "packing list" in their T&C annexure (causing dc_007 to win).
        if winner == "dc_007" and (
            "please supply and deliver" in text or "rate contract" in text
        ) and "tata steel" in text:
            return "dc_003", _conf("dc_003")

        # dc_003 vs dc_007: TSL PO keywords are very broad and can match packing
        # lists that mention "Tata Steel" + "TSL/" as marks & nos. Explicit
        # "packing list" phrase is a reliable dc_007 discriminator — UNLESS the text
        # also contains "please supply and deliver", which is the unambiguous TSL PO
        # header and proves this is a purchase order, not a packing list.
        if winner == "dc_003" and "packing list" in text and "please supply and deliver" not in text:
            return "dc_007", _conf("dc_007")

        # dc_003 vs dc_017: Quality/test certificates that mention "Tata Steel"
        # may score dc_003 if test-cert keywords are sparse OCR output. Any
        # explicit test/quality cert phrase with a dc_017 match score ≥ 1 wins.
        # GUARD: TSL POs include phytosanitary/packing requirements in their T&C
        # annexure that match dc_017 keywords; "please supply and deliver" is the
        # unambiguous PO header that proves the document is not a test certificate.
        if (winner == "dc_003"
                and "please supply and deliver" not in text
                and scores.get("dc_017", 0) >= 1
                and any(k in text for k in (
                    "test certificate", "certificate of conformance", "material test",
                    "test report", "quality certificate", "we hereby certify",
                    "chemical composition", "mechanical properties",
                ))):
            return "dc_017", _conf("dc_017")

        return winner, round(min(raw_confidence, 1.0), 4)

    # ──────────────────────────────────────────────────────────────────────────
    def _ai_classify(
        self,
        text: str,
        existing_classes: list,
    ) -> tuple[str | None, float]:
        """
        Use Claude to classify the document when keyword scoring fails or the DB
        has no classes yet.

        Strategy:
          • Give Claude the list of existing DocumentClass names so it can match
            against them if the document type is already known.
          • If Claude identifies a NEW type (nothing close in the existing list),
            create a DocumentClass row on the fly and return its new ID.
          • Returns (None, 0.0) only if the API key is missing or the call fails.
        """
        from app.config import settings as _cfg
        if not _cfg.anthropic_api_key:
            logger.warning("AI classification skipped — ANTHROPIC_API_KEY not set")
            return None, 0.0

        import anthropic
        import json
        import re
        import uuid

        options = [{"id": dc.id, "name": dc.name} for dc in existing_classes]
        options_block = (
            f"Existing document types in the system:\n{json.dumps(options, indent=2)}\n\n"
            if options else ""
        )

        prompt = (
            "You are classifying a business or logistics document.\n\n"
            + options_block
            + "Based on the document text below, determine the document type.\n"
            "Return ONLY this JSON (no markdown, no explanation):\n"
            '{"document_type": "2-4 word name", "match_id": "existing_id_or_null", "confidence": 0.0-1.0}\n\n'
            "Rules:\n"
            "  • If the document matches one of the existing types exactly, set match_id to that type's id.\n"
            "  • If it is a genuinely different type, set match_id to null and give a clear document_type name.\n"
            "  • document_type examples: 'Purchase Order', 'Commercial Invoice', 'Packing List',\n"
            "    'Airway Bill', 'Certificate of Origin', 'Freight Invoice', 'Bill of Lading'.\n"
            "  • confidence should reflect how certain you are (0.5 = unsure, 0.9 = very clear).\n\n"
            "Document text (first 4000 chars):\n---\n"
            + text[:4000]
            + "\n---"
        )

        try:
            client = anthropic.Anthropic(api_key=_cfg.anthropic_api_key)
            response = client.messages.create(
                model=_cfg.anthropic_model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            result = json.loads(raw)
        except Exception as exc:
            logger.warning("AI classification failed: %s", exc)
            return None, 0.0

        confidence = float(result.get("confidence", 0.70))
        match_id   = result.get("match_id")
        doc_type   = (result.get("document_type") or "Unknown Document").strip()

        # ── Matched an existing class ─────────────────────────────────────────
        if match_id and any(dc.id == match_id for dc in existing_classes):
            logger.info("AI classifier: matched existing class %s ('%s') conf=%.2f",
                        match_id, doc_type, confidence)
            return match_id, confidence

        # ── New type — create a DocumentClass on the fly ──────────────────────
        from app.models.document import DocumentClass
        from app.models.client import ClientProfile, DocumentTypeProfile
        import re as _re

        slug = _re.sub(r"[^a-z0-9]+", "-", doc_type.lower()).strip("-")
        new_id = "dc_" + uuid.uuid4().hex[:6]

        new_class = DocumentClass(
            id=new_id,
            name=doc_type,
            slug=slug,
            treatment="PROCESS",
        )
        self.db.add(new_class)
        self.db.flush()

        # Wire up a DTP for the default client profile
        client_profile = self.db.query(ClientProfile).first()
        if client_profile:
            dtp = DocumentTypeProfile(
                client_id=client_profile.id,
                document_class_id=new_id,
                confirmed=False,
                active=True,
            )
            self.db.add(dtp)

        self.db.commit()
        logger.info(
            "AI classifier: created new DocumentClass %s '%s' (slug=%s) conf=%.2f",
            new_id, doc_type, slug, confidence,
        )
        return new_id, confidence


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF, normalised to lowercase.

    Three-pass strategy:
      1. pypdf  — instant; works for PDFs with a text layer.
      2. OCR    — fallback when pypdf yields nothing (image-only / scanned PDFs).
                  Uses pdf2image → PIL images → pytesseract (Tesseract LSTM).
      3. Claude Vision — last resort when OCR also returns nothing (very low
                  quality scans, rotated pages, non-Latin scripts, etc.).
                  Asks the model to transcribe the document text verbatim.
                  Only used when ANTHROPIC_API_KEY is configured.
    """
    import re

    # ── Pass 1: pypdf ─────────────────────────────────────────────────────────
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        text = " ".join(parts).strip()
        if text:
            text = re.sub(r"\s+", " ", text)
            return text.lower()
    except Exception as exc:
        logger.debug("pypdf extraction failed: %s", exc)

    # ── Pass 2: OCR fallback ──────────────────────────────────────────────────
    ocr_text = _ocr_pdf_text(pdf_bytes)
    if ocr_text:
        return ocr_text

    # ── Pass 3: Claude Vision transcription ───────────────────────────────────
    return _vision_transcribe_pdf(pdf_bytes)


def _ocr_pdf_text(pdf_bytes: bytes) -> str:
    """
    Convert each PDF page to a raster image and OCR it with Tesseract.
    Returns normalised lowercase text, or empty string on any failure.

    DPI 300 is the sweet spot for Tesseract accuracy vs. speed on A4 docs.
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(pdf_bytes, dpi=300)
        parts = []
        for img in images:
            t = pytesseract.image_to_string(img, lang="eng")
            if t:
                parts.append(t)
        result = " ".join(parts).strip()
        if result:
            logger.debug("OCR extracted %d chars from scanned PDF", len(result))
        else:
            logger.debug("OCR returned no text — PDF may be blank or graphics-only")
        return result.lower()
    except Exception as exc:
        logger.debug("OCR fallback failed: %s", exc)
        return ""


def _vision_transcribe_pdf(pdf_bytes: bytes) -> str:
    """
    Pass 3: use Claude Vision to transcribe a PDF that both pypdf and OCR
    could not read (pure image scan, rotated text, non-Latin scripts, etc.).

    Renders up to 3 pages at 200 DPI, sends them as JPEG images to Claude
    claude-sonnet-4-6, and asks for a full verbatim text transcription.

    Returns normalised lowercase text, or empty string if unavailable.
    Only runs when ANTHROPIC_API_KEY is set in settings.
    """
    from app.config import settings as _settings

    if not _settings.anthropic_api_key:
        return ""

    try:
        import anthropic
        import base64

        logger.debug("Classification Pass 3: Claude Vision transcription for unreadable scan")

        # ── Render PDF pages to JPEG bytes ────────────────────────────────────────
        # Prefer PyMuPDF (pure Python, no poppler needed) over pdf2image.
        jpeg_pages: list[bytes] = []
        try:
            import fitz  # PyMuPDF
            doc_fitz = fitz.open(stream=pdf_bytes, filetype="pdf")
            mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
            for page_num in range(min(3, len(doc_fitz))):
                pix = doc_fitz[page_num].get_pixmap(matrix=mat)
                jpeg_pages.append(pix.tobytes("jpeg"))
        except ImportError:
            from pdf2image import convert_from_bytes
            pil_imgs = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=3)
            for img in pil_imgs[:3]:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                jpeg_pages.append(buf.getvalue())

        if not jpeg_pages:
            return ""

        image_blocks = []
        for jpeg_bytes in jpeg_pages:
            b64 = base64.standard_b64encode(jpeg_bytes).decode()
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

        image_blocks.append({
            "type": "text",
            "text": (
                "Please transcribe ALL visible text from this document image exactly as it appears. "
                "Include every word, number, label, header, footer, and reference number you can see. "
                "Preserve the structure where possible. "
                "Do not summarise — output the raw transcribed text only."
            ),
        })

        client = anthropic.Anthropic(api_key=_settings.anthropic_api_key)
        response = client.messages.create(
            model=_settings.anthropic_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": image_blocks}],
        )

        transcribed = response.content[0].text.strip()
        if transcribed:
            import re
            transcribed = re.sub(r"\s+", " ", transcribed)
            logger.info(
                "Classification Pass 3: Vision transcribed %d chars from unreadable scan",
                len(transcribed),
            )
            return transcribed.lower()
        return ""

    except Exception as exc:
        logger.warning("Classification Pass 3 (Vision) failed: %s", exc)
        return ""


def _specificity(class_id: str) -> int:
    """
    Tiebreaker: prefer classes with more discriminating keyword sets.
    Higher number = preferred when scores are equal.
    """
    priority = {
        "dc_012": 10,   # A2 invoice — most specific phrase
        "dc_013": 9,    # Freight invoice — MAWB/HAWB specific
        "dc_005": 9,    # DCC — very specific phrase
        "dc_008": 9,    # Inspection cert — E/IC/ format
        "dc_016": 9,    # Bill of Entry
        "dc_014": 8,    # Insurance cert
        "dc_018": 8,    # FTA
        "dc_009": 8,    # DGD
        "dc_010": 7,    # Order Acknowledgement — more specific than invoice
        "dc_011": 7,    # TLL Sales Invoice
        "dc_002": 7,    # TMPVL PO
        "dc_017": 7,    # Quality/Test Certificate — more specific than TSL PO
        "dc_001": 6,    # TML PO
        "dc_003": 5,    # TSL PO (broad keywords)
    }
    return priority.get(class_id, 3)
