"""
Governance Agent — AI-powered sanity check on classification + extraction coherence.

Invoked when:
  1. All 3 extraction tiers fail for a document (zero usable fields), OR
  2. Classification confidence < GOVERNANCE_CONFIDENCE_THRESHOLD (0.35)

The agent uses Claude Haiku (fast, cheap) to answer three questions:
  A. Does the assigned document class match what the document actually is?
  B. If not, which of the 20 known classes fits better (if any)?
  C. Is this a genuinely new document type not covered by any existing class?

Returns a structured GovernanceResult with:
  - verdict: "CORRECT" | "WRONG_CLASS" | "NEW_TYPE"
  - reasoning: plain-English explanation (stored in pipeline event)
  - suggested_class: dc_xxx if WRONG_CLASS (None otherwise)
  - suggested_class_name: human-readable name if NEW_TYPE
  - suggested_class_description: 1-sentence description if NEW_TYPE
  - suggested_keywords: list of discriminating terms if NEW_TYPE
  - confidence: 0.0–1.0 (how sure the governance agent is of its verdict)

The pipeline runner acts on the verdict:
  CORRECT    → NEEDS_REVIEW (extraction is genuinely hard, human needed)
  WRONG_CLASS → reclassify + retry extraction with the suggested class
  NEW_TYPE   → CANDIDATE_NEW_CLASS (surfaced in discovery queue for user)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.config import settings
from app.models.document import Document

logger = logging.getLogger(__name__)

# Confidence below this triggers governance review even if extraction passed partially
GOVERNANCE_CONFIDENCE_THRESHOLD = 0.35

# Known document classes — provided to the LLM for WRONG_CLASS suggestions
KNOWN_CLASSES = {
    "dc_001": "TML Import Contract PO — Tata Motors Limited import purchase orders",
    "dc_002": "TMPVL Purchase Order — Tata Motors Passenger Vehicles Limited POs",
    "dc_003": "Tata Steel Purchase Order — TSL purchase orders to suppliers",
    "dc_004": "Airway Bill / House AWB — IATA air waybills and house AWBs",
    "dc_005": "Dispatch Clearance Certificate — DCC issued by Tata Limited",
    "dc_006": "Supplier Invoice — commercial invoices from suppliers to Tata",
    "dc_007": "Packing List — packing lists from suppliers or shippers",
    "dc_008": "Inspection Certificate — third-party inspection certificates",
    "dc_009": "Dangerous Goods Declaration — IATA/IMDG DGDs from shippers",
    "dc_010": "Order Acknowledgement — supplier order confirmations",
    "dc_011": "TLL Sales Invoice — Tata Limited A-series commission invoices",
    "dc_012": "TLL A2 Invoice — Tata Limited A2 commission invoices",
    "dc_013": "Freight Agent Invoice — freight agent invoices with MAWB references",
    "dc_014": "Insurance Certificate — cargo insurance certificates",
    "dc_015": "RFQ — Request for Quotation issued by Tata Limited",
    "dc_016": "Customs Bill of Entry — Indian customs BE documents",
    "dc_017": "Quality / Test Certificate — material or conformance test certs",
    "dc_018": "FTA Certificate of Origin — Form I / preferential origin certs",
    "dc_019": "Quotation / RFQ Response — supplier price quotations",
    "dc_020": "Customer Remittance Advice — payment advices from customers",
}


@dataclass
class GovernanceResult:
    verdict: str                              # "CORRECT" | "WRONG_CLASS" | "NEW_TYPE"
    reasoning: str                            # Plain-English explanation
    confidence: float                         # Governance agent's self-assessed confidence
    suggested_class: str | None = None        # dc_xxx — only for WRONG_CLASS
    suggested_class_name: str | None = None   # Human name — only for NEW_TYPE
    suggested_class_description: str | None = None
    suggested_keywords: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggested_class": self.suggested_class,
            "suggested_class_name": self.suggested_class_name,
            "suggested_class_description": self.suggested_class_description,
            "suggested_keywords": self.suggested_keywords,
        }

    def as_event_detail(self) -> str:
        parts = [f"GovernanceAgent verdict: {self.verdict} (conf={self.confidence:.0%})."]
        parts.append(self.reasoning[:300])
        if self.suggested_class:
            parts.append(f"Suggested class: {self.suggested_class}.")
        if self.suggested_class_name:
            parts.append(f"Suggested new type: '{self.suggested_class_name}'.")
        return " ".join(parts)


class GovernanceAgent(BaseAgent):
    """
    Uses Claude claude-haiku-4-5-20251001 (fast + cheap) to assess classification coherence.
    Falls back to a heuristic rule if the API is unavailable.
    """

    name = "GovernanceAgent"

    def review(
        self,
        doc: Document,
        pdf_text: str,
        assigned_class_id: str | None,
        classification_confidence: float,
        extracted_field_names: list[str],
    ) -> GovernanceResult:
        """
        Run the governance review.

        Args:
            doc: the Document ORM object
            pdf_text: raw text extracted from the PDF (first 3000 chars used)
            assigned_class_id: the class the pipeline assigned (may be None)
            classification_confidence: 0.0–1.0 score from ClassificationAgent
            extracted_field_names: field names that were successfully extracted
                                   (empty list = all tiers failed)
        """
        if not settings.anthropic_api_key:
            logger.warning("GovernanceAgent: ANTHROPIC_API_KEY not set — using heuristic fallback")
            return self._heuristic_fallback(
                assigned_class_id, classification_confidence, extracted_field_names
            )

        try:
            return self._claude_review(
                pdf_text, assigned_class_id, classification_confidence, extracted_field_names
            )
        except Exception as exc:
            logger.error("GovernanceAgent: Claude call failed (%s) — using heuristic fallback", exc)
            return self._heuristic_fallback(
                assigned_class_id, classification_confidence, extracted_field_names
            )

    # ── Claude-backed review ──────────────────────────────────────────────────

    def _claude_review(
        self,
        pdf_text: str,
        assigned_class_id: str | None,
        classification_confidence: float,
        extracted_field_names: list[str],
    ) -> GovernanceResult:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        assigned_class_name = KNOWN_CLASSES.get(assigned_class_id or "", "Unknown / None")
        known_classes_block = "\n".join(
            f"  {dc_id}: {desc}" for dc_id, desc in KNOWN_CLASSES.items()
        )

        text_snippet = pdf_text[:3000].strip() if pdf_text else "(no text extracted — likely a scanned image)"

        prompt = f"""You are a document classification governance agent for Tata Limited's trade document pipeline.

You will review a document that was automatically classified. Your job is to assess whether the classification is correct, and if not, determine the most likely correct class or whether this is a genuinely new document type.

## Assigned Classification
- Class ID: {assigned_class_id or "None (unclassified)"}
- Class name: {assigned_class_name}
- Classifier confidence: {classification_confidence:.0%}
- Fields successfully extracted: {extracted_field_names if extracted_field_names else ["(none — all extraction tiers failed)"]}

## Document Text (first 3000 chars)
```
{text_snippet}
```

## Known Document Classes
{known_classes_block}

## Your Task
Respond with a JSON object (no markdown, no explanation outside the JSON) with these exact keys:

{{
  "verdict": "CORRECT" | "WRONG_CLASS" | "NEW_TYPE",
  "reasoning": "1-3 sentence plain-English explanation of your decision",
  "confidence": 0.0-1.0 (how confident you are in your verdict),
  "suggested_class": "dc_XXX" or null (only for WRONG_CLASS — the better-fitting known class),
  "suggested_class_name": "Short human-readable name" or null (only for NEW_TYPE),
  "suggested_class_description": "One sentence description" or null (only for NEW_TYPE),
  "suggested_keywords": ["keyword1", "keyword2", ...] or [] (only for NEW_TYPE — 5-10 discriminating terms from the document text)
}}

Rules:
- Verdict CORRECT: the assigned class is the right one; extraction failed due to format/scan quality, not wrong class.
- Verdict WRONG_CLASS: the document clearly belongs to a different known class. Set suggested_class to that dc_XXX.
- Verdict NEW_TYPE: the document does not fit any of the 20 known classes. This is a genuinely different document type.
- If no text was extracted (scanned image), use CORRECT if the class assignment seems plausible from the filename/context, or NEW_TYPE if uncertain.
- Be conservative: prefer CORRECT over NEW_TYPE unless the document is clearly not a trade document at all (e.g. a technical manual, an SDS safety sheet, a terms-and-conditions document).
"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        return GovernanceResult(
            verdict=data.get("verdict", "CORRECT"),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.5)),
            suggested_class=data.get("suggested_class"),
            suggested_class_name=data.get("suggested_class_name"),
            suggested_class_description=data.get("suggested_class_description"),
            suggested_keywords=data.get("suggested_keywords", []),
        )

    # ── Heuristic fallback (no API key) ──────────────────────────────────────

    def _heuristic_fallback(
        self,
        assigned_class_id: str | None,
        classification_confidence: float,
        extracted_field_names: list[str],
    ) -> GovernanceResult:
        """
        Simple rule-based fallback when Claude is unavailable.
        Conservative: never suggests a new type, just flags low-confidence mismatches.
        """
        if not assigned_class_id:
            return GovernanceResult(
                verdict="NEW_TYPE",
                reasoning="No class was assigned and Claude is unavailable for assessment. "
                          "Document requires manual classification.",
                confidence=0.4,
                suggested_class_name="Unknown Document Type",
                suggested_class_description="Could not be classified automatically.",
            )

        if classification_confidence >= GOVERNANCE_CONFIDENCE_THRESHOLD and not extracted_field_names:
            # Decent classification confidence but zero extraction — probably a scan
            return GovernanceResult(
                verdict="CORRECT",
                reasoning=f"Classification confidence ({classification_confidence:.0%}) is acceptable. "
                          "Extraction failure is likely due to scanned image — Tier 3 vision required.",
                confidence=0.6,
            )

        if classification_confidence < GOVERNANCE_CONFIDENCE_THRESHOLD:
            return GovernanceResult(
                verdict="NEW_TYPE",
                reasoning=f"Classification confidence is very low ({classification_confidence:.0%}) "
                          "and Claude is unavailable for further assessment. "
                          "Document may be a new type not covered by existing classes.",
                confidence=0.4,
                suggested_class_name="Unrecognised Document",
                suggested_class_description="Document did not match any known class with sufficient confidence.",
            )

        return GovernanceResult(
            verdict="CORRECT",
            reasoning="Classification appears plausible but extraction failed. Manual review required.",
            confidence=0.5,
        )
