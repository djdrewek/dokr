"""
SchemaLearnerAgent — accumulates field statistics across document extractions
and drives the learning lifecycle for each DocumentTypeProfile.

Learning stages
───────────────
ZERO_SHOT        0–2 docs  AI extracts blind, no schema assumed. Lenient pass.
LEARNING         3–N docs  AI uses prior examples as hints. Tracks field patterns.
LEARNED_PROPOSED N docs + quality threshold met → system proposes schema for confirm.
LEARNED          Operator confirmed schema. AI extracts against it precisely.
OPTIMISED        25+ confirmed docs + OPTIMISABLE parsability → fast patterns active.

Parsability assessment
──────────────────────
Once LEARNED, the agent asks Claude to assess whether this document type can
eventually be handled by regex/text rules (OPTIMISABLE), needs AI but patterns
can cache (CAN_LEARN), or always requires full AI reading (ALWAYS_AI).

This assessment is shown in the Setup page so operators set expectations correctly:
"This doc type will always need an AI call — it's too variable/handwritten/complex."
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.config import settings
from app.models.document import Document, DocumentClass
from app.models.extracted_field import ExtractedField

logger = logging.getLogger(__name__)

# ── Stage advancement thresholds ──────────────────────────────────────────────
ZERO_TO_LEARNING        = 3     # docs before we start tracking patterns
LEARNING_TO_PROPOSED    = 10    # docs before proposing schema confirmation
MIN_OCCURRENCE_FOR_REQ  = 0.70  # field must appear in ≥70% of docs to be "required"
MIN_OCCURRENCE_FOR_OPT  = 0.40  # field must appear in ≥40% to be "optional"
MIN_AVG_CONFIDENCE      = 0.76  # minimum avg confidence across present fields
MIN_FIELD_COVERAGE      = 0.55  # at least 55% of target fields must meet thresholds
LEARNED_TO_OPTIMISED    = 25    # confirmed docs before parsability assessment


class SchemaLearnerAgent(BaseAgent):
    """
    Accumulates knowledge from every extraction run and advances the learning
    lifecycle for a DocumentTypeProfile.
    """

    name = "SchemaLearnerAgent"

    # ── Public API ──────────────────────────────────────────────────────────

    def record_extraction(
        self,
        doc: Document,
        raw_fields: dict[str, tuple[str, float]],
        extracted_text: str = "",
        target_fields: Optional[list[str]] = None,
    ) -> None:
        """
        Called after every AI extraction attempt. Updates field stats for the
        matching DocumentTypeProfile and checks whether the stage should advance.
        """
        dtp = self._get_dtp(doc.document_class_id)
        if not dtp:
            return

        if target_fields is None:
            from app.agents.extraction import FIELDS_BY_CLASS
            target_fields = FIELDS_BY_CLASS.get(doc.document_class_id, [])

        stats = dtp.field_stats()

        # Update per-field statistics
        for fname in target_fields:
            s = stats.setdefault(fname, {
                "total_seen":     0,
                "found_count":    0,
                "confidence_sum": 0.0,
                "examples":       [],
            })
            s["total_seen"] += 1
            if fname in raw_fields:
                val, conf = raw_fields[fname]
                s["found_count"]    += 1
                s["confidence_sum"] += conf
                if val and val not in s["examples"] and len(s["examples"]) < 6:
                    s["examples"].append(val)

        dtp.field_stats_json = json.dumps(stats)
        dtp.doc_count = (dtp.doc_count or 0) + 1

        # Check whether the stage should advance
        self._check_stage_advancement(dtp, stats, target_fields)

        try:
            self.db.commit()
        except Exception as exc:
            logger.warning("SchemaLearnerAgent: commit failed: %s", exc)
            self.db.rollback()

    def get_schema_hints(
        self,
        doc_class_id: str,
    ) -> dict:
        """
        Return schema hints for the AI extraction prompt, tailored to the current
        learning stage of this document type's profile.

        Returns a dict that _extract_via_ai_text uses to enrich the prompt:
          {"field_name": {"example_values": [...], "description_override": "..."}}
        """
        dtp = self._get_dtp(doc_class_id)
        if not dtp:
            return {}

        stage = dtp.learning_stage

        # LEARNED / OPTIMISED — use confirmed schema as precise instructions
        if stage in ("LEARNED", "OPTIMISED") and dtp.field_schema_json:
            try:
                schema = json.loads(dtp.field_schema_json)
                # Convert to hints format with description overrides
                hints: dict = {}
                for fname, fdata in schema.items():
                    if isinstance(fdata, dict):
                        hints[fname] = {k: v for k, v in fdata.items() if v}
                return hints
            except Exception:
                pass

        # LEARNING — use accumulated examples as weak hints
        if stage in ("LEARNING", "LEARNED_PROPOSED") and dtp.field_stats_json:
            try:
                stats = json.loads(dtp.field_stats_json)
                hints = {}
                doc_count = max(dtp.doc_count or 1, 1)
                for fname, s in stats.items():
                    found = s.get("found_count", 0)
                    if found > 0 and s.get("examples"):
                        occ = round(found / doc_count, 2)
                        hints[fname] = {
                            "example_values": s["examples"][:3],
                            "occurrence_note": f"found in {occ:.0%} of prior docs",
                        }
                return hints
            except Exception:
                pass

        return {}  # ZERO_SHOT — no hints; AI works blind

    def run_parsability_assessment(self, doc_class_id: str) -> None:
        """
        Ask Claude whether this document type can ever be handled by fast text
        parsing, or always requires AI. Updates dtp.parsability and
        dtp.parsability_reason. Safe to call multiple times (idempotent).
        """
        if not settings.anthropic_api_key:
            return

        dtp = self._get_dtp(doc_class_id)
        if not dtp:
            return

        stats = dtp.field_stats()
        if not stats:
            return

        dc = self.db.query(DocumentClass).filter(
            DocumentClass.id == doc_class_id
        ).first()
        dc_name = dc.name if dc else doc_class_id

        # Build a summary of what we've seen for the prompt
        field_summary_lines: list[str] = []
        doc_count = max(dtp.doc_count or 0, 1)
        for fname, s in stats.items():
            found = s.get("found_count", 0)
            occ   = round(found / doc_count, 2)
            avg_c = round(s.get("confidence_sum", 0) / max(found, 1), 2)
            exs   = ", ".join(s.get("examples", [])[:3])
            field_summary_lines.append(
                f"  {fname}: found in {occ:.0%} of docs, avg confidence {avg_c:.0%}"
                + (f", e.g. {exs}" if exs else "")
            )

        field_summary = "\n".join(field_summary_lines)
        ai_desc = dtp.ai_description or "(no description yet)"

        prompt = (
            f"You are assessing whether a document type can be reliably processed "
            f"by fast text-parsing rules (regex) after training, or always requires "
            f"full AI reading.\n\n"
            f"Document type: {dc_name}\n"
            f"AI description: {ai_desc}\n"
            f"Docs analysed: {doc_count}\n\n"
            f"Field extraction statistics so far:\n{field_summary}\n\n"
            f"Classify this document type as ONE of:\n"
            f"  OPTIMISABLE — Structured, consistent layout. Regex can handle it reliably "
            f"once field formats are learned. Fast path achievable.\n"
            f"  CAN_LEARN   — Semi-structured. Field positions/labels vary but AI confidence "
            f"is high. AI will always be faster/more accurate but caching can reduce cost.\n"
            f"  ALWAYS_AI   — Complex or variable. Handwriting, free-form text, heavy tables, "
            f"scanned/image-only pages, or highly inconsistent layouts. Regex cannot reliably "
            f"replace AI for this type.\n\n"
            f"Return JSON only:\n"
            f'{{ "parsability": "OPTIMISABLE|CAN_LEARN|ALWAYS_AI", '
            f'"reason": "one sentence explanation", '
            f'"confidence": 0.0-1.0 }}'
        )

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))

            parsability = data.get("parsability", "UNKNOWN")
            if parsability not in ("OPTIMISABLE", "CAN_LEARN", "ALWAYS_AI"):
                parsability = "UNKNOWN"

            dtp.parsability             = parsability
            dtp.parsability_reason      = data.get("reason", "")
            dtp.parsability_assessed_at = datetime.utcnow()
            self.db.commit()
            logger.info(
                "Parsability assessment for %s: %s — %s",
                doc_class_id, parsability, dtp.parsability_reason,
            )
        except Exception as exc:
            logger.warning("Parsability assessment failed: %s", exc)

    def generate_fast_patterns(self, doc_class_id: str) -> None:
        """
        For OPTIMISABLE document types: ask Claude to generate regex patterns
        from the accumulated field examples. Stored in generated_patterns_json.
        Only called when transitioning to OPTIMISED.
        """
        if not settings.anthropic_api_key:
            return

        dtp = self._get_dtp(doc_class_id)
        if not dtp or not dtp.field_stats_json:
            return

        stats = json.loads(dtp.field_stats_json)
        dc = self.db.query(DocumentClass).filter(DocumentClass.id == doc_class_id).first()
        dc_name = dc.name if dc else doc_class_id

        # Build example table for Claude
        field_examples_lines: list[str] = []
        for fname, s in stats.items():
            exs = s.get("examples", [])
            if exs:
                field_examples_lines.append(f'  "{fname}": {json.dumps(exs[:5])}')

        if not field_examples_lines:
            return

        prompt = (
            f"Generate regex patterns for extracting fields from '{dc_name}' documents.\n\n"
            "For each field, I'll give you example values observed in real documents.\n"
            "Return a JSON object where each key is the field name and the value is:\n"
            '  {"pattern": "regex_that_matches", "group": 1, "confidence": 0.85-0.95}\n\n'
            "The regex should:\n"
            "- Match in MULTILINE, IGNORECASE mode\n"
            "- Use a capture group for the actual value\n"
            "- Anchor to nearby label text for precision (not just the value alone)\n"
            "- Return null for a field if examples are too variable for reliable regex\n\n"
            f"Field examples from {dtp.doc_count} real documents:\n"
            "{\n" + ",\n".join(field_examples_lines) + "\n}\n\n"
            "Return ONLY the JSON object, no explanation."
        )

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            patterns = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            # Filter out nulls
            patterns = {k: v for k, v in patterns.items() if v}
            dtp.generated_patterns_json = json.dumps(patterns)
            self.db.commit()
            logger.info(
                "Generated %d fast-path patterns for %s", len(patterns), doc_class_id
            )
        except Exception as exc:
            logger.warning("Fast pattern generation failed: %s", exc)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_dtp(self, doc_class_id: str):
        """Find the active DocumentTypeProfile for this doc class (any client)."""
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

    def _check_stage_advancement(
        self,
        dtp,
        stats: dict,
        target_fields: list[str],
    ) -> None:
        """Advance learning_stage if thresholds are met. Mutates dtp in-place."""
        stage     = dtp.learning_stage
        doc_count = dtp.doc_count or 0

        # ZERO_SHOT → LEARNING (just needs a few docs)
        if stage == "ZERO_SHOT" and doc_count >= ZERO_TO_LEARNING:
            dtp.learning_stage = "LEARNING"
            logger.info("DTP %s advanced ZERO_SHOT → LEARNING (%d docs)", dtp.id, doc_count)
            return

        # LEARNING → LEARNED_PROPOSED (quality threshold)
        if stage == "LEARNING" and doc_count >= LEARNING_TO_PROPOSED:
            if self._quality_threshold_met(stats, target_fields, doc_count):
                dtp.learning_stage    = "LEARNED_PROPOSED"
                dtp.schema_proposed_at = datetime.utcnow()
                # Auto-generate the proposed schema from accumulated stats
                self._auto_generate_schema(dtp, stats, target_fields, doc_count)
                logger.info(
                    "DTP %s advanced LEARNING → LEARNED_PROPOSED (%d docs) — schema ready for confirmation",
                    dtp.id, doc_count,
                )

        # LEARNED → check parsability + consider OPTIMISED
        # (LEARNED → OPTIMISED happens via operator action in the Setup page,
        #  after parsability assessment returns OPTIMISABLE. We only auto-trigger
        #  the assessment here, not the transition itself.)
        if stage == "LEARNED" and doc_count >= LEARNED_TO_OPTIMISED:
            if dtp.parsability == "UNKNOWN":
                # Fire assessment asynchronously (best-effort; not awaited)
                try:
                    self.run_parsability_assessment(dtp.document_class_id)
                except Exception as exc:
                    logger.debug("Parsability assessment deferred: %s", exc)

    def _quality_threshold_met(
        self,
        stats: dict,
        target_fields: list[str],
        doc_count: int,
    ) -> bool:
        """
        Return True if field statistics are consistent enough to propose a schema.
        Requires MIN_FIELD_COVERAGE of target fields to have:
          - occurrence_rate ≥ MIN_OCCURRENCE_FOR_OPT
          - avg_confidence  ≥ MIN_AVG_CONFIDENCE
        """
        if not target_fields:
            return False
        qualifying = 0
        for fname in target_fields:
            s = stats.get(fname, {})
            found = s.get("found_count", 0)
            total = max(s.get("total_seen", doc_count), 1)
            occ   = found / total
            conf  = s.get("confidence_sum", 0.0) / max(found, 1) if found else 0.0
            if occ >= MIN_OCCURRENCE_FOR_OPT and conf >= MIN_AVG_CONFIDENCE:
                qualifying += 1

        coverage = qualifying / len(target_fields)
        return coverage >= MIN_FIELD_COVERAGE

    def _auto_generate_schema(
        self,
        dtp,
        stats: dict,
        target_fields: list[str],
        doc_count: int,
    ) -> None:
        """
        Build a field_schema_json from accumulated stats. Uses FIELD_DESCRIPTIONS
        for base descriptions and enriches with observed format hints + examples.
        Only overwrites if not already confirmed by operator.
        """
        if dtp.confirmed:
            return  # Don't overwrite a human-confirmed schema

        from app.agents.extraction import FIELD_DESCRIPTIONS

        schema: dict = {}
        for fname in target_fields:
            s        = stats.get(fname, {})
            found    = s.get("found_count", 0)
            total    = max(s.get("total_seen", doc_count), 1)
            occ      = round(found / total, 2)
            avg_conf = round(s.get("confidence_sum", 0.0) / max(found, 1), 2) if found else 0.0
            examples = s.get("examples", [])

            schema[fname] = {
                "description":  FIELD_DESCRIPTIONS.get(fname, fname.replace("_", " ").title()),
                "required":     occ >= MIN_OCCURRENCE_FOR_REQ,
                "occurrence_rate": occ,
                "avg_confidence":  avg_conf,
                "format_hint":  _infer_format_hint(examples),
                "example_values": examples[:3],
            }


        dtp.field_schema_json = json.dumps(schema)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _infer_format_hint(examples: list[str]) -> str:
    """
    Delegate to the FormatAgent's full pattern classifier.
    Replaces the old 5-pattern stub with a comprehensive type inference engine.
    """
    try:
        from app.agents.format_agent import infer_format_hint as _fmt_hint
        return _fmt_hint(examples)
    except Exception:
        # Fallback: legacy lightweight version
        if not examples:
            return ""
        sample = examples[0]
        if re.match(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", sample):
            return "date DD/MM/YYYY"
        if re.match(r"\d{4}[\/\-]\d{2}[\/\-]\d{2}", sample):
            return "date YYYY-MM-DD"
        if re.match(r"[£€$][\d,]+\.?\d*", sample) or re.match(r"[\d,]+\.\d{2}$", sample):
            return "monetary amount"
        if re.match(r"[A-Z]{2,6}[\/\-]\d{4,}", sample):
            return f"reference like {sample}"
        if re.match(r"\d{8,}", sample):
            return "numeric reference"
        return ""
