"""
Client configuration models — multi-tenant support.

ClientProfile       — a customer/tenant using Dokr
DocumentTypeProfile — client-specific config for a document class
                      (custom field schema, learned from sample PDFs)
SampleDocument      — a sample PDF used to teach the AI for a doc type
ClientAgentConfig   — per-client pipeline agent configuration (plug-in overrides)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class ClientProfile(Base):
    """A client organisation using the Dokr platform."""

    __tablename__ = "client_profiles"

    id: Mapped[str] = mapped_column(
        String(30), primary_key=True, default=lambda: _new_id("cp")
    )
    name: Mapped[str] = mapped_column(String(120))          # Internal short name
    display_name: Mapped[str] = mapped_column(String(200))  # Full legal name
    domain: Mapped[Optional[str]] = mapped_column(String(120))   # e.g. "tata.co.uk"
    industry: Mapped[Optional[str]] = mapped_column(String(100)) # e.g. "Steel Manufacturing"
    erp_system: Mapped[Optional[str]] = mapped_column(String(80))# e.g. "Business Central"
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Free-form JSON for anything else (ERP endpoint, SharePoint site, etc.)
    settings_json: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document_type_profiles: Mapped[list["DocumentTypeProfile"]] = relationship(
        "DocumentTypeProfile",
        back_populates="client",
        cascade="all, delete-orphan",
    )
    agent_config: Mapped[Optional["ClientAgentConfig"]] = relationship(
        "ClientAgentConfig",
        back_populates="client",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ClientProfile {self.id} {self.name!r}>"


class DocumentTypeProfile(Base):
    """
    Client-specific extraction configuration for one document class.

    Operators upload sample PDFs → AI analyses them → suggests a field schema
    → operator confirms → schema is saved here and used for future extractions.
    """

    __tablename__ = "document_type_profiles"

    id: Mapped[str] = mapped_column(
        String(30), primary_key=True, default=lambda: _new_id("dtp")
    )
    client_id: Mapped[str] = mapped_column(
        String(30), ForeignKey("client_profiles.id"), nullable=False
    )
    document_class_id: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # e.g. "dc_015" — soft FK to document_classes.id

    # ── Learning stage ────────────────────────────────────────────────────────
    # ZERO_SHOT        — 0-2 docs seen; AI extracts blind, no assumed schema
    # LEARNING         — 3+ docs; AI uses accumulated examples as hints
    # LEARNED_PROPOSED — system confident enough to propose schema; awaiting confirmation
    # LEARNED          — operator confirmed schema; AI uses it as ground truth
    # OPTIMISED        — 25+ confirmed docs; fast patterns generated; AI spot-checks
    learning_stage: Mapped[str] = mapped_column(String(20), default="ZERO_SHOT")

    # Total real documents processed (not sample uploads)
    doc_count: Mapped[int] = mapped_column(Integer, default=0)

    # Field statistics accumulated across all extractions.
    # JSON: {field_name: {total_seen, found_count, confidence_sum, examples: [...]}}
    field_stats_json: Mapped[Optional[str]] = mapped_column(Text)

    # When system proposed schema for confirmation (triggers badge in Setup nav)
    schema_proposed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # ── Field schema ─────────────────────────────────────────────────────────
    # JSON: {"field_name": {"description": "...", "required": true, "format_hint": "..."}}
    # Populated either by: (a) AI discovery from sample PDFs, or (b) auto-generated
    # from field_stats once LEARNED_PROPOSED threshold is reached.
    # When confirmed=True this schema is the source-of-truth for extraction.
    field_schema_json: Mapped[Optional[str]] = mapped_column(Text)

    # ── Parsability assessment ────────────────────────────────────────────────
    # UNKNOWN       — not yet assessed
    # OPTIMISABLE   — structured, consistent format; regex can handle it once learned
    # CAN_LEARN     — semi-structured; AI recommended but patterns can be cached
    # ALWAYS_AI     — complex/variable layout, handwriting, or scans; always needs Claude
    parsability: Mapped[str] = mapped_column(String(20), default="UNKNOWN")
    parsability_reason: Mapped[Optional[str]] = mapped_column(Text)
    parsability_assessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # AI-generated fast-path patterns (for OPTIMISED stage).
    # JSON: {field_name: {patterns: ["regex1", ...], format_hint: "..."}}
    generated_patterns_json: Mapped[Optional[str]] = mapped_column(Text)

    # Number of sample PDFs that have been uploaded and analysed
    sample_count: Mapped[int] = mapped_column(Integer, default=0)

    # AI-generated summary of what this document type looks like for this client
    ai_description: Mapped[Optional[str]] = mapped_column(Text)

    # ── Signature detection toggle ────────────────────────────────────────────
    # When True, the ExtractionAgent runs a signature detection pass after
    # extracting fields and stores is_signed + evidence on the Document.
    # Default: False (opt-in per document class).
    check_signature: Mapped[bool] = mapped_column(Boolean, default=False)

    # True = operator has reviewed and confirmed the schema
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    client: Mapped["ClientProfile"] = relationship(
        "ClientProfile", back_populates="document_type_profiles"
    )
    samples: Mapped[list["SampleDocument"]] = relationship(
        "SampleDocument",
        back_populates="document_type_profile",
        cascade="all, delete-orphan",
        order_by="SampleDocument.created_at",
    )

    def field_schema(self) -> dict:
        """Return field_schema_json parsed as a dict, or {} if not set."""
        if not self.field_schema_json:
            return {}
        try:
            import json
            return json.loads(self.field_schema_json)
        except Exception:
            return {}

    def field_stats(self) -> dict:
        """Return field_stats_json parsed as a dict, or {} if not set."""
        if not self.field_stats_json:
            return {}
        try:
            import json
            return json.loads(self.field_stats_json)
        except Exception:
            return {}

    def computed_field_stats(self) -> list[dict]:
        """
        Return a list of per-field stat dicts with derived metrics, sorted by
        occurrence_rate descending. Used by the Setup UI.
        """
        raw = self.field_stats()
        result = []
        total_docs = max(self.doc_count or 0, 1)
        for fname, s in raw.items():
            found = s.get("found_count", 0)
            seen  = s.get("total_seen", total_docs)
            conf_sum = s.get("confidence_sum", 0.0)
            result.append(dict(
                field=fname,
                occurrence_rate=round(found / max(seen, 1), 2),
                avg_confidence=round(conf_sum / max(found, 1), 2) if found else 0.0,
                examples=s.get("examples", [])[:3],
                total_seen=seen,
                found_count=found,
            ))
        result.sort(key=lambda x: x["occurrence_rate"], reverse=True)
        return result

    @property
    def stage_label(self) -> str:
        return {
            "ZERO_SHOT":        "Learning (0–2 docs)",
            "LEARNING":         "Learning (tracking patterns)",
            "LEARNED_PROPOSED": "Ready to confirm",
            "LEARNED":          "Schema confirmed",
            "OPTIMISED":        "Optimised (fast path)",
        }.get(self.learning_stage, self.learning_stage)

    @property
    def stage_pct(self) -> int:
        """0–100 progress towards OPTIMISED, for UI progress bar."""
        return {
            "ZERO_SHOT":        5,
            "LEARNING":         min(5 + int((self.doc_count or 0) * 4.5), 49),
            "LEARNED_PROPOSED": 55,
            "LEARNED":          70,
            "OPTIMISED":        100,
        }.get(self.learning_stage, 0)

    @property
    def needs_attention(self) -> bool:
        """True when operator action is required (confirmation pending)."""
        return self.learning_stage == "LEARNED_PROPOSED"

    def __repr__(self) -> str:
        return f"<DocumentTypeProfile {self.id} client={self.client_id} dc={self.document_class_id}>"


class SampleDocument(Base):
    """
    A sample PDF uploaded to teach the AI about a client's document type.

    The extracted text and AI analysis are stored here so that re-running
    discovery doesn't require re-uploading the PDFs.
    """

    __tablename__ = "sample_documents"

    id: Mapped[str] = mapped_column(
        String(30), primary_key=True, default=lambda: _new_id("sd")
    )
    document_type_profile_id: Mapped[str] = mapped_column(
        String(30),
        ForeignKey("document_type_profiles.id"),
        nullable=False,
    )
    file_name: Mapped[str] = mapped_column(String(300))
    file_sha256: Mapped[str] = mapped_column(String(64))
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    # Raw text extracted from the PDF (stored so AI can re-analyse without re-upload)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text)

    # AI analysis of this sample:
    # {fields_found: [...], confidence_scores: {...}, notes: "..."}
    ai_analysis_json: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # Relationships
    document_type_profile: Mapped["DocumentTypeProfile"] = relationship(
        "DocumentTypeProfile", back_populates="samples"
    )

    def __repr__(self) -> str:
        return f"<SampleDocument {self.id} {self.file_name!r}>"


class ClientAgentConfig(Base):
    """
    Per-client pipeline agent configuration.

    Controls which pipeline stages are disabled for this client and which agent
    classes are used in place of the defaults.  One row per client (unique on
    client_id).

    If no row exists for a client, all defaults apply:
      • all stages run (subject to per-document skip_stages and treatment routing)
      • default agent classes from PIPELINE_AGENT_REGISTRY are used

    agent_overrides_json example
    ----------------------------
    {
        "extraction": "app.agents.extraction_freight.FreightExtractionAgent",
        "validation": "app.agents.validation_steel.SteelValidationAgent"
    }
    Stage keys are lowercase and must match PIPELINE_AGENT_REGISTRY.

    disabled_stages_json example
    ----------------------------
    ["MATCHING", "POSTING"]
    Uses the same uppercase keys as Document.skip_stages.

    stage_params_json example
    -------------------------
    {
        "validation": {"strict_nigo": true, "confidence_threshold": 0.92},
        "extraction": {"max_pages": 20}
    }
    Params are passed to the agent via BaseAgent.config at instantiation.
    """

    __tablename__ = "client_agent_configs"

    id: Mapped[str] = mapped_column(
        String(30), primary_key=True, default=lambda: _new_id("cac")
    )
    client_id: Mapped[str] = mapped_column(
        String(30), ForeignKey("client_profiles.id"), nullable=False, unique=True
    )

    # ── Stage enablement ─────────────────────────────────────────────────────
    # JSON list of uppercase stage names to DISABLE (mirrors skip_stages convention).
    # e.g. ["MATCHING", "POSTING"]
    # NULL → no client-level stages are disabled.
    disabled_stages_json: Mapped[Optional[str]] = mapped_column(Text)

    # ── Agent class overrides ────────────────────────────────────────────────
    # JSON dict: {lowercase_stage_key: "fully.qualified.ClassName"}
    # e.g. {"extraction": "app.agents.extraction_freight.FreightExtractionAgent"}
    # Stages absent from this dict use the default from PIPELINE_AGENT_REGISTRY.
    agent_overrides_json: Mapped[Optional[str]] = mapped_column(Text)

    # ── Per-stage parameters ─────────────────────────────────────────────────
    # JSON dict: {lowercase_stage_key: {param: value, ...}}
    # Passed to the agent as BaseAgent.config at instantiation time.
    # e.g. {"validation": {"strict_nigo": true, "confidence_threshold": 0.92}}
    stage_params_json: Mapped[Optional[str]] = mapped_column(Text)

    # ── Display / audit ───────────────────────────────────────────────────────
    label: Mapped[Optional[str]] = mapped_column(String(120))  # e.g. "Tata Steel Pipeline"
    notes: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    client: Mapped["ClientProfile"] = relationship(
        "ClientProfile", back_populates="agent_config"
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def disabled_stages(self) -> set[str]:
        """Return the set of uppercase stage names disabled for this client."""
        if not self.disabled_stages_json:
            return set()
        try:
            return set(json.loads(self.disabled_stages_json))
        except Exception:
            return set()

    def agent_overrides(self) -> dict[str, str]:
        """Return {stage_key: dotted_class_path} overrides dict."""
        if not self.agent_overrides_json:
            return {}
        try:
            return json.loads(self.agent_overrides_json)
        except Exception:
            return {}

    def stage_params(self) -> dict[str, dict]:
        """Return {stage_key: {param: value}} config dict."""
        if not self.stage_params_json:
            return {}
        try:
            return json.loads(self.stage_params_json)
        except Exception:
            return {}

    def __repr__(self) -> str:
        return f"<ClientAgentConfig {self.id} client={self.client_id!r}>"
