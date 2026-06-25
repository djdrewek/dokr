"""
Pipeline Agent Registry — plug-in system for per-client agent specialisation.

Every pipeline stage has a default agent class registered here.  Per-client
overrides are stored in ClientAgentConfig.agent_overrides_json as dotted class
paths, e.g. "app.agents.extraction_freight.FreightExtractionAgent".

Usage — in runner.py
--------------------
    from app.agents.registry import get_pipeline_agent_class

    # Default agent:
    cls = get_pipeline_agent_class("extraction")
    agent = cls(db)

    # Per-client override:
    cls = get_pipeline_agent_class(
        "extraction",
        override="app.agents.extraction_freight.FreightExtractionAgent",
    )
    agent = cls(db)

    # With stage params (config dict passed to __init__):
    agent = cls(db, config={"confidence_threshold": 0.92})

Usage — registering a custom default (replaces globally)
---------------------------------------------------------
    from app.agents.registry import register_pipeline_agent
    from app.agents.extraction import ExtractionAgent

    @register_pipeline_agent("extraction")
    class FreightExtractionAgent(ExtractionAgent):
        ...

    # For per-client use without touching the global default, reference
    # the class by dotted path in ClientAgentConfig.agent_overrides_json.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Registry — populated by _register_defaults() below
# ─────────────────────────────────────────────────────────────────────────────

# Maps stage key (lowercase) → agent class
PIPELINE_AGENT_REGISTRY: dict[str, type] = {}


def register_pipeline_agent(stage: str):
    """
    Class decorator that registers an agent as the *global* default for a stage.

    @register_pipeline_agent("extraction")
    class FreightExtractionAgent(ExtractionAgent):
        ...

    To specialise per-client without touching the global default, set
    ClientAgentConfig.agent_overrides_json = {"extraction": "my.module.MyAgent"}.
    """
    def _decorator(cls):
        PIPELINE_AGENT_REGISTRY[stage] = cls
        cls.pipeline_stage = stage
        return cls
    return _decorator


# ─────────────────────────────────────────────────────────────────────────────
#  Stage metadata catalogue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStageInfo:
    """Descriptor for a single pipeline stage."""
    key: str           # lowercase, used as registry key and in agent_overrides_json
    label: str         # human-readable name shown in the UI
    description: str
    skippable: bool    # can this stage be disabled per-client or per-document?
    default_class: str # dotted import path of the default agent class


PIPELINE_STAGES: list[PipelineStageInfo] = [
    # ── Required stages — classification and extraction are the irreducible core.
    # ── All other stages can be disabled per-client via ClientAgentConfig.
    PipelineStageInfo(
        key="splitting",
        label="Document Splitting",
        description=(
            "Detects multi-document PDFs by scoring each page against known Document "
            "Classes.  When a page-class transition is found the PDF is split into "
            "logical segments, each processed as an independent document.  "
            "Disable to treat every uploaded PDF as a single document regardless of content."
        ),
        skippable=True,
        default_class="app.agents.splitting.SplittingAgent",
    ),
    PipelineStageInfo(
        key="deduplication",
        label="Deduplication",
        description=(
            "Fingerprints the document with SimHash and checks for exact, content, "
            "and near-duplicates against all previously seen documents.  "
            "Disable to re-process every upload regardless of prior submissions."
        ),
        skippable=True,
        default_class="app.agents.deduplication.DeduplicationAgent",
    ),
    PipelineStageInfo(
        key="classification",
        label="Classification",
        description=(
            "Matches the document against known Document Classes using keyword "
            "fingerprinting; falls back to OCR for image-only PDFs.  "
            "Cannot be disabled — classification is required for all downstream stages."
        ),
        skippable=False,
        default_class="app.agents.classification.ClassificationAgent",
    ),
    PipelineStageInfo(
        key="extraction",
        label="Extraction",
        description=(
            "Runs 3-tier extraction (text layer → OCR → AI vision) to pull "
            "structured fields.  Uses variant schema when available.  "
            "Cannot be disabled — extraction is the core pipeline output."
        ),
        skippable=False,
        default_class="app.agents.extraction.ExtractionAgent",
    ),
    PipelineStageInfo(
        key="variant_discovery",
        label="Variant Discovery",
        description=(
            "Identifies the document variant (issuer + field fingerprint) after "
            "extraction and links the document to the correct variant node.  "
            "Disable to skip variant tracking and always use class-level schemas."
        ),
        skippable=True,
        default_class="app.agents.variant_discovery.VariantDiscoveryAgent",
    ),
    PipelineStageInfo(
        key="signature",
        label="Signature Detection",
        description=(
            "Detects whether a document has been signed.  First run per variant uses "
            "Claude Haiku Vision; after 3 confirmed detections the agent uses the "
            "learned page/region, skipping the API call entirely.  "
            "Also gated per document type via DocumentTypeProfile.check_signature."
        ),
        skippable=True,
        default_class="app.agents.signature.SignatureAgent",
    ),
    PipelineStageInfo(
        key="address",
        label="Address Parsing",
        description=(
            "Parses address fields into structured JSON using Claude Haiku, then "
            "verifies them — UK postcodes via postcodes.io (free), international "
            "addresses via Google Maps Geocoding.  "
            "Disable if address verification is not needed."
        ),
        skippable=True,
        default_class="app.agents.address.AddressAgent",
    ),
    PipelineStageInfo(
        key="validation",
        label="Validation",
        description=(
            "Applies per-class business rules and NIGO conditions against "
            "extracted fields; fires the Instruction Engine.  "
            "Disable to pass all documents through without business-rule checks."
        ),
        skippable=True,
        default_class="app.agents.validation.ValidationAgent",
    ),
    PipelineStageInfo(
        key="governance",
        label="Governance (AI Review)",
        description=(
            "AI review of extraction failures — reclassifies wrong-class docs, "
            "flags genuinely new document types, or routes to human review.  "
            "Disable to route all extraction failures directly to NEEDS_REVIEW."
        ),
        skippable=True,
        default_class="app.agents.governance.GovernanceAgent",
    ),
    PipelineStageInfo(
        key="linking",
        label="Linking",
        description=(
            "Links the validated document to a ShipmentRecord by reference keys "
            "(PO number, AWB, bill of lading, etc.).  "
            "Disabling also auto-disables Matching (which requires a shipment)."
        ),
        skippable=True,
        default_class="app.agents.linking.LinkingAgent",
    ),
    PipelineStageInfo(
        key="matching",
        label="Matching",
        description=(
            "Runs three-way match (PO ↔ invoice ↔ receipt) for PROCESS-treatment "
            "documents.  Skipped automatically for STORE-treatment docs and when "
            "Linking is disabled."
        ),
        skippable=True,
        default_class="app.agents.matching.MatchingAgent",
    ),
    PipelineStageInfo(
        key="posting",
        label="ERP Posting",
        description=(
            "Posts the validated, matched document to the configured ERP system "
            "(Business Central, Greentree, SAP, etc.)."
        ),
        skippable=True,
        default_class="app.agents.posting.PostingAgent",
    ),
    PipelineStageInfo(
        key="filing",
        label="Filing",
        description=(
            "Files the document to the configured document library "
            "(SharePoint, Azure Blob, local disk, etc.)."
        ),
        skippable=True,
        default_class="app.agents.filing.FilingAgent",
    ),
    PipelineStageInfo(
        key="notifying",
        label="Notifying",
        description=(
            "Dispatches document.completed webhook events and email notifications "
            "on successful processing."
        ),
        skippable=True,
        default_class="app.agents.notifying.NotifyingAgent",
    ),
]

# Fast lookup by key
PIPELINE_STAGE_MAP: dict[str, PipelineStageInfo] = {s.key: s for s in PIPELINE_STAGES}


# ─────────────────────────────────────────────────────────────────────────────
#  Default registrations
# ─────────────────────────────────────────────────────────────────────────────

def _register_defaults() -> None:
    """
    Import and register all default pipeline agent classes.

    Called once at module import time.  Any class already registered via
    @register_pipeline_agent (i.e. a global override) is left untouched.
    """
    for stage_info in PIPELINE_STAGES:
        if stage_info.key in PIPELINE_AGENT_REGISTRY:
            continue  # already overridden by a decorator
        try:
            module_path, cls_name = stage_info.default_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)
            PIPELINE_AGENT_REGISTRY[stage_info.key] = cls
        except Exception as exc:
            logger.warning(
                "Failed to register default pipeline agent for stage %r: %s",
                stage_info.key, exc,
            )


_register_defaults()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline_agent_class(stage: str, override: Optional[str] = None) -> type:
    """
    Return the agent class to use for *stage*.

    Parameters
    ----------
    stage:
        Lowercase stage key, e.g. ``"extraction"``.
    override:
        Optional fully-qualified class path that takes precedence over the
        registry, e.g. ``"app.agents.extraction_freight.FreightExtractionAgent"``.
        Loaded dynamically via importlib so no static import is required.

    Raises
    ------
    KeyError
        If *stage* is not in the registry and no *override* is given.
    ImportError / AttributeError
        If *override* points to a non-existent module or class.
    """
    if override:
        module_path, cls_name = override.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    if stage not in PIPELINE_AGENT_REGISTRY:
        raise KeyError(
            f"No pipeline agent registered for stage {stage!r}. "
            f"Known stages: {sorted(PIPELINE_AGENT_REGISTRY)}"
        )
    return PIPELINE_AGENT_REGISTRY[stage]


def list_registered_stages() -> list[dict]:
    """
    Return a list of dicts describing all registered stages.  Useful for the
    admin UI and health-check endpoints.

    Each dict has: key, label, description, skippable, default_class, active_class.
    """
    result = []
    for stage in PIPELINE_STAGES:
        cls = PIPELINE_AGENT_REGISTRY.get(stage.key)
        result.append({
            "key":           stage.key,
            "label":         stage.label,
            "description":   stage.description,
            "skippable":     stage.skippable,
            "default_class": stage.default_class,
            "active_class":  f"{cls.__module__}.{cls.__qualname__}" if cls else None,
        })
    return result
