"""
Pipeline runner — orchestrates the full agent chain for a submitted document.

Runs as a FastAPI BackgroundTask after the submit endpoint returns.

Stage routing
─────────────
  PROCESS treatment (POs, invoices, AWBs, etc.):
    DEDUP → CLASSIFY → VARIANT → EXTRACT → VALIDATE
    → LINK → MATCH → POST → FILE → NOTIFY → COMPLETED

  STORE treatment (certs, FTA, RFQ, remittances, etc.):
    DEDUP → CLASSIFY → VARIANT → EXTRACT → VALIDATE
    → LINK → FILE → NOTIFY → COMPLETED  (skip MATCHING and POSTING)

  skip_stages override (per-document, set at submit time):
    Caller can pass e.g. skip_stages=["MATCHING","POSTING"] to force
    STORE-style routing even on a PROCESS-treatment document.

Plug-in architecture
────────────────────
  Every agent is loaded dynamically from the Pipeline Agent Registry
  (app.agents.registry).  Per-client overrides are stored in
  ClientAgentConfig.agent_overrides_json; per-client disabled stages in
  ClientAgentConfig.disabled_stages_json.

  Passing client_id to run_pipeline or run_pipeline_from loads the client's
  config and assembles a custom pipeline for that customer.  Omitting
  client_id (or passing None) uses all defaults — fully backward-compatible.

  To swap in a custom agent for a customer without touching global defaults:
    1. Write your subclass, e.g. app/agents/extraction_freight.py
    2. In the DB:  INSERT INTO client_agent_configs (client_id, agent_overrides_json)
                   VALUES ('cp_abc123', '{"extraction": "app.agents.extraction_freight.FreightExtractionAgent"}')
    3. Pass client_id when calling run_pipeline — done.

In production: replace with Azure Service Bus message processing where each
agent listens on its own queue and state transitions publish events to the
next agent's queue. The Recovery Agent subscribes to the dead-letter queue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from app.agents.registry import get_pipeline_agent_class
from app.agents.classification import _extract_pdf_text
from app.agents.instruction_runner import evaluate_instructions
from app.database import SessionLocal
from app.models.document import Document
from app.models.shipment import ShipmentRecord
from app.pipeline.states import PipelineState


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run pipeline configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _PipelineCfg:
    """
    Resolved pipeline configuration for a single run.

    Loaded from ClientAgentConfig for the given client_id, or populated
    with empty defaults when client_id is None (no per-client overrides).
    """
    disabled_stages: set[str] = field(default_factory=set)
    # {lowercase stage key -> dotted class path}
    agent_overrides: dict[str, str] = field(default_factory=dict)
    # {lowercase stage key -> {param: value}}
    stage_params: dict[str, dict] = field(default_factory=dict)


def _load_pipeline_cfg(client_id: Optional[str], db) -> _PipelineCfg:
    """
    Load ClientAgentConfig for *client_id* from the DB and return a
    _PipelineCfg.  Returns an empty (all-defaults) config when client_id
    is None or no record exists.
    """
    if not client_id:
        return _PipelineCfg()

    try:
        from app.models.client import ClientAgentConfig
        cac = db.query(ClientAgentConfig).filter(
            ClientAgentConfig.client_id == client_id
        ).first()
        if not cac:
            return _PipelineCfg()
        return _PipelineCfg(
            disabled_stages=cac.disabled_stages(),
            agent_overrides=cac.agent_overrides(),
            stage_params=cac.stage_params(),
        )
    except Exception:
        return _PipelineCfg()


def _make_agent(stage: str, pcfg: _PipelineCfg, db):
    """
    Instantiate the agent for *stage*, applying any per-client override
    and passing the stage's config params via BaseAgent.config.
    """
    override = pcfg.agent_overrides.get(stage)
    params = pcfg.stage_params.get(stage, {})
    cls = get_pipeline_agent_class(stage, override=override)
    return cls(db, config=params)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    document_id: str,
    pdf_bytes: bytes,
    client_id: Optional[str] = None,
) -> None:
    """
    Full pipeline from RECEIVED -> COMPLETED (or terminal error state).

    Parameters
    ----------
    document_id:
        Primary key of the Document row to process.
    pdf_bytes:
        Raw PDF content passed directly to agents that need the binary.
    client_id:
        Optional ClientProfile.id.  When provided, any ClientAgentConfig for
        that client is loaded and used to override default agent classes and
        disable unwanted stages.
    """
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return

        # Load per-client pipeline configuration
        pcfg = _load_pipeline_cfg(client_id, db)

        # Determine which stages to skip (doc-level + client-level)
        skipped = set(doc.skip_stages or []) | pcfg.disabled_stages

        # Track whether this doc is a near-duplicate so we can diff after extraction
        near_dup_original_id: str | None = None

        # -- Stage 0.pre: Document Splitting ----------------------------------
        # Must run before deduplication so each child segment gets its own
        # independent pipeline run (including dedup, classification, extraction).
        if "SPLITTING" not in skipped:
            try:
                from app.agents.splitting import SplittingAgent
                splitter = SplittingAgent(db)
                split_result = splitter.run(doc, pdf_bytes)
                if split_result and split_result.child_ids:
                    _log_event(
                        db, document_id, PipelineState.COMPLETED, "SplittingAgent",
                        f"Multi-document PDF split into {len(split_result.child_ids)} segment(s): "
                        + ", ".join(split_result.child_ids) + ". "
                        "Each segment is being processed independently.",
                    )
                    doc.status = PipelineState.COMPLETED
                    db.commit()
                    return
            except Exception as _split_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "doc %s: SplittingAgent non-fatal error: %s", document_id, _split_exc
                )
                # Continue with normal single-document pipeline

        # -- Stage 0: Deduplication -------------------------------------------
        if "DEDUPLICATION" not in skipped:
            dedup_agent = _make_agent("deduplication", pcfg, db)
            dedup_agent.transition(
                doc,
                PipelineState.DEDUPLICATING,
                "Computing content fingerprint and checking for duplicates.",
            )

            dedup_result = dedup_agent.deduplicate(doc, pdf_bytes)
            db.commit()

            if dedup_result.is_duplicate:
                dup_detail = (
                    f"{dedup_result.duplicate_type} detected. "
                    f"Original document: {dedup_result.original_document_id}. "
                    f"SimHash Hamming distance: {dedup_result.hamming_distance}."
                )

                if dedup_result.duplicate_type in ("EXACT_DUPLICATE", "CONTENT_DUPLICATE"):
                    # Hard stop - no value in re-processing an exact or content clone
                    dedup_agent.transition(doc, PipelineState[dedup_result.duplicate_type], dup_detail)
                    return

                # NEAR_DUPLICATE - continue to extraction for a field-level diff.
                # We log the detection event WITHOUT changing status (pipeline continues).
                near_dup_original_id = dedup_result.original_document_id
                _log_event(
                    db, document_id, PipelineState.DEDUPLICATING, "DeduplicationAgent",
                    dup_detail + " Near-duplicate: continuing to extraction for field-level diff analysis.",
                )

        # -- Stage 1: Classification ------------------------------------------
        classification_agent = _make_agent("classification", pcfg, db)
        classification_agent.transition(
            doc,
            PipelineState.CLASSIFYING,
            "Fingerprinting document against Knowledge Base.",
        )

        class_id, classification_confidence = classification_agent.classify(doc, pdf_bytes)

        if not class_id:
            # No keyword match at all - could be a pure scan or a genuinely new type.
            # Run the GovernanceAgent: it may recognise it from document text (OCR) or
            # flag it as CANDIDATE_NEW_CLASS rather than silently parking it UNCLASSIFIED.
            _run_governance(
                db=db,
                doc=doc,
                pdf_bytes=pdf_bytes,
                class_id=None,
                classification_confidence=0.0,
                extracted_fields=[],
                extraction_agent=None,
                classification_agent=classification_agent,
                failure_reason=(
                    "Document did not match any known Document Class. "
                    "Likely a scanned PDF, pure-image document, or a genuinely new type."
                ),
                pcfg=pcfg,
                skipped=skipped,
            )
            return

        doc.document_class_id = class_id
        doc.classification_confidence = classification_confidence
        db.commit()
        db.refresh(doc)

        class_name = doc.document_class.name if doc.document_class else class_id
        treatment   = doc.document_class.treatment if doc.document_class else "PROCESS"

        # STORE treatment always skips matching and posting (unless explicitly overridden)
        if treatment == "STORE":
            skipped.update(["MATCHING", "POSTING"])

        # -- Stage 2: Extraction ----------------------------------------------
        # Variant discovery runs AFTER extraction (needs extracted issuer + fields).
        # We log the EXTRACTING transition with "ZERO_SHOT" stage until we know the
        # variant; the completion event below updates with the real variant info.
        classification_agent.transition(
            doc,
            PipelineState.EXTRACTING,
            f"Classified as '{class_name}'. "
            "Running ZERO_SHOT free-form discovery to identify variant and fields.",
        )

        extraction_agent = _make_agent("extraction", pcfg, db)
        fields = extraction_agent.extract(doc, pdf_bytes)

        if not fields:
            # -- Governance check --------------------------------------------
            _run_governance(
                db=db,
                doc=doc,
                pdf_bytes=pdf_bytes,
                class_id=class_id,
                classification_confidence=classification_confidence,
                extracted_fields=[],
                extraction_agent=extraction_agent,
                classification_agent=classification_agent,
                failure_reason=(
                    f"All 3 extraction tiers failed proofreading for class {class_id}. "
                    "Tier 1 (text layer), Tier 2 (OCR), and Tier 3 (AI vision) all produced "
                    "insufficient results."
                ),
                pcfg=pcfg,
                skipped=skipped,
            )
            return

        # -- Stage 2b: Variant Discovery (post-extraction) --------------------
        # Now that fields are extracted we can identify the issuer and compute the
        # field fingerprint, then match/create the variant node.
        db.refresh(doc)   # pick up extracted_fields relationship
        variant_agent = _make_agent("variant_discovery", pcfg, db)
        variant_id    = variant_agent.discover(doc, pdf_bytes=pdf_bytes)
        doc.variant_id = variant_id
        db.commit()
        db.refresh(doc)

        stage       = doc.variant.learning_stage if doc.variant else "ZERO_SHOT"
        variant_key = doc.variant_key or "unknown"
        variant_lbl = doc.variant.variant_label if doc.variant else "unknown"

        _log_event(
            db, document_id, PipelineState.EXTRACTING, "VariantDiscoveryAgent",
            f"Variant identified: '{variant_lbl}' [{variant_key}] stage={stage}. "
            f"Issuer: {doc.variant.issuer_slug if doc.variant else 'unknown'}. "
            f"Field fingerprint: {doc.variant.field_fingerprint if doc.variant else 'n/a'}.",
        )

        # -- Stage 2b-i: StructuralProfileAgent — update variant structural profile --
        # Runs on every document so the profile continuously learns page count,
        # headings, and footer text. Non-fatal — never blocks the pipeline.
        if doc.variant_id:
            try:
                from app.agents.structural_profile import StructuralProfileAgent as _SPA
                _spa = _SPA(db)
                _spa.update_variant(doc, doc.variant_id, pdf_bytes)
            except Exception as _spa_exc:
                logger.warning("doc %s: StructuralProfileAgent non-fatal: %s", document_id, _spa_exc)

        # -- Stage 2b-iii: FormatAgent — update format_hints as more data arrives ---
        # Run opportunistically on every 5th document for a LEARNED/OPTIMISED variant
        # so hints continuously improve. Non-fatal — failure never blocks the pipeline.
        if stage in ("LEARNED", "OPTIMISED") and doc.variant and doc.variant.field_schema_json:
            _doc_count = doc.variant.confirmed_instance_count or 0
            if _doc_count > 0 and _doc_count % 5 == 0:
                try:
                    from app.agents.format_agent import FormatAgent as _FormatAgent
                    _fa = _FormatAgent(db)
                    _fa_result = _fa.run_for_variant(doc.variant_id)
                    _fa_updated = _fa_result.get("fields_updated", 0)
                    if _fa_updated:
                        _log_event(
                            db, document_id, PipelineState.EXTRACTING, "FormatAgent",
                            f"Format analysis updated {_fa_updated} field hints for variant "
                            f"'{variant_lbl}' (doc #{_doc_count}).",
                        )
                except Exception as _fa_exc:
                    logger.warning("doc %s: FormatAgent non-fatal: %s", document_id, _fa_exc)

        # -- Stage 2b-iv: If variant is LEARNED, re-extract with known schema ----
        # The first extraction was free-form (ZERO_SHOT discovery). Now that we
        # know the variant is already LEARNED, run a targeted re-extraction using
        # the confirmed schema so field names and values are normalised.
        if stage in ("LEARNED", "OPTIMISED") and doc.variant and doc.variant.field_schema_json:
            _log_event(
                db, document_id, PipelineState.EXTRACTING, "ExtractionAgent",
                f"Variant '{variant_lbl}' is {stage}. Re-extracting with confirmed schema "
                f"({len(json.loads(doc.variant.field_schema_json))} fields).",
            )
            retry_fields = extraction_agent.extract(doc, pdf_bytes)
            if retry_fields:
                fields = retry_fields

        # -- Stage 3b: Near-dup post-extraction diff --------------------------
        # Now that extraction is done, compute the field-level diff against the
        # original document. If fields have changed, route to NEEDS_REVIEW.
        if near_dup_original_id:
            from app.agents.deduplication import _build_field_diff
            original_doc = db.query(Document).filter(Document.id == near_dup_original_id).first()
            if original_doc:
                field_diff = _build_field_diff(doc, original_doc, db)
                if field_diff:
                    changed_fields = [d["field_name"] for d in field_diff]
                    diff_lines = " | ".join(
                        f"{d['field_name']}: {d['original_value']!r} -> {d['incoming_value']!r}"
                        for d in field_diff[:6]
                    )
                    diff_summary = (
                        f"NEAR_DUPLICATE - {len(field_diff)} field(s) changed vs "
                        f"original {near_dup_original_id}. "
                        f"Changed: {', '.join(changed_fields[:10])}. "
                        f"Detail: {diff_lines}"
                    )
                    extraction_agent.needs_review(doc, diff_summary)
                    return
                else:
                    # Post-extraction diff is empty: same content as original -> full content dup
                    dedup_agent.transition(
                        doc,
                        PipelineState.CONTENT_DUPLICATE,
                        f"Post-extraction diff: no field changes found vs {near_dup_original_id}. "
                        "Reclassified as CONTENT_DUPLICATE.",
                    )
                    return

        # -- Stage 3c: AI Review (second-opinion pass on low-confidence fields)
        low_confidence = [f for f in fields if f.confidence < 0.85]
        if low_confidence and stage in ("ZERO_SHOT", "LEARNING"):
            extraction_agent.transition(
                doc,
                PipelineState.AI_REVIEWING,
                f"{len(low_confidence)} field(s) below 0.85 confidence threshold. "
                "Running second-opinion pass with independent AI model.",
            )
            # Actually run the second pass - updates confidence in-place
            fields = extraction_agent.second_opinion(doc, fields)
            # Recount low-confidence after boost
            still_low = [f for f in fields if f.confidence < 0.85]
            _log_event(
                db, document_id, PipelineState.AI_REVIEWING, "ExtractionAgent",
                f"Second-opinion complete. "
                f"{len(low_confidence) - len(still_low)} field(s) promoted above 0.85 threshold. "
                f"{len(still_low)} field(s) remain below threshold."
                + (f" Low fields: {', '.join(f.field_name for f in still_low[:5])}." if still_low else ""),
            )

        avg_confidence = round(sum(f.confidence for f in fields) / len(fields), 4)

        # -- Stage 3d: Signature Detection (opt-in per document type) ---------
        # Runs after variant discovery so the agent can load/update the
        # variant's signature_profile_json (the learned location cache).
        # Gated on DocumentTypeProfile.check_signature — skipped by default.
        try:
            from app.models.client import DocumentTypeProfile
            _dtp_for_sig = db.query(DocumentTypeProfile).filter_by(
                client_id="cp_001", document_class_id=doc.document_class_id
            ).first() if doc.document_class_id else None

            if _dtp_for_sig and getattr(_dtp_for_sig, "check_signature", False) and "SIGNATURE" not in skipped:
                sig_agent = _make_agent("signature", pcfg, db)
                sig_agent.run(doc, pdf_bytes)
                sig_status = "signed" if doc.is_signed else "unsigned"
                sig_conf   = f"{(doc.signature_confidence or 0):.0%}"
                _log_event(
                    db, document_id, PipelineState.EXTRACTING, "SignatureAgent",
                    f"Signature check: {sig_status} ({sig_conf} confidence).",
                )
        except Exception as _sig_exc:
            logger.warning("doc %s: signature detection non-fatal error: %s", document_id, _sig_exc)

        # -- Stage 3e: AddressAgent — parse and verify address fields ----------
        if "ADDRESS" not in skipped:
            try:
                from app.agents.address import AddressAgent as _AddressAgent
                _aa = _AddressAgent(db)
                _aa_count = _aa.run(doc)
                if _aa_count:
                    _log_event(
                        db, document_id, PipelineState.EXTRACTING, "AddressAgent",
                        f"Parsed and verified {_aa_count} address field(s).",
                    )
            except Exception as _aa_exc:
                logger.warning("doc %s: AddressAgent non-fatal: %s", document_id, _aa_exc)

        # -- Stage 4: Validation ----------------------------------------------
        if "VALIDATION" not in skipped:
            extraction_agent.transition(
                doc,
                PipelineState.VALIDATING,
                f"Extraction complete. {len(fields)} fields extracted. "
                f"Avg confidence: {avg_confidence:.1%}. "
                "Running business rule validation.",
            )

            validation_agent = _make_agent("validation", pcfg, db)
            result = validation_agent.validate(doc)

            for w in result.warnings:
                validation_agent.transition(doc, PipelineState.VALIDATING, f"WARNING: {w}")

            if not result.is_valid:
                nigo_detail = (
                    f"NIGO - {len(result.nigo_conditions)} condition(s) failed. "
                    + " | ".join(result.nigo_conditions)
                )
                validation_agent.needs_review(doc, nigo_detail)
                return

            # -- Stage 4b: Instruction Engine ---------------------------------
            # Evaluate per-class and global rules. Must run AFTER validation (IGO)
            # so extracted fields are guaranteed to be populated and business-valid.
            fired = evaluate_instructions(db, doc)
            if fired:
                from datetime import datetime
                from app.models.document import PipelineEvent

                for fi in fired:
                    _desc = fi.description or fi.action

                    if fi.action == "REQUIRE_APPROVAL":
                        reason = (
                            f"Instruction #{fi.instruction_id} ({_desc}) requires manual approval. "
                            f"Condition: {fi.condition_summary}."
                        )
                        validation_agent.needs_review(doc, reason)
                        return

                    elif fi.action == "SKIP_POSTING":
                        skipped.add("POSTING")
                        _log_event(db, document_id, PipelineState.VALIDATING, "InstructionRunner",
                                   f"Instruction #{fi.instruction_id} ({_desc}): SKIP_POSTING applied. "
                                   f"Condition: {fi.condition_summary}.")

                    elif fi.action == "SKIP_MATCHING":
                        skipped.add("MATCHING")
                        _log_event(db, document_id, PipelineState.VALIDATING, "InstructionRunner",
                                   f"Instruction #{fi.instruction_id} ({_desc}): SKIP_MATCHING applied. "
                                   f"Condition: {fi.condition_summary}.")

                    elif fi.action == "FLAG_WARNING":
                        note = fi.action_value or "(no note)"
                        _log_event(db, document_id, PipelineState.VALIDATING, "InstructionRunner",
                                   f"WARNING - Instruction #{fi.instruction_id} ({_desc}): {note}. "
                                   f"Condition: {fi.condition_summary}.")

                    elif fi.action == "NOTIFY_EMAIL":
                        target = fi.action_value or ""
                        _fire_instruction_notify(document_id, fi, target)
                        _log_event(db, document_id, PipelineState.VALIDATING, "InstructionRunner",
                                   f"Instruction #{fi.instruction_id} ({_desc}): NOTIFY_EMAIL -> {target}. "
                                   f"Condition: {fi.condition_summary}.")
        else:
            _log_event(db, document_id, PipelineState.VALIDATING, "PipelineRunner",
                       "Validation skipped (disabled in pipeline configuration).")

        # -- Stage 5: Linking -------------------------------------------------
        link_result = None
        if "LINKING" not in skipped:
            linking_agent = _make_agent("linking", pcfg, db)
            linking_agent.transition(
                doc,
                PipelineState.LINKING,
                "Validation passed (IGO). Querying shipment graph by reference keys.",
            )

            link_result = linking_agent.link(doc)
            db.refresh(doc)

            linking_agent.transition(
                doc,
                PipelineState.LINKING,
                f"{'Created new' if link_result.is_new_shipment else 'Joined existing'} "
                f"ShipmentRecord {link_result.shipment_id}. "
                f"Reference key: {link_result.reference_key}. "
                f"Shipment now has {link_result.document_count} document(s). "
                f"All keys: {', '.join(link_result.all_keys[:5])}.",
            )
        else:
            # Matching requires a shipment — auto-disable it when linking is off.
            skipped.add("MATCHING")
            _log_event(db, document_id, PipelineState.LINKING, "PipelineRunner",
                       "Linking skipped (disabled in pipeline configuration). "
                       "Matching also disabled (requires a shipment).")

        # -- Stage 6: Matching ------------------------------------------------
        erp_reference = None
        match_outcome = None

        if "MATCHING" not in skipped:
            matching_agent = _make_agent("matching", pcfg, db)
            matching_agent.transition(
                doc,
                PipelineState.MATCHING,
                f"Running three-way match for shipment {link_result.shipment_id}.",
            )

            match_result = matching_agent.match(doc)
            match_outcome = match_result.outcome

            # Persist match result on ShipmentRecord
            if doc.shipment_id:
                shipment = db.query(ShipmentRecord).filter(
                    ShipmentRecord.id == doc.shipment_id
                ).first()
                if shipment:
                    shipment.match_result = match_outcome
                    shipment.match_detail = {
                        "summary": match_result.summary,
                        "checks": [
                            {"name": c.name, "status": c.status, "detail": c.detail}
                            for c in match_result.checks
                        ],
                    }
                    db.commit()

            if match_outcome == "FAIL":
                matching_agent.needs_review(
                    doc,
                    f"NIGO - Three-way match FAIL. {match_result.summary}",
                )
                return

            matching_agent.transition(
                doc,
                PipelineState.MATCHING,
                match_result.summary,
            )

        # -- Stage 7: Posting -------------------------------------------------
        if "POSTING" not in skipped:
            posting_agent = _make_agent("posting", pcfg, db)
            posting_agent.transition(
                doc,
                PipelineState.POSTING,
                f"Match {'PASS_PARTIAL' if match_outcome == 'PASS_PARTIAL' else 'PASS'}. "
                "Submitting to ERP (Business Central).",
            )

            post_result = posting_agent.post(doc)
            erp_reference = post_result.erp_reference

            if not post_result.success:
                posting_agent.needs_review(doc, f"ERP posting failed. {post_result.detail}")
                return

            posting_agent.transition(
                doc,
                PipelineState.POSTING,
                post_result.detail,
            )

        # -- Stage 8: Filing --------------------------------------------------
        file_result = None
        if "FILING" not in skipped:
            filing_agent = _make_agent("filing", pcfg, db)
            filing_agent.transition(
                doc,
                PipelineState.FILING,
                "Filing document to SharePoint document library.",
            )
            file_result = filing_agent.file(doc)
            filing_agent.transition(doc, PipelineState.FILING, file_result.detail)
        else:
            _log_event(db, document_id, PipelineState.FILING, "PipelineRunner",
                       "Filing skipped (disabled in pipeline configuration).")

        # -- Stage 9: Notification --------------------------------------------
        if "NOTIFYING" not in skipped:
            notify_agent = _make_agent("notifying", pcfg, db)
            notify_agent.transition(
                doc,
                PipelineState.NOTIFYING,
                "Dispatching document.completed webhook event.",
            )
            notify_result = notify_agent.notify(
                doc,
                sharepoint_path=file_result.sharepoint_path if file_result else None,
                erp_reference=erp_reference,
            )
            notify_agent.transition(doc, PipelineState.NOTIFYING, notify_result.detail)
        else:
            _log_event(db, document_id, PipelineState.NOTIFYING, "PipelineRunner",
                       "Notification skipped (disabled in pipeline configuration).")

        # -- Terminal: COMPLETED ----------------------------------------------
        skipped_note = f" Skipped stages: {', '.join(sorted(skipped))}." if skipped else ""
        _complete_document(
            db, doc,
            f"Document processed successfully. "
            f"Class: {class_name} | Variant: {variant_lbl} [{stage}] | "
            f"Fields: {len(fields)} | Avg confidence: {avg_confidence:.1%} | "
            f"Treatment: {treatment}"
            + (f" | SharePoint: {file_result.sharepoint_path}" if file_result else "")
            + (f" | ERP ref: {erp_reference}" if erp_reference else "")
            + "."
            + skipped_note,
        )

        # Update ShipmentRecord to COMPLETE if all docs in shipment are done
        _maybe_close_shipment(db, doc.shipment_id)

    except Exception as exc:
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                from datetime import datetime
                from app.models.document import PipelineEvent
                doc.status = PipelineState.FAILED
                doc.updated_at = datetime.utcnow()
                event = PipelineEvent(
                    document_id=document_id,
                    state=PipelineState.FAILED,
                    agent="RecoveryAgent",
                    detail=f"Unhandled pipeline exception: {type(exc).__name__}: {exc}",
                )
                db.add(event)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _maybe_close_shipment(db, shipment_id: str | None) -> None:
    """Mark shipment COMPLETE if every linked document is in a terminal state."""
    if not shipment_id:
        return
    from app.models.document import Document as Doc
    shipment = db.query(ShipmentRecord).filter(ShipmentRecord.id == shipment_id).first()
    if not shipment:
        return
    terminal = {
        PipelineState.COMPLETED, PipelineState.EXACT_DUPLICATE,
        PipelineState.NEEDS_REVIEW, PipelineState.CANDIDATE_NEW_CLASS, PipelineState.FAILED,
    }
    docs = db.query(Doc).filter(Doc.id.in_(shipment.document_ids or [])).all()
    if docs and all(d.status in terminal for d in docs):
        shipment.status = "COMPLETE"
        db.commit()


def run_pipeline_from(
    document_id: str,
    from_stage: str,
    client_id: Optional[str] = None,
) -> None:
    """
    Re-enter the pipeline at a specific stage after human approval.

    Called by the review approval endpoint when a NEEDS_REVIEW document
    is approved to resume at MATCHING, POSTING, etc.

    Stages that need the original PDF bytes (DEDUPLICATING, CLASSIFYING)
    cannot be re-entered this way - those require a full retry with bytes.
    """
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return

        # Load per-client pipeline config
        pcfg = _load_pipeline_cfg(client_id, db)

        skipped = set(doc.skip_stages or []) | pcfg.disabled_stages
        treatment = doc.document_class.treatment if doc.document_class else "PROCESS"
        if treatment == "STORE":
            skipped.update(["MATCHING", "POSTING"])

        class_name = doc.document_class.name if doc.document_class else (doc.document_class_id or "unknown")

        # Re-run from the approved stage onwards
        stage_order = ["MATCHING", "POSTING", "FILING", "NOTIFYING", "COMPLETED"]
        start_idx = stage_order.index(from_stage) if from_stage in stage_order else 0

        erp_reference = None
        file_result = None
        all_fields = db.query(
            __import__('app.models.extracted_field', fromlist=['ExtractedField']).ExtractedField
        ).filter_by(document_id=document_id).all()
        avg_confidence = round(
            sum(f.confidence for f in all_fields) / len(all_fields), 4
        ) if all_fields else 0.0

        if start_idx <= 0 and "MATCHING" not in skipped:
            matching_agent = _make_agent("matching", pcfg, db)
            matching_agent.transition(
                doc, PipelineState.MATCHING,
                "Resuming three-way match after human approval.",
            )
            match_result = matching_agent.match(doc)
            match_outcome = match_result.outcome
            if doc.shipment_id:
                from app.models.shipment import ShipmentRecord as SR
                shp = db.query(SR).filter(SR.id == doc.shipment_id).first()
                if shp:
                    shp.match_result = match_outcome
                    shp.match_detail = {
                        "summary": match_result.summary,
                        "checks": [
                            {"name": c.name, "status": c.status, "detail": c.detail}
                            for c in match_result.checks
                        ],
                    }
                    db.commit()
            if match_outcome == "FAIL":
                matching_agent.needs_review(
                    doc, f"NIGO - Match still FAIL after approval. {match_result.summary}"
                )
                return
            matching_agent.transition(doc, PipelineState.MATCHING, match_result.summary)

        if start_idx <= 1 and "POSTING" not in skipped:
            posting_agent = _make_agent("posting", pcfg, db)
            posting_agent.transition(
                doc, PipelineState.POSTING,
                "ERP posting resumed after human approval.",
            )
            post_result = posting_agent.post(doc)
            erp_reference = post_result.erp_reference
            if not post_result.success:
                posting_agent.needs_review(doc, f"ERP posting failed. {post_result.detail}")
                return
            posting_agent.transition(doc, PipelineState.POSTING, post_result.detail)

        file_result = None
        if "FILING" not in skipped:
            filing_agent = _make_agent("filing", pcfg, db)
            filing_agent.transition(doc, PipelineState.FILING, "Filing document to SharePoint.")
            file_result = filing_agent.file(doc)
            filing_agent.transition(doc, PipelineState.FILING, file_result.detail)
        else:
            _log_event(db, document_id, PipelineState.FILING, "PipelineRunner",
                       "Filing skipped (disabled in pipeline configuration).")

        if "NOTIFYING" not in skipped:
            notify_agent = _make_agent("notifying", pcfg, db)
            notify_agent.transition(
                doc, PipelineState.NOTIFYING,
                "Dispatching document.completed webhook.",
            )
            notify_result = notify_agent.notify(
                doc,
                sharepoint_path=file_result.sharepoint_path if file_result else None,
                erp_reference=erp_reference,
            )
            notify_agent.transition(doc, PipelineState.NOTIFYING, notify_result.detail)
        else:
            _log_event(db, document_id, PipelineState.NOTIFYING, "PipelineRunner",
                       "Notification skipped (disabled in pipeline configuration).")

        _complete_document(
            db, doc,
            f"Document completed after human review approval. "
            f"Class: {class_name} | Fields: {len(all_fields)} | "
            f"Avg confidence: {avg_confidence:.1%}."
            + (f" ERP ref: {erp_reference}." if erp_reference else "")
            + (f" SharePoint: {file_result.sharepoint_path}." if file_result else ""),
        )
        _maybe_close_shipment(db, doc.shipment_id)

    except Exception as exc:
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                from datetime import datetime
                from app.models.document import PipelineEvent
                doc.status = PipelineState.FAILED
                doc.updated_at = datetime.utcnow()
                db.add(PipelineEvent(
                    document_id=document_id,
                    state=PipelineState.FAILED,
                    agent="RecoveryAgent",
                    detail=f"run_pipeline_from exception: {exc}",
                ))
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Governance helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_governance(
    db,
    doc: Document,
    pdf_bytes: bytes,
    class_id: Optional[str],
    classification_confidence: float,
    extracted_fields: list,
    extraction_agent,
    classification_agent,
    failure_reason: str,
    pcfg: _PipelineCfg,
    skipped: Optional[set] = None,
) -> None:
    """
    Run the GovernanceAgent after extraction failure.

    Branches:
      CORRECT     -> NEEDS_REVIEW  (extraction genuinely failed, human needed)
      WRONG_CLASS -> EXTRACTING    (reclassify to suggested class, retry once)
      NEW_TYPE    -> CANDIDATE_NEW_CLASS

    The function always terminates the pipeline branch by transitioning to
    one of the three terminal-or-retry states. After it returns, runner.py
    should `return` immediately.
    """
    # If GOVERNANCE is disabled, skip AI review and route straight to NEEDS_REVIEW.
    if skipped and "GOVERNANCE" in skipped:
        _set_needs_review(
            db, doc,
            f"Extraction failed: {failure_reason} "
            "(Governance AI review disabled in pipeline configuration).",
        )
        return

    # Extract text for the governance prompt (same extraction used in classification)
    pdf_text = _extract_pdf_text(pdf_bytes)

    # Transition to GOVERNING state
    _log_event(
        db, doc.id, PipelineState.GOVERNING, "GovernanceAgent",
        f"Extraction failed ({failure_reason}). "
        f"Classification confidence: {classification_confidence:.0%}. "
        "Invoking AI governance review.",
    )
    from datetime import datetime
    doc.status = PipelineState.GOVERNING
    doc.updated_at = datetime.utcnow()
    db.commit()

    governance_agent = _make_agent("governance", pcfg, db)
    result = governance_agent.review(
        doc=doc,
        pdf_text=pdf_text,
        assigned_class_id=class_id,
        classification_confidence=classification_confidence,
        extracted_field_names=[f.field_name for f in extracted_fields],
    )

    # Persist governance result
    doc.ai_governance_result = json.dumps(result.as_dict())
    db.commit()

    _log_event(
        db, doc.id, PipelineState.GOVERNING, "GovernanceAgent",
        result.as_event_detail(),
    )

    if result.verdict == "WRONG_CLASS" and result.suggested_class:
        # Reclassify and retry extraction with the suggested class - once only.
        # (If extraction fails again after reclassification, we go to NEEDS_REVIEW
        #  to avoid an infinite governance loop.)
        new_class_id = result.suggested_class

        # Update classification
        doc.document_class_id = new_class_id
        doc.document_class_override = new_class_id  # pin so re-classify won't change it
        doc.classification_confidence = result.confidence
        db.commit()
        db.refresh(doc)

        new_class_name = doc.document_class.name if doc.document_class else new_class_id
        _log_event(
            db, doc.id, PipelineState.EXTRACTING, "GovernanceAgent",
            f"Reclassified from {class_id} -> {new_class_id} ({new_class_name}). "
            "Retrying extraction with updated class.",
        )
        doc.status = PipelineState.EXTRACTING
        doc.updated_at = datetime.utcnow()
        db.commit()

        # Retry extraction once - use the (potentially overridden) extraction agent
        retry_agent = _make_agent("extraction", pcfg, db)
        retry_fields = retry_agent.extract(doc, pdf_bytes)

        if retry_fields:
            # Extraction succeeded after reclassification - resume normally.
            avg_conf = round(sum(f.confidence for f in retry_fields) / len(retry_fields), 4)
            _log_event(
                db, doc.id, PipelineState.VALIDATING, "GovernanceAgent",
                f"Extraction succeeded after reclassification. "
                f"{len(retry_fields)} fields extracted. Avg confidence: {avg_conf:.1%}. "
                "Routing to validation.",
            )
            validation_agent = _make_agent("validation", pcfg, db)
            result_v = validation_agent.validate(doc)
            if result_v.is_valid:
                doc.status = PipelineState.COMPLETED
                doc.updated_at = datetime.utcnow()
                db.commit()
                _log_event(
                    db, doc.id, PipelineState.COMPLETED, "GovernanceAgent",
                    f"Document reclassified and processed. "
                    f"Class: {new_class_name} | Fields: {len(retry_fields)} | "
                    f"Avg confidence: {avg_conf:.1%}.",
                )
            else:
                nigo = " | ".join(result_v.nigo_conditions)
                validation_agent.needs_review(
                    doc,
                    f"Reclassified to {new_class_id} but validation NIGO: {nigo}",
                )
        else:
            # Even with correct class, extraction failed - genuine difficulty
            _set_needs_review(
                db, doc,
                f"Extraction still failed after governance reclassification to {new_class_id}. "
                "Manual field entry required.",
            )

    elif result.verdict == "NEW_TYPE":
        # Flag as candidate for new document class
        doc.status = PipelineState.CANDIDATE_NEW_CLASS
        doc.suggested_class_name = result.suggested_class_name or "Unknown Document Type"
        doc.candidate_reason = (
            f"{result.reasoning} "
            f"Suggested description: {result.suggested_class_description or 'N/A'}. "
            f"Keywords: {', '.join(result.suggested_keywords[:10])}."
        )
        doc.updated_at = datetime.utcnow()
        db.commit()
        _log_event(
            db, doc.id, PipelineState.CANDIDATE_NEW_CLASS, "GovernanceAgent",
            f"Document flagged as probable new class: '{doc.suggested_class_name}'. "
            f"{result.reasoning[:200]}",
        )

    else:
        # CORRECT - extraction failed for non-classification reasons (scan quality, etc.)
        _set_needs_review(
            db, doc,
            f"GovernanceAgent confirmed class {class_id or '(unclassified)'} is correct. "
            f"{failure_reason} "
            f"Governance reasoning: {result.reasoning[:200]}. "
            "Manual field entry required.",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _complete_document(db, doc: Document, detail: str) -> None:
    """
    Transition a document to COMPLETED without requiring a specific agent instance.
    Used when NOTIFYING (or other late stages) may be disabled.
    """
    from datetime import datetime
    from app.models.document import PipelineEvent
    doc.status = PipelineState.COMPLETED
    doc.updated_at = datetime.utcnow()
    db.add(PipelineEvent(
        document_id=doc.id,
        state=PipelineState.COMPLETED,
        agent="PipelineRunner",
        detail=detail,
    ))
    db.commit()


def _set_needs_review(db, doc: Document, reason: str) -> None:
    """
    Transition a document to NEEDS_REVIEW without requiring a specific agent object.
    Used in _run_governance where the extraction_agent may be None.
    Also persists error_reason and fires failure notifications.
    """
    from datetime import datetime
    from app.models.document import PipelineEvent
    doc.status = PipelineState.NEEDS_REVIEW
    doc.error_reason = reason   # persist for dashboard display + notifications
    doc.updated_at = datetime.utcnow()
    event = PipelineEvent(
        document_id=doc.id,
        state=PipelineState.NEEDS_REVIEW,
        agent="GovernanceAgent",
        detail=reason,
    )
    db.add(event)
    db.commit()
    # Fire failure notifications best-effort — never block the pipeline.
    try:
        from app.agents.notifying import fire_failure_notifications
        fire_failure_notifications(db, doc)
    except Exception:
        pass


def _log_event(db, document_id: str, state, agent: str, detail: str) -> None:
    """Append a PipelineEvent without changing the document's status."""
    from datetime import datetime
    from app.models.document import PipelineEvent
    event = PipelineEvent(
        document_id=document_id,
        state=state,
        agent=agent,
        detail=detail,
    )
    db.add(event)
    db.commit()


def _fire_instruction_notify(document_id: str, fi, target: str) -> None:
    """
    Fire a lightweight webhook for a NOTIFY_EMAIL instruction.
    target is any URL or email address stored in action_value.
    If it looks like a URL (http/https), we POST JSON. Otherwise we log only.
    """
    if not target.startswith(("http://", "https://")):
        return  # email-only - in production, route through SMTP relay
    try:
        import httpx
        payload = {
            "event": "instruction.notify",
            "document_id": document_id,
            "instruction_id": fi.instruction_id,
            "description": fi.description,
            "condition_summary": fi.condition_summary,
        }
        httpx.post(target, json=payload, timeout=8,
                   headers={"User-Agent": "Dokr/1.0", "Content-Type": "application/json"})
    except Exception:
        pass  # never block the pipeline for a notification failure


def _model_for_stage(stage: str) -> str:
    """Human-readable extraction model description logged at EXTRACTING stage."""
    return {
        "ZERO_SHOT": "regex-tier1 -> regex-tier2-ocr -> claude-sonnet-4-6-vision (3-tier)",
        "LEARNING":  "regex-tier1 -> regex-tier2-ocr -> claude-sonnet-4-6-vision (3-tier)",
        "LEARNED":   "regex-tier1 (pattern-learned)",
        "OPTIMISED": "regex-tier1 (optimised patterns)",
    }.get(stage, "regex-tier1 -> regex-tier2-ocr -> claude-sonnet-4-6-vision (3-tier)")
