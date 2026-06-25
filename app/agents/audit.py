"""
Audit agents — on-demand intelligence tools that scan the accumulated document
history and surface actionable insights.

These never run inline with document processing. They are triggered explicitly
by an operator, an API call, or a schedule, and write their results to AgentRun.

Available agents
----------------
ExtractionQualityAuditAgent   — confidence, hit-rate, and correction-rate per variant
PageScoreAuditAgent           — page contribution analysis + cost savings from PageProfileAgent
FieldUsageAuditAgent          — schema fields that are never / rarely extracted
CostOptimisationReportAgent   — full cost picture: tokens consumed, saved, projected
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document, DocumentClass, DocumentVariant
from app.models.extracted_field import ExtractedField

logger = logging.getLogger(__name__)

COST_PER_TOKEN = 3.0 / 1_000_000   # Sonnet input pricing
AVG_TOKENS_PER_PAGE = 300           # conservative heuristic


# ─────────────────────────────────────────────────────────────────────────────
#  CATALOG — drives the dashboard UI
# ─────────────────────────────────────────────────────────────────────────────

AGENT_CATALOG: list[dict] = [
    {
        "name":         "extraction_quality_audit",
        "label":        "Extraction Quality Audit",
        "description":  (
            "Reviews the last N documents for extraction quality — confidence "
            "scores, field hit-rates, and correction rates per variant. Flags "
            "variants that need attention."
        ),
        "category":     "Quality",
        "category_color": "#7C84E8",
        "typical_duration": "5–15 seconds",
        "params": [
            {
                "key":     "last_n",
                "label":   "Documents to review",
                "type":    "number",
                "default": 20,
                "min":     5,
                "max":     200,
            },
        ],
    },
    {
        "name":         "page_score_audit",
        "label":        "Page Score Audit",
        "description":  (
            "Analyses page contribution scores across variants and surfaces "
            "additional pages that could be added to the skip list, reducing "
            "token consumption and API cost."
        ),
        "category":     "Cost Optimisation",
        "category_color": "#00C97A",
        "typical_duration": "2–5 seconds",
        "params": [],
    },
    {
        "name":         "field_usage_audit",
        "label":        "Field Usage Audit",
        "description":  (
            "Finds schema fields that are consistently empty or never extracted "
            "across your documents. Recommends pruning the schema to tighten "
            "prompts and improve accuracy."
        ),
        "category":     "Schema Health",
        "category_color": "#F5A623",
        "typical_duration": "5–10 seconds",
        "params": [
            {
                "key":     "last_n",
                "label":   "Documents to review",
                "type":    "number",
                "default": 50,
                "min":     10,
                "max":     500,
            },
        ],
    },
    {
        "name":         "cost_optimisation_report",
        "label":        "Cost Optimisation Report",
        "description":  (
            "Calculates total token usage, savings accumulated by PageProfileAgent, "
            "and projects future costs as the learning system matures. "
            "Quantifies the ROI of your AI pipeline."
        ),
        "category":     "Cost Optimisation",
        "category_color": "#00C97A",
        "typical_duration": "2–3 seconds",
        "params": [],
    },
]

# Name → catalog entry lookup
CATALOG_BY_NAME: dict[str, dict] = {a["name"]: a for a in AGENT_CATALOG}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _variant_label(v: DocumentVariant) -> str:
    return v.variant_label or v.issuer_slug or v.variant_key or f"Variant {v.id[:8]}"


def _safe_json(s: str | None) -> dict | list:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  ExtractionQualityAuditAgent
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionQualityAuditAgent(BaseAgent):
    name = "extraction_quality_audit"

    def run(self, params: dict) -> dict:
        last_n: int = int(params.get("last_n", 20))
        variant_id: str | None = params.get("variant_id") or None

        q = (
            self.db.query(Document)
            .filter(Document.status == "COMPLETED")
            .order_by(Document.created_at.desc())
        )
        if variant_id:
            q = q.filter(Document.variant_id == variant_id)
        docs = q.limit(last_n).all()

        if not docs:
            return {
                "docs_analysed": 0,
                "variants_seen": 0,
                "overall_health": None,
                "variants": [],
                "recommendations": ["No completed documents found — process some documents first."],
                "generated_at": datetime.utcnow().isoformat(),
            }

        # Bucket docs by variant
        by_variant: dict[str, list[Document]] = {}
        for d in docs:
            key = d.variant_id or "__none__"
            by_variant.setdefault(key, []).append(d)

        variant_results = []
        for vid, vdocs in by_variant.items():
            v = (
                self.db.query(DocumentVariant).filter(DocumentVariant.id == vid).first()
                if vid != "__none__" else None
            )
            schema_fields: list[str] = []
            if v and v.field_schema_json:
                try:
                    schema_fields = list(json.loads(v.field_schema_json).keys())
                except Exception:
                    pass

            confidences: list[float] = []
            correction_count = 0
            needs_review_count = 0
            method_counts: dict[str, int] = {}
            fields_found_per_doc: list[int] = []

            for d in vdocs:
                fields = (
                    self.db.query(ExtractedField)
                    .filter(ExtractedField.document_id == d.id)
                    .all()
                )
                scalar_fields = [f for f in fields if (getattr(f, "field_type", "scalar") or "scalar") != "table"]

                doc_confidences = [f.confidence for f in scalar_fields if f.confidence is not None]
                confidences.extend(doc_confidences)

                if any(f.human_corrected for f in scalar_fields):
                    correction_count += 1

                if d.status == "NEEDS_REVIEW":
                    needs_review_count += 1

                for f in scalar_fields:
                    method = f.extraction_method or "UNKNOWN"
                    method_counts[method] = method_counts.get(method, 0) + 1

                extracted_names = {f.field_name for f in scalar_fields if f.field_value}
                if schema_fields:
                    hit = len([n for n in schema_fields if n in extracted_names])
                    fields_found_per_doc.append(hit)

            doc_count = len(vdocs)
            avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
            correction_rate = correction_count / doc_count if doc_count else 0.0
            needs_review_rate = needs_review_count / doc_count if doc_count else 0.0

            # Field hit rate: how many schema fields were found per doc on average
            if schema_fields and fields_found_per_doc:
                field_hit_rate = (sum(fields_found_per_doc) / len(fields_found_per_doc)) / len(schema_fields)
            elif schema_fields:
                field_hit_rate = 0.0
            else:
                field_hit_rate = 1.0  # No schema = can't measure; don't penalise

            # Primary extraction method
            primary_method = max(method_counts, key=method_counts.get) if method_counts else "—"

            # Health score: confidence (40%) + hit rate (40%) + (1 - correction) (20%)
            health = int(
                (avg_confidence * 0.40 + field_hit_rate * 0.40 + (1 - correction_rate) * 0.20)
                * 100
            )
            health = max(0, min(100, health))

            # Recommendations
            recs: list[str] = []
            if avg_confidence < 0.70:
                recs.append(f"Low confidence ({avg_confidence:.0%}) — consider re-training or checking document quality")
            if schema_fields and field_hit_rate < 0.75:
                recs.append(f"Field hit rate {field_hit_rate:.0%} — some schema fields rarely extracted")
            if correction_rate > 0.30:
                recs.append(f"Correction rate {correction_rate:.0%} is high — review extraction accuracy")
            if not recs:
                recs.append("Extraction quality looks good")

            variant_results.append({
                "variant_id":         vid if vid != "__none__" else None,
                "variant_label":      _variant_label(v) if v else "No variant",
                "variant_key":        (v.variant_key or "") if v else "",
                "doc_count":          doc_count,
                "avg_confidence":     round(avg_confidence, 3),
                "avg_confidence_pct": int(avg_confidence * 100),
                "field_hit_rate":     round(field_hit_rate, 3),
                "field_hit_rate_pct": int(field_hit_rate * 100),
                "correction_rate":    round(correction_rate, 3),
                "correction_rate_pct": int(correction_rate * 100),
                "needs_review_rate":  round(needs_review_rate, 3),
                "primary_method":     primary_method,
                "schema_field_count": len(schema_fields),
                "health_score":       health,
                "recommendations":    recs,
            })

        # Overall health = weighted avg of variant health scores (by doc count)
        total_weight = sum(v["doc_count"] for v in variant_results)
        if total_weight > 0:
            overall_health = int(
                sum(v["health_score"] * v["doc_count"] for v in variant_results) / total_weight
            )
        else:
            overall_health = None

        # Top-level recommendations
        top_recs: list[str] = []
        flagged = [v for v in variant_results if v["health_score"] < 70]
        if flagged:
            top_recs.append(f"{len(flagged)} variant(s) below health threshold — see details below")
        good = [v for v in variant_results if v["health_score"] >= 80]
        if good:
            top_recs.append(f"{len(good)} variant(s) performing well (score ≥ 80)")
        if not top_recs:
            top_recs.append("All variants within acceptable quality range")

        # Sort variants: worst health first
        variant_results.sort(key=lambda v: v["health_score"])

        return {
            "docs_analysed":    len(docs),
            "variants_seen":    len(variant_results),
            "overall_health":   overall_health,
            "variants":         variant_results,
            "recommendations":  top_recs,
            "generated_at":     datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  PageScoreAuditAgent
# ─────────────────────────────────────────────────────────────────────────────

class PageScoreAuditAgent(BaseAgent):
    name = "page_score_audit"

    def run(self, params: dict) -> dict:
        variant_id: str | None = params.get("variant_id") or None

        q = self.db.query(DocumentVariant)
        if variant_id:
            q = q.filter(DocumentVariant.id == variant_id)
        variants = q.all()

        # Only keep variants with profiles
        profiled = [v for v in variants if getattr(v, "page_profile_json", None)]

        if not profiled:
            return {
                "variants_analysed":     0,
                "total_pages_skipping":  0,
                "total_cost_saved_usd":  0.0,
                "variants":              [],
                "recommendations":       ["No page profiles built yet — process more documents to build profiles."],
                "generated_at":          datetime.utcnow().isoformat(),
            }

        variant_results = []
        total_skipping = 0
        total_cost = 0.0

        for v in profiled:
            try:
                profile = json.loads(v.page_profile_json)
            except Exception:
                continue

            instances      = profile.get("instances_seen", 0)
            page_data      = profile.get("page_data", {})
            confident_skip = set(profile.get("confident_skip", []))
            cost_saved     = profile.get("cost_saved_usd", 0.0)
            tokens_saved   = profile.get("tokens_saved_estimate", 0)

            total_skipping += len(confident_skip)
            total_cost     += cost_saved

            # Build per-page rows
            pages: list[dict] = []
            for page_key, stats in sorted(page_data.items(), key=lambda x: int(x[0])):
                page_idx = int(page_key)
                present  = stats.get("present_in", 0)
                contrib  = stats.get("contributed_in", 0)
                rate     = (contrib / present) if present > 0 else 0.0
                is_skip  = page_idx in confident_skip

                status = "skipping" if is_skip else ("useful" if rate > 0 else "empty")
                pages.append({
                    "page_0indexed":       page_idx,
                    "page_1indexed":       page_idx + 1,
                    "contribution_rate":   round(rate, 2),
                    "contribution_pct":    int(rate * 100),
                    "present_in":          present,
                    "contributed_in":      contrib,
                    "status":              status,
                    "fields":              stats.get("fields_seen", []),
                })

            # Skip candidates: seen enough times, rate 0, not yet in skip list
            skip_candidates = [
                p for p in pages
                if p["contribution_rate"] == 0
                and p["present_in"] >= 3
                and p["page_0indexed"] not in confident_skip
            ]

            # Potential additional saving
            additional_tokens = len(skip_candidates) * AVG_TOKENS_PER_PAGE * instances
            additional_saving = round(additional_tokens * COST_PER_TOKEN, 4)

            if instances < 3:
                recommendation = "insufficient_data"
            elif skip_candidates:
                recommendation = "candidates_available"
            else:
                recommendation = "optimal"

            variant_results.append({
                "variant_id":            v.id,
                "variant_label":         _variant_label(v),
                "instances_seen":        instances,
                "current_skip_count":    len(confident_skip),
                "total_pages_tracked":   len(page_data),
                "cost_saved_usd":        round(cost_saved, 4),
                "tokens_saved_estimate": tokens_saved,
                "skip_candidates":       [p["page_1indexed"] for p in skip_candidates],
                "additional_saving_usd": additional_saving,
                "recommendation":        recommendation,
                "pages":                 pages,
            })

        variant_results.sort(key=lambda v: v["cost_saved_usd"], reverse=True)

        top_recs: list[str] = []
        candidates_total = sum(len(v["skip_candidates"]) for v in variant_results)
        if candidates_total > 0:
            top_recs.append(
                f"{candidates_total} additional skip candidate(s) identified — "
                "will be auto-learned with more document instances"
            )
        if len(profiled) > 0:
            top_recs.append(f"${total_cost:.3f} cost saved to date across {len(profiled)} variant(s)")
        insufficient = [v for v in variant_results if v["recommendation"] == "insufficient_data"]
        if insufficient:
            top_recs.append(
                f"{len(insufficient)} variant(s) need more documents to build a confident profile (min 3)"
            )
        if not top_recs:
            top_recs.append("All profiles are optimal")

        return {
            "variants_analysed":    len(variant_results),
            "total_pages_skipping": total_skipping,
            "total_cost_saved_usd": round(total_cost, 4),
            "variants":             variant_results,
            "recommendations":      top_recs,
            "generated_at":         datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  FieldUsageAuditAgent
# ─────────────────────────────────────────────────────────────────────────────

class FieldUsageAuditAgent(BaseAgent):
    name = "field_usage_audit"

    def run(self, params: dict) -> dict:
        last_n: int = int(params.get("last_n", 50))

        variants = self.db.query(DocumentVariant).all()
        with_schema = [v for v in variants if v.field_schema_json]

        if not with_schema:
            return {
                "variants_analysed":     0,
                "total_unused_fields":   0,
                "total_unreliable_fields": 0,
                "variants":              [],
                "recommendations":       ["No confirmed schemas found — confirm field schemas first via the Documents page."],
                "generated_at":          datetime.utcnow().isoformat(),
            }

        total_unused = 0
        total_unreliable = 0
        variant_results = []

        for v in with_schema:
            try:
                schema = json.loads(v.field_schema_json)
            except Exception:
                continue

            schema_fields = list(schema.keys())
            if not schema_fields:
                continue

            # Get last N completed docs for this variant
            docs = (
                self.db.query(Document)
                .filter(Document.variant_id == v.id, Document.status == "COMPLETED")
                .order_by(Document.created_at.desc())
                .limit(last_n)
                .all()
            )

            if not docs:
                continue

            # For each schema field, count extraction rate + avg confidence
            field_stats: dict[str, dict[str, Any]] = {
                f: {"found_count": 0, "confidences": []} for f in schema_fields
            }

            for d in docs:
                doc_fields = (
                    self.db.query(ExtractedField)
                    .filter(
                        ExtractedField.document_id == d.id,
                        (getattr(ExtractedField, "field_type", None) != "table") if True else True,
                    )
                    .all()
                )
                extracted_map = {
                    ef.field_name: ef for ef in doc_fields
                    if (getattr(ef, "field_type", "scalar") or "scalar") != "table"
                }

                for canonical in schema_fields:
                    # Check canonical name + any aliases from schema
                    schema_meta = schema.get(canonical, {})
                    aliases = schema_meta.get("aliases", [canonical]) if isinstance(schema_meta, dict) else [canonical]
                    match = next((extracted_map[a] for a in aliases if a in extracted_map), None)
                    if match and match.field_value:
                        field_stats[canonical]["found_count"] += 1
                        if match.confidence is not None:
                            field_stats[canonical]["confidences"].append(match.confidence)

            doc_count = len(docs)
            field_rows: list[dict] = []
            unused_fields: list[str] = []
            unreliable_fields: list[str] = []

            for fname in schema_fields:
                stats = field_stats[fname]
                found = stats["found_count"]
                rate = found / doc_count if doc_count > 0 else 0.0
                confs = stats["confidences"]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                if rate < 0.10:
                    status = "unused"
                    unused_fields.append(fname)
                elif avg_conf < 0.60 and doc_count >= 5:
                    status = "unreliable"
                    unreliable_fields.append(fname)
                elif rate >= 0.85 and avg_conf >= 0.80:
                    status = "healthy"
                else:
                    status = "moderate"

                field_rows.append({
                    "field":             fname,
                    "extraction_rate":   round(rate, 3),
                    "extraction_pct":    int(rate * 100),
                    "avg_confidence":    round(avg_conf, 3),
                    "avg_confidence_pct": int(avg_conf * 100),
                    "found_in":          found,
                    "docs_sampled":      doc_count,
                    "status":            status,
                })

            total_unused     += len(unused_fields)
            total_unreliable += len(unreliable_fields)

            recs: list[str] = []
            if unused_fields:
                recs.append(f"Consider removing: {', '.join(unused_fields[:5])} — extraction rate < 10%")
            if unreliable_fields:
                recs.append(f"Low confidence on: {', '.join(unreliable_fields[:3])} — may need prompt tuning")
            if not recs:
                recs.append("All schema fields are being extracted reliably")

            # Sort: unused first, then unreliable, then healthy
            status_order = {"unused": 0, "unreliable": 1, "moderate": 2, "healthy": 3}
            field_rows.sort(key=lambda r: status_order.get(r["status"], 9))

            variant_results.append({
                "variant_id":       v.id,
                "variant_label":    _variant_label(v),
                "schema_fields":    len(schema_fields),
                "docs_sampled":     doc_count,
                "unused_fields":    unused_fields,
                "unreliable_fields": unreliable_fields,
                "fields":           field_rows,
                "recommendations":  recs,
            })

        top_recs: list[str] = []
        if total_unused > 0:
            top_recs.append(
                f"{total_unused} field(s) never or rarely extracted across all variants — "
                "removing them will tighten prompts and improve accuracy"
            )
        if total_unreliable > 0:
            top_recs.append(f"{total_unreliable} field(s) extracted with low confidence — review prompt instructions")
        if not top_recs:
            top_recs.append("Schema health looks good across all variants")

        return {
            "variants_analysed":       len(variant_results),
            "total_unused_fields":     total_unused,
            "total_unreliable_fields": total_unreliable,
            "variants":                variant_results,
            "recommendations":         top_recs,
            "generated_at":            datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CostOptimisationReportAgent
# ─────────────────────────────────────────────────────────────────────────────

class CostOptimisationReportAgent(BaseAgent):
    name = "cost_optimisation_report"

    def run(self, params: dict) -> dict:  # noqa: ARG002
        # ── Basic document counts ──────────────────────────────────────────────
        total_docs = self.db.query(Document).count()
        completed_docs = self.db.query(Document).filter(Document.status == "COMPLETED").count()

        # Documents that have page metadata
        docs_with_meta = (
            self.db.query(Document)
            .filter(Document.pages_total.isnot(None))
            .all()
        )

        # ── Page-level stats ───────────────────────────────────────────────────
        total_pages_seen    = sum((d.pages_total or 0) for d in docs_with_meta)
        total_pages_skipped = sum((d.pages_skipped_count or 0) for d in docs_with_meta)
        total_pages_sampled = total_pages_seen - total_pages_skipped

        tokens_saved_estimate = total_pages_skipped * AVG_TOKENS_PER_PAGE
        cost_saved_usd        = round(tokens_saved_estimate * COST_PER_TOKEN, 4)

        # ── Variant profiles ───────────────────────────────────────────────────
        all_variants    = self.db.query(DocumentVariant).all()
        profiled        = [v for v in all_variants if getattr(v, "page_profile_json", None)]

        stage_counts: dict[str, int] = {}
        profile_cost_total = 0.0
        skip_candidates_potential = 0

        for v in profiled:
            try:
                profile = json.loads(v.page_profile_json)
            except Exception:
                continue

            instances  = profile.get("instances_seen", 0)
            skip_list  = profile.get("confident_skip", [])
            page_data  = profile.get("page_data", {})
            profile_cost_total += profile.get("cost_saved_usd", 0.0)

            # Stage
            if not skip_list:
                stage = "LEARNING" if instances >= 1 else "ZERO_SHOT"
            else:
                stage = "LEARNED"
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

            # Candidates
            for page_key, stats in page_data.items():
                if (
                    int(page_key) not in set(skip_list)
                    and stats.get("present_in", 0) >= 3
                    and stats.get("contributed_in", 0) == 0
                ):
                    skip_candidates_potential += 1

        # Variants not yet profiled
        not_profiled = len(all_variants) - len(profiled)
        if not_profiled > 0:
            stage_counts["NO_PROFILE"] = stage_counts.get("NO_PROFILE", 0) + not_profiled

        # ── Cost per doc estimates ─────────────────────────────────────────────
        # Baseline: if we passed all pages (no skipping)
        avg_pages_per_doc = (total_pages_seen / len(docs_with_meta)) if docs_with_meta else 0
        baseline_cost_per_doc = round(avg_pages_per_doc * AVG_TOKENS_PER_PAGE * COST_PER_TOKEN, 4)

        # Actual: after skipping
        avg_pages_sampled = (total_pages_sampled / len(docs_with_meta)) if docs_with_meta else 0
        actual_cost_per_doc = round(avg_pages_sampled * AVG_TOKENS_PER_PAGE * COST_PER_TOKEN, 4)

        pct_reduction = int(
            (1 - actual_cost_per_doc / baseline_cost_per_doc) * 100
        ) if baseline_cost_per_doc > 0 else 0

        # ── Weekly doc average ─────────────────────────────────────────────────
        # Very rough: if we have 4 weeks of data, divide by 4
        weekly_docs_avg = max(1, completed_docs // 4) if completed_docs >= 4 else completed_docs

        # Annual projection: at current savings rate
        annual_saving = round(cost_saved_usd / max(len(docs_with_meta), 1) * weekly_docs_avg * 52, 2)

        # Additional potential saving from skip candidates
        additional_saving = round(skip_candidates_potential * AVG_TOKENS_PER_PAGE * COST_PER_TOKEN * 10, 4)

        return {
            "total_docs_processed":      total_docs,
            "completed_docs":            completed_docs,
            "docs_with_page_meta":       len(docs_with_meta),
            "total_variants":            len(all_variants),
            "variants_with_profiles":    len(profiled),
            "variants_by_stage":         stage_counts,
            "total_pages_seen":          total_pages_seen,
            "total_pages_skipped":       total_pages_skipped,
            "total_pages_sampled":       total_pages_sampled,
            "tokens_saved_estimate":     tokens_saved_estimate,
            "cost_saved_usd":            cost_saved_usd,
            "avg_pages_per_doc":         round(avg_pages_per_doc, 1),
            "avg_pages_sampled":         round(avg_pages_sampled, 1),
            "baseline_cost_per_doc_usd": baseline_cost_per_doc,
            "actual_cost_per_doc_usd":   actual_cost_per_doc,
            "pct_reduction":             pct_reduction,
            "weekly_docs_avg":           weekly_docs_avg,
            "annual_projected_saving_usd": annual_saving,
            "skip_candidate_count":      skip_candidates_potential,
            "skip_candidate_potential_saving_usd": additional_saving,
            "generated_at":              datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "extraction_quality_audit":  ExtractionQualityAuditAgent,
    "page_score_audit":          PageScoreAuditAgent,
    "field_usage_audit":         FieldUsageAuditAgent,
    "cost_optimisation_report":  CostOptimisationReportAgent,
}


def get_agent(name: str, db: Session) -> BaseAgent | None:
    cls = AGENT_REGISTRY.get(name)
    return cls(db) if cls else None
