"""
Notifying Agent — fires webhook callbacks on document completion.

Delivery model
──────────────
Two independent channels fire in parallel after a document reaches COMPLETED:

  1. Per-document webhook_url (set at submit time via POST /submit?webhook_url=…)
     Fires a single POST with the full document.completed payload.

  2. Subscription fan-out (webhook_subscriptions table, managed via /webhooks/ CRUD)
     Fires the same payload to every active subscription whose events list
     includes "document.completed" (or is empty = subscribe to all events).

Both channels are best-effort: a delivery failure logs to PipelineEvent but
does NOT block the document reaching COMPLETED.

Webhook payload (application/json POST)
────────────────────────────────────────
  {
    "event":              "document.completed",
    "document_id":        "doc_01KVKDGWJ3…",
    "status":             "COMPLETED",
    "document_class":     "dc_006",
    "document_class_name":"Supplier Invoice",
    "variant_key":        "danieli.com",
    "file_name":          "INV-25206544.pdf",
    "field_count":        17,
    "avg_confidence":     0.9241,
    "sharepoint_path":    "/Shared Documents/Dokr/FY2026/…",
    "erp_reference":      "ERP-DC006-A1B2C3D4",
    "match_result":       "PASS",
    "shipment_id":        "shp_01KVKDGWJ3…",
    "created_at":         "2026-06-20T14:33:00Z",
    "completed_at":       "2026-06-20T14:33:05Z"
  }

Security: subscriptions with a secret_key receive an additional
  X-Dokr-Signature: sha256=<hex(HMAC-SHA256(secret, body))>
header.

httpx is used for all HTTP calls (sync, matches the synchronous pipeline runner).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.models.extracted_field import ExtractedField
from app.models.shipment import ShipmentRecord


@dataclass
class NotifyResult:
    webhook_fired: bool
    status_code: int | None
    detail: str


class NotifyingAgent(BaseAgent):
    """
    Builds and fires the document.completed webhook payload to:
      - the per-document webhook_url (if set at submit)
      - all matching active webhook subscriptions
    """

    name = "NotifyingAgent"

    def notify(
        self,
        doc: Document,
        sharepoint_path: str | None = None,
        erp_reference: str | None = None,
    ) -> NotifyResult:
        # ── Build payload ─────────────────────────────────────────────────────
        fields = (
            self.db.query(ExtractedField)
            .filter(ExtractedField.document_id == doc.id)
            .all()
        )
        field_count = len(fields)
        avg_conf = (
            round(sum(f.confidence for f in fields) / field_count, 4)
            if fields else None
        )

        match_result = None
        shp_erp_ref = erp_reference
        if doc.shipment_id:
            shipment = (
                self.db.query(ShipmentRecord)
                .filter(ShipmentRecord.id == doc.shipment_id)
                .first()
            )
            if shipment:
                match_result = shipment.match_result
                if not shp_erp_ref:
                    shp_erp_ref = shipment.erp_reference

        payload = {
            "event":               "document.completed",
            "document_id":         doc.id,
            "status":              "COMPLETED",
            "document_class":      doc.document_class_id,
            "document_class_name": doc.document_class.name if doc.document_class else None,
            "variant_key":         doc.variant_key,
            "file_name":           doc.file_name,
            "field_count":         field_count,
            "avg_confidence":      avg_conf,
            "sharepoint_path":     sharepoint_path,
            "erp_reference":       shp_erp_ref,
            "match_result":        match_result,
            "shipment_id":         doc.shipment_id,
            "created_at":          doc.created_at.isoformat() + "Z",
            "completed_at":        datetime.utcnow().isoformat() + "Z",
        }

        details: list[str] = []

        # ── Channel 1: per-document webhook_url ───────────────────────────────
        if doc.webhook_url:
            fired, status, detail = _post_json(doc.webhook_url, payload)
            details.append(f"per-doc webhook → {detail}")
        else:
            details.append("per-doc webhook → none configured")

        # ── Channel 2: subscription fan-out ───────────────────────────────────
        sub_count = _fan_out_subscriptions(self.db, "document.completed", payload)
        if sub_count:
            details.append(f"subscription fan-out → {sub_count} subscription(s) notified")
        else:
            details.append("subscription fan-out → no active subscriptions")

        return NotifyResult(
            webhook_fired=bool(doc.webhook_url),
            status_code=status if doc.webhook_url else None,
            detail=". ".join(details) + ".",
        )


def fire_event(db: Session, event_name: str, payload: dict) -> None:
    """
    Public helper: fire an arbitrary named event to all matching subscriptions.
    Used outside the NotifyingAgent (e.g. for match.fail, shipment.complete events).
    """
    _fan_out_subscriptions(db, event_name, payload)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict) -> tuple[bool, int | None, str]:
    """POST payload as JSON to url. Returns (success, status_code, detail)."""
    try:
        import httpx
        resp = httpx.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Dokr/1.0",
                "X-Dokr-Event": payload.get("event", ""),
            },
            timeout=10.0,
        )
        if resp.is_success:
            return True, resp.status_code, f"HTTP {resp.status_code}"
        return False, resp.status_code, f"HTTP {resp.status_code} (non-2xx)"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _fan_out_subscriptions(db: Session, event_name: str, payload: dict) -> int:
    """
    Deliver payload to all active subscriptions matching event_name.
    Returns the count of subscriptions notified (regardless of delivery success).
    Failures are silently swallowed — never block the pipeline.
    """
    try:
        from app.routers.webhooks import _deliver
        from app.models.webhook import WebhookSubscription

        subs = db.query(WebhookSubscription).filter(
            WebhookSubscription.active == True
        ).all()

        notified = 0
        for sub in subs:
            # Empty events list = subscribe to all events
            if sub.events and event_name not in sub.events:
                continue
            try:
                _deliver(sub, payload, update_stats=True, db=db)
                notified += 1
            except Exception:
                pass
        return notified
    except Exception:
        return 0
