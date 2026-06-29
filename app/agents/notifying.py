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
import logging
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from sqlalchemy.orm import Session

from app.agents.base import BaseAgent
from app.models.document import Document
from app.models.extracted_field import ExtractedField
from app.models.shipment import ShipmentRecord

logger = logging.getLogger(__name__)


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


    def notify_failure(
        self,
        doc: Document,
        error_reason: str | None = None,
    ) -> None:
        """
        Fire failure notifications when a document transitions to NEEDS_REVIEW.

        Channels (all best-effort — never block the pipeline):
          1. Email to doc.submitter_email   (requires SMTP_HOST in config)
          2. Teams adaptive card            (requires TEAMS_WEBHOOK_URL in config)
        """
        from app.config import settings

        if not settings.failure_notifications_enabled:
            return

        reason = error_reason or doc.error_reason or "No reason recorded."

        # ── Channel 1: email to submitter ─────────────────────────────────────
        if settings.smtp_host and doc.submitter_email:
            try:
                _send_failure_email(doc, reason, settings)
                logger.info(
                    "doc %s: failure email sent to %s", doc.id, doc.submitter_email
                )
            except Exception as exc:
                logger.warning(
                    "doc %s: failure email to %s failed: %s", doc.id, doc.submitter_email, exc
                )

        # ── Channel 2: Teams webhook ──────────────────────────────────────────
        if settings.teams_webhook_url:
            try:
                _post_teams_failure_card(doc, reason, settings.teams_webhook_url)
                logger.info("doc %s: Teams failure card posted", doc.id)
            except Exception as exc:
                logger.warning("doc %s: Teams webhook failed: %s", doc.id, exc)


def fire_failure_notifications(db: Session, doc: Document) -> None:
    """
    Module-level helper called from base.needs_review() and runner._set_needs_review().
    Instantiates NotifyingAgent and fires notify_failure() best-effort.
    """
    try:
        agent = NotifyingAgent(db)
        agent.notify_failure(doc, error_reason=doc.error_reason)
    except Exception as exc:
        logger.warning("fire_failure_notifications for %s failed: %s", doc.id, exc)


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


def _send_failure_email(doc: Document, reason: str, settings) -> None:
    """
    Send a plain-text + HTML failure notification email to doc.submitter_email.
    Uses STARTTLS by default (smtp_use_ssl=False); set smtp_use_ssl=True for port 465 SSL.
    """
    subject = f"[Dokr] Document needs review — {doc.file_name}"

    dashboard_url = f"http://localhost:8000/dashboard/docs/{doc.id}"

    plain = (
        f"Hi,\n\n"
        f"The document you submitted to Dokr could not be processed automatically "
        f"and requires a human decision.\n\n"
        f"File:     {doc.file_name}\n"
        f"Doc ID:   {doc.id}\n"
        f"Reason:   {reason}\n\n"
        f"Please log in to the Dokr dashboard to review it:\n"
        f"{dashboard_url}\n\n"
        f"— Dokr"
    )

    html = f"""<html><body style="font-family:sans-serif;color:#1a1a1a;max-width:560px;">
<h2 style="color:#d97706;">⚠ Document needs review</h2>
<p>The document you submitted to Dokr could not be processed automatically
and requires a human decision.</p>
<table style="border-collapse:collapse;width:100%;margin:16px 0;">
  <tr><td style="padding:6px 12px 6px 0;color:#666;white-space:nowrap;font-size:13px;">File</td>
      <td style="padding:6px 0;font-size:13px;font-weight:600;">{doc.file_name}</td></tr>
  <tr><td style="padding:6px 12px 6px 0;color:#666;white-space:nowrap;font-size:13px;">Doc ID</td>
      <td style="padding:6px 0;font-size:13px;font-family:monospace;">{doc.id}</td></tr>
  <tr><td style="padding:6px 12px 6px 0;color:#666;white-space:nowrap;vertical-align:top;font-size:13px;">Reason</td>
      <td style="padding:6px 0;font-size:13px;color:#b45309;">{reason[:300]}</td></tr>
</table>
<a href="{dashboard_url}"
   style="display:inline-block;background:#d97706;color:#fff;padding:10px 20px;
          border-radius:6px;text-decoration:none;font-weight:600;font-size:13px;">
  Review in Dokr →
</a>
<p style="margin-top:24px;font-size:12px;color:#999;">— Dokr automated notification</p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = settings.smtp_from
    msg["To"]      = doc.submitter_email

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    if settings.smtp_use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context) as server:
            if settings.smtp_user and settings.smtp_password:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_from, doc.submitter_email, msg.as_string())
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.ehlo()
            server.starttls()
            if settings.smtp_user and settings.smtp_password:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_from, doc.submitter_email, msg.as_string())


def _post_teams_failure_card(doc: Document, reason: str, webhook_url: str) -> None:
    """
    Post a simple Teams Adaptive Card message via an incoming webhook.
    Uses httpx (already a dependency for the notifying agent).
    """
    import httpx

    dashboard_url = f"http://localhost:8000/dashboard/docs/{doc.id}"

    # Microsoft Teams incoming webhook expects a MessageCard or AdaptiveCard payload.
    # MessageCard is simpler and works in all Teams versions.
    card = {
        "@type":       "MessageCard",
        "@context":    "https://schema.org/extensions",
        "themeColor":  "d97706",
        "summary":     f"Dokr: {doc.file_name} needs review",
        "sections": [
            {
                "activityTitle":    "⚠ Document needs review",
                "activitySubtitle": f"File: **{doc.file_name}**",
                "facts": [
                    {"name": "Doc ID",   "value": doc.id},
                    {"name": "Class",    "value": doc.document_class_id or "—"},
                    {"name": "Reason",   "value": reason[:300]},
                    {"name": "Submitter","value": doc.submitter_email or "—"},
                ],
                "markdown": True,
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name":  "Review in Dokr →",
                "targets": [{"os": "default", "uri": dashboard_url}],
            }
        ],
    }

    httpx.post(
        webhook_url,
        json=card,
        headers={"Content-Type": "application/json"},
        timeout=10.0,
    )


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
