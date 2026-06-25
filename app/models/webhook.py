"""
WebhookSubscription — persistent endpoint registrations.

When a subscription is active, the NotifyingAgent fans out the
document.completed event (and future events) to every matching
subscription URL in addition to the per-document webhook_url.

Events are a list of strings; an empty list means "all events".
Currently supported event names:
  document.completed     — document reached COMPLETED state
  document.needs_review  — document routed to NEEDS_REVIEW
  document.failed        — document reached FAILED state
  match.fail             — three-way match failed on a shipment
  match.pass             — three-way match passed
  shipment.complete      — all documents in a shipment are terminal

secret_key is an optional HMAC-SHA256 signing secret.
When set, the NotifyingAgent adds an X-Dokr-Signature header
to each delivery:  sha256=<hex(HMAC(secret, body))>
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.database import Base

SUPPORTED_EVENTS = {
    "document.completed",
    "document.needs_review",
    "document.failed",
    "match.fail",
    "match.pass",
    "shipment.complete",
}


class WebhookSubscription(Base):
    __tablename__ = "webhook_subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Destination
    url:        Mapped[str]       = mapped_column(String, nullable=False)
    events:     Mapped[list]      = mapped_column(JSON, nullable=False, default=list)
    # Empty list = subscribe to ALL supported events

    # Auth / security
    secret_key: Mapped[str | None] = mapped_column(String, nullable=True)
    # HMAC-SHA256 signing secret — stored as plaintext for scaffold.
    # In production: encrypt at rest (Azure Key Vault / AWS Secrets Manager).

    # Status
    active:     Mapped[bool]      = mapped_column(Boolean, default=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_by:  Mapped[str | None] = mapped_column(String, nullable=True)

    # Delivery tracking (updated on every delivery attempt)
    last_delivery_at:     Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_delivery_status: Mapped[int | None]      = mapped_column(Integer, nullable=True)
    last_delivery_detail: Mapped[str | None]      = mapped_column(String, nullable=True)
    total_deliveries:     Mapped[int]             = mapped_column(Integer, default=0)
    total_failures:       Mapped[int]             = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
